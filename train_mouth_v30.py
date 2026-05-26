#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim
from gaussian_renderer import render, render_motion, render_motion_mouth
import sys
from scene import Scene, GaussianModel, MouthMotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import loadCamOnTheFly
from utils.soft_mask_utils import build_soft_mouth_masks
from models.cross_attn_driver import GaussianCrossAttnDriver
from train_fuse_v18 import render_motion_mouth_v18
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# NOTE (2026-05-14): V31.x box-penalty experiments removed. V31.1 wall-piled at
# z=0.065, V31.2 wall-piled at z=0.0565, V31.3 collapsed to single point at
# z=0.0345. Root cause of "FixAC has teeth but post-FixAC doesn't" is Fix B
# (5/12) commenting out the violent green-bg prune. Restore upstream FixAC
# behavior: violent prune ON (lines ~587 below), no extra geometric constraint.


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations]
    # V30: full 50k retrain from scratch. Default densify_until_iter applies;
    # cavity_idx is recomputed after each densify_and_prune (Gaussian count
    # changes) so the dual-head split stays valid throughout training.

    # vars
    warm_step = 3000
    bg_iter = opt.iterations-1000 # opt.densify_until_iter
    lpips_start_iter = bg_iter
    motion_stop_iter = bg_iter
    mouth_select_iter = bg_iter - 10000
    mouth_step = 1 / mouth_select_iter
    select_interval = 7

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    motion_net = MouthMotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / bg_iter))

    # V30 change C: split cross-attn into cavity-only and lip-only heads.
    # V30 fix v7 (2026-05-11): residual_scale increased 5x. Original lip 1e-3
    # / cavity 2e-3 caps displacement at ~1-2mm canonical. This was enough for
    # Macron (max cavity ~1224 px, narrow opening range) but tanh-saturates on
    # subjects with wider mouth-opening (Obama max=1691, May=1301). Probe data:
    # pred cavity brightness was constant ~79 across train+test regardless of
    # GT (varies 32-142) → motion network couldn't learn audio→displacement
    # mapping because output was capped. Fix: 5x larger range so the cap
    # doesn't bind for cross-subject. lip 5e-3 / cavity 1e-2.
    audio_dim_m = motion_net.audio_dim
    cross_attn_driver_lip = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net.in_dim,
        audio_seq_len=8, audio_dim=audio_dim_m, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=1.2e-2, residual_scale_scale=5e-3, residual_scale_rot=5e-3,  # R3 (5/20): 5e-3→1.2e-2 motion budget
    ).cuda()
    cross_attn_driver_cavity = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net.in_dim,
        audio_seq_len=8, audio_dim=audio_dim_m, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=2.0e-2, residual_scale_scale=1e-2, residual_scale_rot=1e-2,  # R3 (5/20): 1e-2→2e-2 motion budget
    ).cuda()
    # E2 (2026-05-20): mouth-stage audio/AU-conditioned RGB residual.
    # Trained for full 50k iters alongside mouth Gaussians (vs D26's 15k-only fuse
    # window which proved too short). residual_scale=0.35 lets cavity swing from
    # canonical pink (~0.40) to dark cavity (~0.05).
    from models.v12_heads import PerGaussianAlbedoMLP
    albedo_head_mouth = PerGaussianAlbedoMLP(
        audio_dim=audio_dim_m, au_dim=17, hidden=128, residual_scale=0.35,
    ).cuda()

    attn_optimizer = torch.optim.Adam(
        list(cross_attn_driver_lip.parameters()) + list(cross_attn_driver_cavity.parameters())
        + list(albedo_head_mouth.parameters()),
        lr=5e-4, weight_decay=1e-6,
    )
    # Backward-compat alias used by the resume code path below.
    cross_attn_driver = cross_attn_driver_lip
    print(f"[v30 v7] mouth cross_attn dual heads: lip(au17,1.2e-2) + cavity(au25/26,2.0e-2) [R3]")
    print(f"[E2] albedo_head_mouth attached (residual_scale=0.35, audio_dim={audio_dim_m})")
    # AU index mask: cavity head sees only AU25 (idx 14) + AU26 (idx 15), zero
    # everything else. au_id_order = [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45].
    _CAVITY_AU_MASK = torch.zeros(17, device='cuda')
    _CAVITY_AU_MASK[14] = 1.0  # AU25
    _CAVITY_AU_MASK[15] = 1.0  # AU26

    # AU sliding window full array.
    # R-DATA-5: read TG_AU_CSV so cross_attn AU17 source matches dataset_readers'
    # corrected AU (motion_net path). Without this, cross_attn sees raw while
    # motion_net sees corrected → conflicting AU signal across the two paths.
    au_full = None
    try:
        _au_csv_name = os.environ.get('TG_AU_CSV', 'au.csv')
        df = pd.read_csv(os.path.join(dataset.source_path, _au_csv_name))
        print(f"[R-DATA-5] au_full loaded from {_au_csv_name}")
        ids = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        cols = []
        for _i in ids:
            _k = ' AU' + str(_i).zfill(2) + '_r'
            _v = df[_k].values if _k in df.columns else df[_k.strip()].values
            if _i == 45: _v = _v.clip(0, 2)
            cols.append(_v[:, None])
        au_full = torch.from_numpy(np.concatenate(cols, axis=-1).astype(np.float32)).cuda()
        print(f"[v3] AU full loaded T={au_full.shape[0]}")
    except Exception as _e:
        print(f"[v3] AU pre-load failed: {_e}")

    AU_WINDOW_T = 8

    # A+C (2026-05-18): apex-aware loss reweighting — load mouth_area per frame.
    # Loss for frame t multiplied by (1 + apex_boost * (ma_t / ma_max)).
    # Effect: apex frames (large mouth_area) dominate gradient → motion_net forced
    # to learn extreme lip-opening motions. Default boost=3.0 → apex frames have
    # 4x loss vs neutral frames.
    APEX_BOOST = 3.0
    apex_ma_arr = None
    apex_ma_max = 1.0
    try:
        _ma_path = os.path.join(dataset.source_path, 'mouth_area_per_frame.npy')
        if os.path.exists(_ma_path):
            apex_ma_arr = np.load(_ma_path)
            apex_ma_max = float(apex_ma_arr[apex_ma_arr > 0].max())
            print(f"[A+C] apex weighting enabled, ma_max={apex_ma_max:.0f}, boost={APEX_BOOST}")
    except Exception as _e:
        print(f"[A+C] mouth_area load failed: {_e}")

    # Q1 (2026-05-19): load audio→lip-landmark predictor (frozen).
    # For each training frame, predict 20 lip landmarks from audio window,
    # pass to motion_net as additional condition. No GT landmark at inference.
    q1_predictor = None
    q1_aud_window_path = None
    q1_aud_arr = None
    q1_window = 3
    try:
        from models.audio_to_landmark import AudioToLipLandmark
        _q1_ckpt = 'models/ckpts/audio_to_landmark_obama.pth'
        if os.path.exists(_q1_ckpt):
            _q1_state = torch.load(_q1_ckpt)
            q1_window = _q1_state.get('window', 3)
            q1_predictor = AudioToLipLandmark(window=q1_window).cuda()
            q1_predictor.load_state_dict(_q1_state['model'])
            q1_predictor.eval()
            for _p in q1_predictor.parameters(): _p.requires_grad = False
            # pre-load HuBERT array for window indexing
            q1_aud_arr = np.load(os.path.join(dataset.source_path, 'aud_hu.npy')).astype(np.float32)
            print(f"[Q1] landmark predictor loaded, test_L1={_q1_state.get('test_l1','?'):.3f}, window={q1_window}")
    except Exception as _e:
        print(f"[Q1] predictor load failed: {_e}")
        q1_predictor = None

    def _gather_landmark_for_fid(_fid):
        """Predict landmark for given fid from audio window, returns (20,2) tensor."""
        if q1_predictor is None or q1_aud_arr is None: return None
        w = q1_window
        N = q1_aud_arr.shape[0]
        lo = max(0, _fid - w); hi = min(N, _fid + w + 1)
        l_pad = max(0, w - _fid); r_pad = max(0, _fid + w + 1 - N)
        seq = q1_aud_arr[lo:hi]
        if l_pad > 0: seq = np.concatenate([np.tile(seq[:1], (l_pad, 1, 1)), seq], axis=0)
        if r_pad > 0: seq = np.concatenate([seq, np.tile(seq[-1:], (r_pad, 1, 1))], axis=0)
        seq_t = torch.from_numpy(seq).unsqueeze(0).cuda()   # (1, T, 2, 1024)
        with torch.no_grad():
            lmk_pred = q1_predictor(seq_t)[0]                # (20, 2)
        return lmk_pred

    # R3 (2026-05-20): audio→mouth_area predictor — amplifies motion when audio
    # implies mouth opening, suppresses otherwise. Multiplies (d_xyz, d_rot, d_scale)
    # by amp = 0.7 + 0.6 * ma_norm  ∈ [0.7, 1.3]   (ma_norm ∈ [0,1]).
    r3_predictor = None
    r3_aud_arr = None
    r3_window = 3
    try:
        from models.audio_to_mouth_area import AudioToMouthArea
        _r3_ckpt = 'models/ckpts/audio_to_mouth_area_obama.pth'
        if os.path.exists(_r3_ckpt):
            _r3_state = torch.load(_r3_ckpt)
            r3_window = _r3_state.get('window', 3)
            r3_predictor = AudioToMouthArea(window=r3_window).cuda()
            r3_predictor.load_state_dict(_r3_state['model'])
            r3_predictor.eval()
            for _p in r3_predictor.parameters(): _p.requires_grad = False
            if q1_aud_arr is not None:
                r3_aud_arr = q1_aud_arr   # reuse the already-loaded HuBERT array
            else:
                r3_aud_arr = np.load(os.path.join(dataset.source_path, 'aud_hu.npy')).astype(np.float32)
            print(f"[R3] mouth_area predictor loaded, test_L1={_r3_state.get('test_l1','?'):.3f}, window={r3_window}")
    except Exception as _e:
        print(f"[R3] predictor load failed: {_e}")
        r3_predictor = None

    def _gather_amp_for_fid(_fid):
        if r3_predictor is None or r3_aud_arr is None: return None
        w = r3_window; N = r3_aud_arr.shape[0]
        lo = max(0, _fid - w); hi = min(N, _fid + w + 1)
        l_pad = max(0, w - _fid); r_pad = max(0, _fid + w + 1 - N)
        seq = r3_aud_arr[lo:hi]
        if l_pad > 0: seq = np.concatenate([np.tile(seq[:1], (l_pad, 1, 1)), seq], axis=0)
        if r_pad > 0: seq = np.concatenate([seq, np.tile(seq[-1:], (r_pad, 1, 1))], axis=0)
        seq_t = torch.from_numpy(seq).unsqueeze(0).cuda()
        with torch.no_grad():
            ma_norm = r3_predictor(seq_t)[0]
        return ma_norm

    # R2 (2026-05-19): lip y-coord supervision via 2D landmark projection.
    # Force mouth Gaussians' projected 2D y to match GT lip y per frame.
    # This teaches cross_attn (94% of motion) to produce GT-aligned lip motion,
    # not just any motion that minimises L1.
    R2_LIP_Y_W = 1.0    # R3 (5/20): 10.0→1.0 to not dominate L1+SSIM gradient (R2 was making texture blurry)
    r2_gt_lip_y = None
    r2_upper_idx = None
    r2_lower_idx = None
    try:
        _p = os.path.join(dataset.source_path, 'gt_lip_y_inner.npy')
        if os.path.exists(_p):
            r2_gt_lip_y = torch.from_numpy(np.load(_p).astype(np.float32)).cuda()  # (N, 2)
            print(f"[R2] GT lip y loaded, shape={tuple(r2_gt_lip_y.shape)}, weight={R2_LIP_Y_W}")
    except Exception as _e:
        print(f"[R2] GT lip y load failed: {_e}")

    def _r2_project_to_2d(xyz, P, H, W):
        """Project (N,3) world xyz to (N,2) image (row, col)."""
        homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)  # (N,4)
        clip = homo @ P
        w = clip[:, 3:4].clamp_min(1e-6)
        ndc = clip[:, :3] / w
        col = (ndc[:, 0] * 0.5 + 0.5) * W
        row = (1 - (ndc[:, 1] * 0.5 + 0.5)) * H
        return row, col   # both (N,)

    def _gather_au(_img_id):
        if au_full is None or _img_id < 0:
            return torch.zeros(17, device="cuda")
        _half = AU_WINDOW_T // 2
        _lo = _img_id - _half; _hi = _img_id + (AU_WINDOW_T - _half)
        _l_pad = max(0, -_lo); _r_pad = max(0, _hi - au_full.shape[0])
        _lo_c = max(0, _lo); _hi_c = min(au_full.shape[0], _hi)
        _seq = au_full[_lo_c:_hi_c]
        if _l_pad: _seq = torch.cat([_seq[:1].expand(_l_pad, -1), _seq], dim=0)
        if _r_pad: _seq = torch.cat([_seq, _seq[-1:].expand(_r_pad, -1)], dim=0)
        return _seq.float()

    lpips_criterion = lpips.LPIPS(net='vgg').eval().cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        raw = torch.load(checkpoint)
        model_params, motion_params, motion_optimizer_params = raw[0], raw[1], raw[2]
        # V30: ignore the saved iter so the finetune actually runs
        # opt.iterations more steps (otherwise range(50000, 5001) is empty).
        first_iter = 0
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)
        if len(raw) >= 5 and isinstance(raw[4], dict):
            extras = raw[4]
            if 'cross_attn_driver_lip' in extras and 'cross_attn_driver_cavity' in extras:
                cross_attn_driver_lip.load_state_dict(extras['cross_attn_driver_lip'])
                cross_attn_driver_cavity.load_state_dict(extras['cross_attn_driver_cavity'])
                print('[v30] resumed both cross_attn driver heads from ckpt')
            elif 'cross_attn_driver' in extras:
                # V29a-style ckpt: load same single driver into both heads as warm-start.
                cross_attn_driver_lip.load_state_dict(extras['cross_attn_driver'])
                cross_attn_driver_cavity.load_state_dict(extras['cross_attn_driver'])
                print('[v30] V29a ckpt: cloned single cross_attn into both heads')

    feat_anchor_init = None

    # V30 fix v3 (2026-05-10): tried bg=[0,0,0] black to avoid green color
    # drift, but black bg removed densification gradient signal (rasterizer
    # clear=0 trivially matches GT outside mouth_overlap) → Gaussian count
    # collapsed from 6k to 2k by iter 10k. REVERTED to original green bg.
    # Macron compatibility preserved: identical to TG original mouth-stage.
    bg_color = [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # V30 change C support: compute cavity_idx by projecting all mouth
    # Gaussians to the most-cavity-revealing training frame and checking which
    # ones land inside cavity_mask. Re-callable so we can refresh after every
    # densify_and_prune in a 50k-iter retrain.
    import glob as _glob
    _best_cam_cache = {'cam': None, 'cav_t': None, 'fid': -1, 'pix': -1}

    def _resolve_best_cavity_cam():
        if _best_cam_cache['cam'] is not None:
            return _best_cam_cache['cam'], _best_cam_cache['cav_t'], \
                   _best_cam_cache['fid'], _best_cam_cache['pix']
        _cav_files = _glob.glob(os.path.join(dataset.source_path, 'cavity_mask', '*.npy'))
        best_fid = -1; best_cav = -1
        for _f in _cav_files:
            _fid = int(os.path.splitext(os.path.basename(_f))[0])
            _s = int(np.load(_f).sum())
            if _s > best_cav:
                best_cav = _s; best_fid = _fid
        if best_fid < 0 or best_cav <= 32:
            return None, None, -1, -1
        best_cam = None
        for _cams in (scene.getTrainCameras(), scene.getTestCameras()):
            for _cam in _cams:
                if int(_cam.talking_dict.get('img_id', -1)) == best_fid:
                    best_cam = _cam; break
            if best_cam is not None:
                break
        if best_cam is None:
            return None, None, -1, -1
        if best_cam.original_image is None:
            best_cam = loadCamOnTheFly(copy.deepcopy(best_cam))
        cav_t = torch.as_tensor(best_cam.talking_dict['cavity_mask']).cuda().bool()
        _best_cam_cache.update({'cam': best_cam, 'cav_t': cav_t,
                                 'fid': best_fid, 'pix': best_cav})
        return best_cam, cav_t, best_fid, best_cav

    def _compute_cavity_idx(verbose=False):
        try:
            best_cam, cav_t, best_fid, best_cav = _resolve_best_cavity_cam()
            if best_cam is None:
                return None
            xyz = gaussians.get_xyz.detach()
            ones = torch.ones(xyz.shape[0], 1, device=xyz.device)
            homog = torch.cat([xyz, ones], dim=1)
            ndc = homog @ best_cam.full_proj_transform.cuda()
            ndc = ndc[:, :3] / ndc[:, 3:4].clamp_min(1e-6)
            px = ((ndc[:, 0] + 1) * best_cam.image_width  * 0.5).long()
            py = ((ndc[:, 1] + 1) * best_cam.image_height * 0.5).long()
            H, W = cav_t.shape
            valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            px_clipped = px.clamp(0, W - 1)
            py_clipped = py.clamp(0, H - 1)
            hit = cav_t[py_clipped, px_clipped]
            ci = valid & hit
            if verbose:
                print(f"[v30] cavity_idx: {int(ci.sum().item())}/{xyz.shape[0]} "
                      f"(cam fid={best_fid}, cav_pixels={best_cav})")
            return ci
        except Exception as _e:
            print(f"[v30] cavity_idx compute failed ({_e})")
            return None

    cavity_idx = _compute_cavity_idx(verbose=True)

    # J1 (2026-05-18): temporal smoothness — pre-build {fid: cam} map so we can
    # look up adjacent-frame audio cheaply during training. Used only if
    # TEMPORAL_SMOOTH_W > 0. Adjacent-frame motion_net forward + L1 penalty
    # damps high-freq jitter without affecting inference speed.
    TEMPORAL_SMOOTH_W = 0.5
    _cam_by_fid = {}
    try:
        for _c in scene.getTrainCameras():
            _f = int(_c.talking_dict.get('img_id', -1))
            if _f >= 0 and _f not in _cam_by_fid:
                _cam_by_fid[_f] = _c
        print(f"[J1] temporal smoothness enabled, weight={TEMPORAL_SMOOTH_W}, "
              f"fid map size={len(_cam_by_fid)}")
    except Exception as _e:
        print(f"[J1] cam_by_fid build failed: {_e}")
        TEMPORAL_SMOOTH_W = 0.0

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # R-SUP-4 (gated, default OFF): mouth AU apex oversampling. Same as face stage
    # — prior unconditional run hurt PSNR. Set TG_RSUP=1 to enable.
    _rsup_on_m = os.environ.get('TG_RSUP', '0') == '1'
    all_train_cams_m = list(scene.getTrainCameras())
    if _rsup_on_m:
        sample_weights_m = []
        for _cam in all_train_cams_m:
            _td = getattr(_cam, 'talking_dict', {}) or {}
            au45_n = float(_td.get('blink', 0)) * 2.0
            au45_n = min(au45_n / 1.0, 1.0)
            _au25 = _td.get('au25', None)
            if _au25 is not None and len(_au25) >= 5 and _au25[4] > 0:
                au25_n = min(float(_au25[0]) / max(float(_au25[4]), 1e-3), 1.0)
            else:
                au25_n = 0.0
            sample_weights_m.append(min(1.0 + 2.0 * max(au45_n, au25_n), 3.0))
        sample_weights_m = np.array(sample_weights_m, dtype=np.float64)
        sample_weights_m /= sample_weights_m.sum()
        print(f"[R-SUP-4 ON] mouth apex oversampling weights [{sample_weights_m.min()*len(all_train_cams_m):.2f}, {sample_weights_m.max()*len(all_train_cams_m):.2f}]× per cam")
        def _refill_viewpoint_stack_m():
            idxs = np.random.choice(len(all_train_cams_m), size=len(all_train_cams_m),
                                    p=sample_weights_m, replace=True)
            return [all_train_cams_m[i] for i in idxs]
    else:
        print("[R-SUP-4 OFF] mouth uniform frame sampling")
        def _refill_viewpoint_stack_m():
            return all_train_cams_m.copy()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = _refill_viewpoint_stack_m()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # find a big mouth

        au_global_lb = viewpoint_cam.talking_dict['au25'][1]
        au_global_ub = viewpoint_cam.talking_dict['au25'][4]
        au_window = (au_global_ub - au_global_lb) * 0.2

        au_ub = au_global_ub
        au_lb = au_ub - mouth_step * iteration * (au_global_ub - au_global_lb)

        if iteration < warm_step:
            while viewpoint_cam.talking_dict['au25'][0] < au_global_ub:
                if not viewpoint_stack:
                    viewpoint_stack = _refill_viewpoint_stack_m()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if warm_step < iteration < mouth_select_iter:
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['au25'][0] < au_lb or viewpoint_cam.talking_dict['au25'][0] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = _refill_viewpoint_stack_m()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            if viewpoint_cam.original_image == None:
                viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

            while torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda().sum() < 20:
                if not viewpoint_stack:
                    viewpoint_stack = _refill_viewpoint_stack_m()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                if viewpoint_cam.original_image == None:
                    viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        if viewpoint_cam.original_image == None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration > bg_iter:
            # turn to black
            bg_color = [0, 0, 0] # if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask
        # V30: cavity gets 3x weight inside mouth supervision so teeth/tongue
        # pixels (a tiny minority) drive the loss harder than lip pixels.
        _cavity_t = viewpoint_cam.talking_dict.get('cavity_mask', None)
        if _cavity_t is not None:
            cavity_mask = torch.as_tensor(_cavity_t).cuda().bool()
        else:
            cavity_mask = None

        # V17 Soft Mask Boundary: extend mouth-branch supervision out to a
        # dilated ring (mouth_overlap). Both face and mouth branches receive
        # GT in this ring, eliminating the hard seam where lip Gaussians used
        # to disappear when the mouth_mask jittered into lip pixels.
        mouth_core, mouth_overlap = build_soft_mouth_masks(mouth_mask, erode_k=3, dilate_k=5)

        [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
        lips_mask = torch.zeros_like(mouth_mask)
        lips_mask[xmin:xmax, ymin:ymax] = True

        # V23: after warm_step, mouth render goes through render_motion_mouth_v18
        # with a live cross-attn residual computed from audio + AU window.
        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            cross_attn_residual = None
        else:
            au17_window = _gather_au(int(viewpoint_cam.talking_dict.get('img_id', -1)))
            audio_raw = viewpoint_cam.talking_dict['auds'].cuda()
            xyz_feat = motion_net.encode_x(gaussians.get_xyz, bound=motion_net.bound)
            audio_seq = motion_net.audio_net(audio_raw)
            if audio_seq.dim() == 1:
                audio_seq = audio_seq.unsqueeze(0)
            # V30 change C: dual-head cavity/lip cross-attn split.
            # Guard against densification: if Gaussian count changed, the
            # frozen cavity_idx is stale -> fall back to single head.
            if cavity_idx is not None and cavity_idx.shape[0] != xyz_feat.shape[0]:
                cavity_idx = None  # disable for the rest of training
                print(f"[v30] iter {iteration}: Gaussian count changed to {xyz_feat.shape[0]}, "
                      f"disabling cavity_idx for remaining iters")
            if cavity_idx is not None and int(cavity_idx.sum().item()) > 8 \
                    and int((~cavity_idx).sum().item()) > 8:
                # Split xyz_feat by cavity_idx; run two heads; scatter back.
                xf_lip = xyz_feat[~cavity_idx]
                xf_cav = xyz_feat[cavity_idx]
                au_lip_in = au17_window
                au_cav_in = au17_window * _CAVITY_AU_MASK if au17_window.dim() == 1 \
                    else au17_window * _CAVITY_AU_MASK[None, :]
                res_lip = cross_attn_driver_lip(xf_lip, audio_seq, au_lip_in)
                res_cav = cross_attn_driver_cavity(xf_cav, audio_seq, au_cav_in)
                N = xyz_feat.shape[0]
                cross_attn_residual = {}
                for k in res_lip:
                    if isinstance(res_lip[k], torch.Tensor) and res_lip[k].dim() >= 1 \
                            and res_lip[k].shape[0] == xf_lip.shape[0]:
                        merged = torch.zeros((N,) + tuple(res_lip[k].shape[1:]),
                                              device=res_lip[k].device, dtype=res_lip[k].dtype)
                        merged[~cavity_idx] = res_lip[k]
                        merged[cavity_idx]  = res_cav[k]
                        cross_attn_residual[k] = merged
                    else:
                        cross_attn_residual[k] = res_lip[k]
            else:
                # Fallback: single (lip) head over all Gaussians.
                cross_attn_residual = cross_attn_driver_lip(xyz_feat, audio_seq, au17_window)
            # Q1: predict lip landmark from audio for this frame
            _q1_lmk = None
            _r3_amp = None
            _fid_now = int(viewpoint_cam.talking_dict.get('img_id', -1))
            if _fid_now >= 0:
                if q1_predictor is not None:
                    _q1_lmk = _gather_landmark_for_fid(_fid_now)
                if r3_predictor is not None:
                    _r3_amp = _gather_amp_for_fid(_fid_now)
            # E2: audio/AU-conditioned RGB residual on mouth Gaussians. Gated.
            albedo_residual_mouth = None
            if os.environ.get('TG_USE_D2', '0') == '1':
                with torch.no_grad():
                    _audio_emb_for_alb = motion_net.encode_audio(audio_raw)
                _au17_single = au17_window if au17_window.dim() == 1 else au17_window[au17_window.shape[0]//2]
                albedo_residual_mouth = albedo_head_mouth(
                    gaussians.get_xyz.detach(), _audio_emb_for_alb.squeeze(0), _au17_single,
                )
            render_pkg = render_motion_mouth_v18(
                viewpoint_cam, gaussians, motion_net, pipe, background,
                cross_attn_residual=cross_attn_residual,
                landmark=_q1_lmk,
                motion_amplifier=_r3_amp,
                albedo_residual=albedo_residual_mouth,
            )

        image_green, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        # V17: supervise the mouth branch on mouth_overlap (mouth + dilated ring).
        # Lip pixels next to the seam now receive a real GT colour instead of
        # being painted with the green background, so the mouth branch can keep
        # rendering them when the segmentation jitter assigns them to "mouth".
        gt_image_green = gt_image * mouth_overlap + background[:, None, None] * ~mouth_overlap

        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
        if iteration > bg_iter:
            gaussians._xyz.requires_grad = False
            gaussians._opacity.requires_grad = False
            # gaussians._features_dc.requires_grad = False
            # gaussians._features_rest.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False
        
        # Loss
        # V17: clear only the lips_rect pixels that are OUTSIDE mouth_overlap,
        # so the seam ring inside lips_rect is co-rendered (instead of the old
        # `lips_mask ^ mouth_mask` which forcibly erased the whole lip area).
        clear_mask = lips_mask & (~mouth_overlap)
        image_green[:, clear_mask] = background[:, None]

        # V30: cavity-region 3x L1 weight. Mean L1 over the whole image is
        # dominated by lip pixels (~10x more than cavity pixels). Lifting the
        # cavity weight pushes the mouth Gaussians to reconstruct teeth/tongue
        # accurately, even on infrequent open-mouth frames.
        if cavity_mask is not None and cavity_mask.sum() > 8:
            w_map = torch.ones_like(mouth_mask, dtype=torch.float32)
            w_map[cavity_mask] = 3.0
            diff = (image_green - gt_image_green).abs().mean(dim=0)
            Ll1 = (diff * w_map).sum() / w_map.sum().clamp_min(1.0)
        else:
            Ll1 = l1_loss(image_green, gt_image_green)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_green, gt_image_green))

        # R-ANISO-REG (2026-05-26): penalize extreme anisotropy in mouth Gaussians.
        # Set env TG_ANISO_REG_W>0 to enable. Subjects with sparse effective signal
        # (e.g. may) produce extreme-aspect-ratio Gaussians that look like needles
        # and cause smear at transition frames. Gentle hinge keeps log(s_max/s_min)
        # below log(thr) — adaptive: macron/obama naturally stay below thr so penalty=0.
        import os as _os_aniso
        _aniso_w = float(_os_aniso.environ.get('TG_ANISO_REG_W', '0.0'))
        if _aniso_w > 0:
            import numpy as _np_aniso
            _thr = float(_os_aniso.environ.get('TG_ANISO_THR', '50.0'))
            _scale = torch.exp(gaussians._scaling)
            _s_max = _scale.max(dim=1).values
            _s_min = _scale.min(dim=1).values
            _aniso_log = torch.log(_s_max / (_s_min + 1e-8))
            _aniso_penalty = (_aniso_log - _np_aniso.log(_thr)).clamp(min=0).mean()
            loss = loss + _aniso_w * _aniso_penalty
            if iteration % 5000 == 0:
                with torch.no_grad():
                    _aniso = _s_max / (_s_min + 1e-8)
                    _frac_over = (_aniso > _thr).float().mean().item()
                    print(f'[R-ANISO-REG iter {iteration}] thr={_thr} w={_aniso_w} '
                          f'frac>thr={100*_frac_over:.1f}%, penalty={_aniso_penalty.item():.4f}', flush=True)

        # A+C apex-aware reweighting: scale total render loss by (1 + boost * ma_norm)
        if apex_ma_arr is not None:
            _fid = int(viewpoint_cam.talking_dict.get('img_id', -1))
            if 0 <= _fid < len(apex_ma_arr):
                _ma_norm = float(apex_ma_arr[_fid]) / apex_ma_max
                _apex_w = 1.0 + APEX_BOOST * _ma_norm
                loss = loss * _apex_w

        # J1 temporal smoothness: penalize d_xyz drift between t and t-1.
        # Only kicks in after warm_step (render_pkg has 'motion').
        if TEMPORAL_SMOOTH_W > 0 and 'motion' in render_pkg \
                and isinstance(render_pkg.get('motion'), dict) \
                and 'd_xyz' in render_pkg['motion']:
            _fid_t = int(viewpoint_cam.talking_dict.get('img_id', -1))
            _fid_adj = _fid_t - 1 if _fid_t > 0 else _fid_t + 1
            _cam_adj = _cam_by_fid.get(_fid_adj, None)
            if _cam_adj is not None:
                try:
                    _adj_audio = _cam_adj.talking_dict.get('auds', None)
                    if _adj_audio is not None:
                        _adj_audio = _adj_audio.cuda()
                        # cheap motion_net forward (no rasterize, no cross-attn)
                        _motion_adj = motion_net(gaussians.get_xyz, _adj_audio)
                        _d_xyz_t   = render_pkg['motion']['d_xyz']
                        _d_xyz_adj = _motion_adj['d_xyz']
                        _temporal = (_d_xyz_t - _d_xyz_adj).abs().mean()
                        loss = loss + TEMPORAL_SMOOTH_W * _temporal
                except Exception:
                    pass

        # R2 (2026-05-19): lip-y supervision. Project current mouth Gaussians
        # (after motion + cross_attn additive — taken from render_pkg motion)
        # to 2D image, compute upper/lower lip subset y-mean, compare to GT.
        if r2_gt_lip_y is not None and 'motion' in render_pkg \
                and isinstance(render_pkg.get('motion'), dict) \
                and 'd_xyz' in render_pkg['motion']:
            _fid_r2 = int(viewpoint_cam.talking_dict.get('img_id', -1))
            if 0 <= _fid_r2 < r2_gt_lip_y.shape[0]:
                try:
                    _d_xyz_main = render_pkg['motion']['d_xyz']
                    # Get cross_attn d_xyz if available (it's already included in render
                    # but motion['d_xyz'] is JUST main motion_net output). Re-add
                    # cross_attn to get total motion as rendered.
                    _total_dxyz = _d_xyz_main
                    if cross_attn_residual is not None and 'd_xyz' in cross_attn_residual:
                        _total_dxyz = _d_xyz_main + cross_attn_residual['d_xyz']
                    _xyz_now = gaussians.get_xyz + _total_dxyz  # (N, 3) world
                    _H = int(viewpoint_cam.image_height)
                    _W = int(viewpoint_cam.image_width)
                    _P = viewpoint_cam.full_proj_transform.cuda().float()
                    _row, _col = _r2_project_to_2d(_xyz_now, _P, _H, _W)
                    # Filter Gaussians inside this frame's lip_mask
                    _lip_t_r2 = viewpoint_cam.talking_dict.get('lip_mask', None)
                    if _lip_t_r2 is not None:
                        _lip_bool = torch.as_tensor(_lip_t_r2).cuda().bool()
                        # use long indices and clamp
                        _row_l = _row.detach().long().clamp(0, _H - 1)
                        _col_l = _col.detach().long().clamp(0, _W - 1)
                        _in_lip = _lip_bool[_row_l, _col_l]
                        if int(_in_lip.sum()) >= 20:
                            _row_lip = _row[_in_lip]
                            # split upper/lower by current median row (smaller row = upper, larger = lower)
                            _med = _row_lip.detach().median()
                            _upper_y = _row_lip[_row_lip.detach() < _med].mean()
                            _lower_y = _row_lip[_row_lip.detach() >= _med].mean()
                            _gt_u = r2_gt_lip_y[_fid_r2, 0]
                            _gt_l = r2_gt_lip_y[_fid_r2, 1]
                            # pixel L1 normalised by image height
                            _r2_loss = ((_upper_y - _gt_u).abs() + (_lower_y - _gt_l).abs()) / _H
                            loss = loss + R2_LIP_Y_W * _r2_loss
                except Exception:
                    pass


        if iteration > warm_step:
            # V31 ALPHA-HINGE-RESTORE-FixAC (2026-05-13 afternoon): undo my
            # earlier lips_mask revert. Pure FixAC reproduction = violent prune
            # ON + V17 mouth_core alpha hinge. Earlier test showed lips_mask
            # hinge + violent prune are antagonistic (Gaussians pushed to lips
            # periphery, cavity empty -> N capped at 8k but 0 cavity render).
            loss += 1e-3 * (((1 - alpha) * mouth_core.float()).mean()
                            + (alpha * (~lips_mask).float()).mean())

        # V30 change B: cavity depth prior. Push pixels inside cavity_mask
        # to render at greater depth than pixels inside lip_mask (i.e. cavity
        # is physically "behind" the lips). Doesn't require any mesh -- pure
        # 2D depth-map comparison drives mouth Gaussians toward correct z.
        _lip_t = viewpoint_cam.talking_dict.get('lip_mask', None)
        if cavity_mask is not None and _lip_t is not None and 'depth' in render_pkg:
            depth_map = render_pkg['depth']
            if depth_map.dim() == 3: depth_map = depth_map[0]   # [H,W]
            alpha_map = alpha[0] if alpha.dim() == 3 else alpha
            lip_t = torch.as_tensor(_lip_t).cuda().bool()
            cavity_pix = cavity_mask & (alpha_map > 0.5)
            lip_pix    = lip_t       & (alpha_map > 0.5)
            if cavity_pix.sum() > 32 and lip_pix.sum() > 32:
                z_lip_med = depth_map[lip_pix].median().detach()
                z_cavity  = depth_map[cavity_pix]
                # cavity depth must be at least 0.005 (≈5mm in normalised units)
                # behind median lip depth; hinge loss only on violations.
                loss_depth = torch.relu(z_lip_med + 0.005 - z_cavity).mean()
                # softHinge (5/17): weight 0.5 → 0.05 to stop pushing cavity Gaussian
                # in +z direction which feeds the sink ("cavity must be deeper than
                # lip" + "Gaussians flying off-screen are deeper still" → loss decreases
                # as Gaussians fly toward +∞).
                loss = loss + 0.05 * loss_depth

        image_t = image_green.clone()
        gt_image_t = gt_image_green.clone()

        if iteration > lpips_start_iter:
            patch_size = random.randint(16, 21) * 2
            loss += 0.5 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()

        # V23: cross-attn residual L1 reg + Sobel high-freq match
        if iteration > warm_step and cross_attn_residual is not None:
            loss += 1e-5 * cross_attn_residual['d_xyz'].abs().mean()
            loss += 1e-5 * cross_attn_residual['d_rot'].abs().mean()
            loss += 1e-5 * cross_attn_residual['d_scale'].abs().mean()
        # V32 / V32.1 d_xyz hinge (5/16) reverted 5/17 — weight 1.0 killed
        # motion_net entirely (mouth PSNR dropped to 8 dB). Going back to pure
        # FixAC config; using TG_AU_CSV=au_corrected.csv to test if corrected
        # AU alone (without hinge) still triggers the sink optimum.

        if iteration > warm_step:
            sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device='cuda').view(1,1,3,3)
            sobel_y = sobel_x.transpose(2,3)
            kx = sobel_x.expand(3,1,3,3).contiguous()
            ky = sobel_y.expand(3,1,3,3).contiguous()
            _w = 0.2 + 0.3 * min(1.0, (iteration - warm_step) / 20000.0)
            pgx = F.conv2d(image_green[None], kx, padding=1, groups=3)
            pgy = F.conv2d(image_green[None], ky, padding=1, groups=3)
            tgx = F.conv2d(gt_image_green[None], kx, padding=1, groups=3)
            tgy = F.conv2d(gt_image_green[None], ky, padding=1, groups=3)
            loss = loss + _w * ((pgx - tgx).abs().mean() + (pgy - tgy).abs().mean())

        # V23: features_dc anchor (skip if Gaussian count changed via densify)
        if iteration == warm_step + 1000 and feat_anchor_init is None:
            feat_anchor_init = gaussians._features_dc.detach().clone()
            print(f"[v3] mouth features_dc anchor @ iter {iteration}")
        if feat_anchor_init is not None:
            if gaussians._features_dc.shape == feat_anchor_init.shape:
                loss = loss + 0.005 * (gaussians._features_dc - feat_anchor_init).abs().mean()
            else:
                feat_anchor_init = gaussians._features_dc.detach().clone()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration)+'_mouth')

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion_mouth, (pipe, background))
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(),
                        motion_optimizer.state_dict(), iteration,
                        {'cross_attn_driver': cross_attn_driver_lip.state_dict(),
                         'cross_attn_driver_lip': cross_attn_driver_lip.state_dict(),
                         'cross_attn_driver_cavity': cross_attn_driver_cavity.state_dict(),
                         'albedo_head_mouth': albedo_head_mouth.state_dict(),
                         'albedo_residual_scale_mouth': 0.35})
                torch.save(ckpt, scene.model_path + "/chkpnt_mouth_v30_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_mouth_v30_latest.pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

                    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    from utils.sh_utils import eval_sh
                    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                    # V30 fix v2 (2026-05-09 evening): REVERTED Fix A (relaxed
                    # near-green prune). Testing showed it killed legitimate
                    # Gaussians during open-mouth motion: lip Gaussians
                    # transiently displaced into green-bg region while learning
                    # to open the mouth, briefly acquired green tint, got
                    # killed. Network learned "don't move" → mouth doesn't open
                    # in inference. Fix C (boundary loss reweight to 0.3
                    # outside mouth_overlap, in the L1 weight map above) alone
                    # is sufficient: it removes the gradient that pulls
                    # boundary Gaussians toward green, so few drift in the
                    # first place, and the original pure-green prune handles
                    # any that do.
                    bg_color_mask = (colors_precomp[..., 0] < 20/255) * (colors_precomp[..., 1] > 235/255) * (colors_precomp[..., 2] < 20/255)
                    # noPrune (5/17): disable violent green prune on corrected AU.
                    # On obama + corrected AU, violent prune was identified as the
                    # main sink trigger (kills Gaussians that turn green → rewards
                    # "fly off-screen and avoid being rasterized" sink strategy).
                    # TG-orig has no green prune and mouth_xyz stays at z=0.030
                    # under same corrected AU input (vs v30's z=0.149 sink).
                    # Gentler accum decay only — no opacity/scaling assault.
                    gaussians.xyz_gradient_accum[bg_color_mask] /= 2
                    # R3 (5/20) Soft prune — re-enabled but gentle.
                    # Violent: opacity:=0.1 (10x cut), scaling/=10  → kills Gaussian → sink
                    # Soft:    opacity *= 0.5 (2x cut), scaling *= 0.7 → wounds, recoverable
                    # Goal: control mouth Gaussian count (~50k vs noPrune 100k)
                    # without triggering sink optimum.
                    if int(bg_color_mask.sum()) > 0:
                        _old_op = torch.sigmoid(gaussians._opacity[bg_color_mask])
                        _new_op = (_old_op * 0.5).clamp(1e-3, 0.999)
                        gaussians._opacity[bg_color_mask] = gaussians.inverse_opacity_activation(_new_op)
                        # scale: log-space, subtract log(1/0.7) to multiply scale by 0.7
                        import math as _math
                        gaussians._scaling[bg_color_mask] = gaussians._scaling[bg_color_mask] + _math.log(0.7)

                    # R-Z-PRUNE (2026-05-26): gated by env TG_MOUTH_Z_MAX (default disabled).
                    # When set (e.g. 0.05), kill mouth Gaussians whose Z > threshold.
                    # Prevents parallax-smear at apex frames from deep-into-head Gaussians.
                    # TG-orig naturally has Z<=0.045; our v30au25 had 27% with Z>0.08
                    # causing visible smear during head motion on may.
                    import os as _os_zp
                    _z_max = float(_os_zp.environ.get('TG_MOUTH_Z_MAX', '0'))
                    if _z_max > 0:
                        _z = gaussians.get_xyz[:, 2]
                        _z_kill = _z > _z_max
                        if int(_z_kill.sum()) > 0:
                            gaussians.prune_points(_z_kill)
                            if iteration % 5000 == 0:
                                print(f'[R-Z-PRUNE iter {iteration}] killed {int(_z_kill.sum())} Gaussians with Z > {_z_max}', flush=True)

                    # V30: Gaussian count changed after densify -> refresh
                    # cavity_idx so the dual-head split stays valid.
                    cavity_idx = _compute_cavity_idx(verbose=(iteration % 10000 == 0))

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()
                attn_optimizer.step()

                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none = True)
                attn_optimizer.zero_grad(set_to_none=True)

                scheduler.step()



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 10)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if viewpoint.original_image == None:
                        viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
                        
                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)


                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # V17: 7938 frames * full preload ~27 GB → host OOM. Force lazy-load.
    dataset_ns = lp.extract(args)
    dataset_ns.au_editor_mode = True
    training(dataset_ns, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
