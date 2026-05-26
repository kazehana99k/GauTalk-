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
from gaussian_renderer import render, render_motion
import sys
from scene import Scene, GaussianModel, MotionNetwork
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
from train_fuse_v15 import render_motion_with_albedo
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations]

    # vars
    warm_step = 3000
    opt.densify_until_iter = opt.iterations - 1000
    bg_iter = opt.iterations # opt.densify_until_iter
    lpips_start_iter = opt.densify_until_iter - 2000
    motion_stop_iter = bg_iter
    mouth_select_iter = bg_iter - 10000
    mouth_step = 1 / mouth_select_iter
    hair_mask_interval = 7
    select_interval = 15

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / bg_iter))

    # V23: cross-attn driver trained from scratch alongside face Gaussians.
    audio_dim = motion_net.audio_dim
    cross_attn_driver = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net.in_dim,
        audio_seq_len=8, audio_dim=audio_dim, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        # V30: 3x residual headroom — cross-attn can actually move lip
        # Gaussians wide enough on open-mouth frames (fixes frame 7974 case).
        residual_scale_xyz=1e-3, residual_scale_scale=1e-3, residual_scale_rot=1e-3,
    ).cuda()
    attn_optimizer = torch.optim.Adam(cross_attn_driver.parameters(), lr=5e-4, weight_decay=1e-6)
    print(f"[v30] face cross_attn_driver instantiated: in_dim={motion_net.in_dim}, audio_dim={audio_dim}, residual_scale=1e-3")

    # Pre-load full AU17 sequence for sliding-window AU tokens.
    # R-DATA-5: read TG_AU_CSV (matches dataset_readers' active reader) so that
    # motion_net's `e` input and cross_attn_driver's au17_window come from the
    # SAME csv. Otherwise (e.g. obama with corrected.csv exported via TG_AU_CSV),
    # motion_net would see corrected AU while cross_attn saw raw AU.
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

    # P1 FaceLipRelease (2026-05-18): apex-aware face_alpha penalty in lip region.
    # Audit found face_alpha @ apex fid 7715 in lip region = 0.65, occluding 65%
    # of mouth Gaussian's render. Force face to "step aside" (alpha→0) in lip
    # region when mouth_area is large, so mouth can fully take over lip motion.
    # R-ENG-2: weight reduced 30 → 5. The previous 30 was tuned while alpha pass
    # was wrapped in `with torch.no_grad()` (dead-code, gradient = 0), so the
    # author kept raising the coefficient. With R-ENG-1 the alpha gradient now
    # flows; 5 is sufficient and matches the order of other supervised losses.
    P1_APEX_LIP_W = 5.0
    p1_ma_arr = None
    p1_ma_max = 1.0
    try:
        _p = os.path.join(dataset.source_path, 'mouth_area_per_frame.npy')
        if os.path.exists(_p):
            p1_ma_arr = np.load(_p)
            p1_ma_max = float(p1_ma_arr[p1_ma_arr > 0].max())
            print(f"[P1] FaceLipRelease enabled, weight={P1_APEX_LIP_W}, ma_max={p1_ma_max:.0f}")
    except Exception as _e:
        print(f"[P1] mouth_area load failed: {_e}")

    # V23: VGG LPIPS (not alex) — alex over-smooths face skin.
    # V30: use alex LPIPS instead of vgg — vgg's deeper conv stack OOMs at
    # late densification when Gaussian count is high (>30k). Anti-smoothing
    # is now handled by Sobel + features_dc anchor + cross-attn residual.
    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        raw = torch.load(checkpoint)
        # support either v2 4-tuple OR v3 5-tuple (with cross_attn_driver dict)
        model_params, motion_params, motion_optimizer_params = raw[0], raw[1], raw[2]
        first_iter = raw[3]
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)
        if len(raw) >= 5 and isinstance(raw[4], dict) and 'cross_attn_driver' in raw[4]:
            cross_attn_driver.load_state_dict(raw[4]['cross_attn_driver'])
            print('[v3] resumed cross_attn_driver from ckpt')

    # V23: features_dc anchor — capture once after warm_step rather than at
    # iter 0 (so initial random init isn't anchored).  Triggered later in loop.
    feat_anchor_init = None

    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # R-SUP-4 (gated, default OFF): AU apex frame oversampling. Previous run
    # (TG_RSUP=1 implicit) ran face+mouth+fuse and got -9 dB vs baseline because
    # R-SUP-2/3/4 each shifted loss landscape away from the trained equilibrium.
    # Default 0 → uniform sampling = baseline behavior. Set TG_RSUP=1 to revive.
    _rsup_on = os.environ.get('TG_RSUP', '0') == '1'
    all_train_cams = list(scene.getTrainCameras())
    if _rsup_on:
        sample_weights = []
        for _cam in all_train_cams:
            _td = getattr(_cam, 'talking_dict', {}) or {}
            au45_n = float(_td.get('blink', 0)) * 2.0
            au45_n = min(au45_n / 1.0, 1.0)
            _mb = _td.get('mouth_bound', None)
            if _mb is not None and len(_mb) >= 3 and _mb[1] > _mb[0]:
                jaw = float(_mb[2]); jaw_p = float(_mb[1])
                au25_n = min(jaw / max(jaw_p, 1.0), 1.0)
            else:
                au25_n = 0.0
            sample_weights.append(min(1.0 + 2.0 * max(au45_n, au25_n), 3.0))
        sample_weights = np.array(sample_weights, dtype=np.float64)
        sample_weights /= sample_weights.sum()
        print(f"[R-SUP-4 ON] AU apex oversampling: weights [{sample_weights.min()*len(all_train_cams):.2f}, {sample_weights.max()*len(all_train_cams):.2f}]× per cam")
        def _refill_viewpoint_stack():
            idxs = np.random.choice(len(all_train_cams), size=len(all_train_cams),
                                    p=sample_weights, replace=True)
            return [all_train_cams[i] for i in idxs]
    else:
        print("[R-SUP-4 OFF] uniform frame sampling (baseline behavior)")
        def _refill_viewpoint_stack():
            return all_train_cams.copy()

    # R-SUP-3 (gated, default OFF): temporal smoothness on d_xyz
    TEMPORAL_SMOOTH_W = 0.5 if _rsup_on else 0.0
    fid_to_cam = {}
    if _rsup_on:
        for _c in all_train_cams:
            _td = getattr(_c, 'talking_dict', {}) or {}
            _fid = _td.get('img_id', None)
            if _fid is not None:
                fid_to_cam[int(_fid)] = _c
        print(f"[R-SUP-3 ON] temporal smoothness map: {len(fid_to_cam)} frames, w={TEMPORAL_SMOOTH_W}")

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
            # R-SUP-4: refill with AU apex oversampling rather than uniform copy
            viewpoint_stack = _refill_viewpoint_stack()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # find a big mouth
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window


        au_global_lb = 0
        au_global_ub = 1
        au_window = 0.3

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
        au_ub = au_lb + au_window
        au_lb = au_lb - au_window * 0.5


        if iteration < warm_step:
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = _refill_viewpoint_stack()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        if warm_step < iteration < mouth_select_iter:

            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = _refill_viewpoint_stack()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if viewpoint_cam.original_image == None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask

        # V17 Soft Mask Boundary: face only carves OUT the eroded interior
        # (mouth_core); the seam ring around the lips remains in face's GT so
        # face Gaussians keep rendering the lip pixels next to the mouth seam.
        mouth_core, mouth_overlap = build_soft_mouth_masks(mouth_mask, erode_k=3, dilate_k=5)

        if iteration > lpips_start_iter:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
            mouth_core = (-max_pool(-max_pool(mouth_core[None].float())))[0].bool()

        
        hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0

        # V23: after warm_step, render through render_motion_with_albedo with
        # an active cross_attn_residual.  Albedo residual is zeroed (face stage
        # doesn't have an albedo head — that's a fuse-stage thing).
        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            cross_attn_residual = None
        else:
            # build cross-attn residual
            au17_window = _gather_au(int(viewpoint_cam.talking_dict.get('img_id', -1)))
            audio_raw = viewpoint_cam.talking_dict['auds'].cuda()
            xyz_feat = motion_net.encode_x(gaussians.get_xyz, bound=motion_net.bound)
            audio_seq = motion_net.audio_net(audio_raw)
            if audio_seq.dim() == 1:
                audio_seq = audio_seq.unsqueeze(0)
            cross_attn_residual = cross_attn_driver(xyz_feat, audio_seq, au17_window)
            zero_alb = torch.zeros((gaussians.get_xyz.shape[0], 3), device='cuda')
            render_pkg = render_motion_with_albedo(
                viewpoint_cam, gaussians, motion_net, pipe, background,
                albedo_residual=zero_alb,
                cross_attn_residual=cross_attn_residual,
            )

        image_white = render_pkg["render"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        # render_pkg may not include 'attn' from albedo path — guard later use.
        
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

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
        if iteration < bg_iter:
            if hair_mask_iter:
                image_white[:, hair_mask] = background[:, None]
                gt_image_white[:, hair_mask] = background[:, None]
            
            # V17 Soft Mask Boundary: only the eroded interior (mouth_core)
            # is hidden from face — the seam ring is left for face Gaussians to
            # render normally, so lip pixels next to the mouth aren't dropped.
            gt_image_white[:, mouth_core] = background[:, None]

            # V30 change A: per-pixel lip 2x weighted L1.  Replaces V28's
            # bounding-box lips_rect 3x (which over-weights skin around lips).
            # Uses lip_mask from EasyPortrait class 6+LAB refinement so only
            # actual red-lip pixels get the boost.
            _lip_t = viewpoint_cam.talking_dict.get('lip_mask', None)
            if _lip_t is not None:
                lip_t = torch.as_tensor(_lip_t).cuda().bool()
                # P1.v3 (2026-05-18): ALSO mask lip pixels from face GT (let mouth
                # Gaussians take over lip rendering). Without this, face L1 pulls
                # alpha→1 in lip to render lip pink → P1 alpha hinge can't win.
                # Scaled by ma_norm: only at apex do we fully release lip to mouth;
                # at neutral keep face rendering lip normally.
                if p1_ma_arr is not None:
                    _fid_face = int(viewpoint_cam.talking_dict.get('img_id', -1))
                    if 0 <= _fid_face < len(p1_ma_arr):
                        _ma_n = float(p1_ma_arr[_fid_face]) / p1_ma_max
                        if _ma_n > 0.0:
                            # blend: at apex (ma_n=1) lip GT becomes bg
                            # at neutral (ma_n=0) lip GT untouched
                            gt_lip_replacement = background[:, None].expand(-1, int(lip_t.sum()))
                            gt_image_white[:, lip_t] = (
                                gt_image_white[:, lip_t] * (1 - _ma_n)
                                + gt_lip_replacement * _ma_n
                            )

                w_map = torch.ones_like(lip_t, dtype=torch.float32)
                w_map[lip_t] = 2.0
                # R-SUP-2 (gated, default OFF): eye region weight ×4. Previous
                # run showed this shifted loss away from face proper → -3 dB.
                if _rsup_on:
                    _eye_t = viewpoint_cam.talking_dict.get('fp_eye_mask', None)
                    if _eye_t is not None:
                        eye_t = torch.as_tensor(_eye_t).cuda().bool()
                        w_map[eye_t] = 4.0
                diff = (image_white - gt_image_white).abs().mean(dim=0)
                Ll1 = (diff * w_map).sum() / w_map.sum().clamp_min(1.0)
            else:
                Ll1 = l1_loss(image_white, gt_image_white)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))

            # R-ENG-2: activate mouth_core alpha penalty. After R-ENG-1 made the
            # alpha pass grad-enabled, this drives face α→0 inside the eroded
            # mouth interior, so the mouth Gaussians can render through cleanly
            # in the fuse-stage face_over_mouth composite.
            mouth_alpha_loss = 1e-2 * alpha[:, mouth_core].mean()
            if not torch.isnan(mouth_alpha_loss):
                loss = loss + mouth_alpha_loss

            if iteration > warm_step:
                loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()
                # V23: also penalise excessive cross-attn residual magnitudes
                # (small reg keeps it from drifting Gaussians off-place).
                if cross_attn_residual is not None:
                    loss += 1e-5 * cross_attn_residual['d_xyz'].abs().mean()
                    loss += 1e-5 * cross_attn_residual['d_rot'].abs().mean()
                    loss += 1e-5 * cross_attn_residual['d_scale'].abs().mean()

                # P1.v2: exclude lip_mask from alpha=1 hinge (head_mask region)
                # to remove the head_mask vs P1 tug-of-war in lip pixels.
                if _lip_t is not None:
                    _hm_alpha1 = head_mask & ~lip_t   # alpha=1 hinge skips lip area
                else:
                    _hm_alpha1 = head_mask
                loss += 1e-3 * (((1-alpha) * _hm_alpha1).mean() + (alpha * ~head_mask).mean())

                # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])

            # P1.v2 FaceLipRelease — applied EVERY iter (incl. warm_step) so that
            # face _opacity Parameter learns "step aside in lip" early, before
            # _opacity sigmoid saturates to 1.
            if p1_ma_arr is not None and _lip_t is not None:
                _fid_p1 = int(viewpoint_cam.talking_dict.get('img_id', -1))
                if 0 <= _fid_p1 < len(p1_ma_arr):
                    _ma_norm_p1 = float(p1_ma_arr[_fid_p1]) / p1_ma_max
                    if _ma_norm_p1 > 0.0:
                        _a2d = alpha[0] if alpha.dim() == 3 else alpha
                        _face_in_lip = (_a2d * lip_t.float()).mean()
                        loss = loss + P1_APEX_LIP_W * _ma_norm_p1 * _face_in_lip

            # R-SUP-3 (gated, default OFF via TEMPORAL_SMOOTH_W=0): temporal smoothness.
            if (_rsup_on
                    and iteration > warm_step
                    and 'motion' in render_pkg
                    and isinstance(render_pkg.get('motion'), dict)
                    and 'd_xyz' in render_pkg['motion']):
                _fid_cur = int(viewpoint_cam.talking_dict.get('img_id', -1))
                _fid_adj = _fid_cur - 1 if _fid_cur > 0 else _fid_cur + 1
                _cam_adj = fid_to_cam.get(_fid_adj, None)
                if _cam_adj is not None:
                    try:
                        _adj_audio = _cam_adj.talking_dict['auds'].cuda()
                        _ae = _cam_adj.talking_dict.get('au_exp', None)
                        if isinstance(_ae, torch.Tensor):
                            _adj_e = _ae.cuda()
                        else:
                            _adj_e = torch.as_tensor(_ae).cuda()
                        _adj_motion = motion_net(gaussians.get_xyz, _adj_audio, _adj_e)
                        _temporal = (render_pkg['motion']['d_xyz'] - _adj_motion['d_xyz']).abs().mean()
                        loss = loss + TEMPORAL_SMOOTH_W * _temporal
                    except Exception:
                        pass

            # R-MONITOR-1: log d_xyz / d_scale norm + cap-saturation rate so we can
            # post-hoc check whether R-GEO-1 cap is too tight (would show d_xyz
            # saturating > 80% of frames). Cheap stats, no grad impact.
            if (iteration > warm_step and iteration % 200 == 0
                    and 'motion' in render_pkg
                    and isinstance(render_pkg.get('motion'), dict)):
                with torch.no_grad():
                    _dxyz = render_pkg['motion']['d_xyz']
                    _sat_y = ((_dxyz[..., 1].abs() / 0.06) > 0.85).float().mean().item()  # cap_y=0.06
                    _norm_mean = _dxyz.norm(dim=-1).mean().item()
                    if tb_writer is not None:
                        tb_writer.add_scalar('motion/dxyz_norm_mean', _norm_mean, iteration)
                        tb_writer.add_scalar('motion/cap_sat_y_rate', _sat_y, iteration)

            image_t = image_white.clone()
            gt_image_t = gt_image_white.clone()

        else:
            # with real bg
            image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

            Ll1 = l1_loss(image, gt_image)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            image_t = image.clone()
            gt_image_t = gt_image.clone()

        if iteration > lpips_start_iter:
            # mask mouth
            [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
            loss += 0.01 * lpips_criterion(image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1, gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1).mean()

            image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
            gt_image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]

            patch_size = random.randint(32, 48) * 2
            loss += 0.2 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()

        # V23: Sobel gradient L1 (anti-smoothing).  Active after warm_step so
        # face has rough Gaussians in place; weight ramps from 0.2 to 0.5.
        if iteration > warm_step and iteration < bg_iter:
            sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device='cuda').view(1,1,3,3)
            sobel_y = sobel_x.transpose(2,3)
            kx = sobel_x.expand(3,1,3,3).contiguous()
            ky = sobel_y.expand(3,1,3,3).contiguous()
            _w = 0.2 + 0.3 * min(1.0, (iteration - warm_step) / 20000.0)
            pgx = F.conv2d(image_white[None], kx, padding=1, groups=3)
            pgy = F.conv2d(image_white[None], ky, padding=1, groups=3)
            tgx = F.conv2d(gt_image_white[None], kx, padding=1, groups=3)
            tgy = F.conv2d(gt_image_white[None], ky, padding=1, groups=3)
            loss = loss + _w * ((pgx - tgx).abs().mean() + (pgy - tgy).abs().mean())

        # V23: features_dc anchor — capture once at iter == warm_step + 1000
        # then penalise drift to keep skin tone high-freq from collapsing.
        if iteration == warm_step + 1000 and feat_anchor_init is None:
            feat_anchor_init = gaussians._features_dc.detach().clone()
            print(f"[v3] features_dc anchor snapshot @ iter {iteration}")
        if feat_anchor_init is not None:
            # match shape (densify may have grown _features_dc); only anchor
            # if size still matches.
            if gaussians._features_dc.shape == feat_anchor_init.shape:
                loss = loss + 0.005 * (gaussians._features_dc - feat_anchor_init).abs().mean()
            else:
                # densify changed point count → recapture anchor
                feat_anchor_init = gaussians._features_dc.detach().clone()

        # V30: NaN guard + grad clip. V28 face explosion (NaN at iter ~30k) was
        # traced to runaway gradients during densification + lip_rect 3x L1.
        # Skip the step when loss is non-finite, otherwise NaN propagates into
        # _xyz/_scaling/_rotation and the ckpt becomes unrecoverable.
        if not torch.isfinite(loss):
            gaussians.optimizer.zero_grad(set_to_none=True)
            motion_optimizer.zero_grad(set_to_none=True)
            if iteration % 100 == 0:
                print(f"[v30] iter {iteration}: non-finite loss, skipping step")
            iter_end.record()
            with torch.no_grad():
                pass
            continue

        loss.backward()
        # Clip gradients for stability (cross_attn + albedo + motion all share).
        torch.nn.utils.clip_grad_norm_(motion_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(cross_attn_driver.parameters(), 1.0)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"}) # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration)+'_face')

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # V23 5-tuple: 4-tuple base + extras dict (cross_attn_driver)
                ckpt = (gaussians.capture(), motion_net.state_dict(),
                        motion_optimizer.state_dict(), iteration,
                        {'cross_attn_driver': cross_attn_driver.state_dict()})
                torch.save(ckpt, scene.model_path + "/chkpnt_face_v30_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_face_v30_latest.pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

            
            # bg prune
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh

                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                gaussians.prune_points(bg_color_mask.squeeze())


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
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 5)]}, 
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
                        # V23: renderFunc may be standard render_motion (no attn arg)
                        try:
                            render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, *renderArgs)
                        except TypeError:
                            render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)

                        if renderFunc is not render and "attn" in render_pkg:
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)  

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
