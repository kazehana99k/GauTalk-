"""V18 inference: face cross-attn + mouth cross-attn (parallel transformers).

Mirrors render_fuse_v18 from train_fuse_v18.  Both face and mouth Gaussians
receive their own cross-attn residual on top of motion_net outputs, and the
mouth audio_net was unfrozen during V18 training so the aperture aux head can
flow back into it.
"""
import os, copy, sys
import numpy as np
import torch
import imageio.v2 as imageio
import pandas as pd
from tqdm import tqdm
from os import makedirs

from scene import Scene, GaussianModel, MotionNetwork, MouthMotionNetwork
from utils.general_utils import safe_state
from utils.camera_utils import loadCamOnTheFly
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from models.cross_attn_driver import GaussianCrossAttnDriver
from train_fuse_v18 import render_fuse_v18


def gather_au_window(au_full, img_id, T):
    half = T // 2
    lo, hi = img_id - half, img_id + (T - half)
    l_pad = max(0, -lo); r_pad = max(0, hi - au_full.shape[0])
    lo_c = max(0, lo); hi_c = min(au_full.shape[0], hi)
    seq = au_full[lo_c:hi_c]
    if l_pad: seq = torch.cat([seq[:1].expand(l_pad, -1), seq], dim=0)
    if r_pad: seq = torch.cat([seq, seq[-1:].expand(r_pad, -1)], dim=0)
    return seq.float()


_CAVITY_AU_MASK_SYNTH = torch.zeros(17, device='cuda')
_CAVITY_AU_MASK_SYNTH[14] = 1.0  # AU25
_CAVITY_AU_MASK_SYNTH[15] = 1.0  # AU26


@torch.no_grad()
def render_one_v18(view, gaussians, motion_net, gaussians_mouth, motion_net_mouth,
                   pipe, cross_attn_driver, cross_attn_driver_mouth,
                   au_full, au_window_T,
                   cross_attn_driver_mouth_cavity=None,
                   cavity_idx_mouth=None,
                   albedo_head_mouth=None):
    bg = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")
    audio_raw = view.talking_dict['auds'].cuda()
    au17 = view.talking_dict.get('au_exp17', torch.zeros(17)).to("cuda").float()

    img_id = int(view.talking_dict.get('img_id', -1))
    au17_window = au17 if au_full is None or img_id < 0 else gather_au_window(au_full, img_id, au_window_T)

    # Face cross-attn (skip when driver wasn't trained for this face ckpt).
    cross_attn_residual = None
    if cross_attn_driver is not None:
        xyz_feat = motion_net.encode_x(gaussians.get_xyz, bound=motion_net.bound)
        audio_seq = motion_net.audio_net(audio_raw)
        if audio_seq.dim() == 1:
            audio_seq = audio_seq.unsqueeze(0)
        cross_attn_residual = cross_attn_driver(xyz_feat, audio_seq, au17_window)

    # Mouth cross-attn (V30E: dual-head split when cavity_idx + cavity head are available).
    xyz_feat_m = motion_net_mouth.encode_x(gaussians_mouth.get_xyz, bound=motion_net_mouth.bound)
    audio_seq_m = motion_net_mouth.audio_net(audio_raw)
    if audio_seq_m.dim() == 1:
        audio_seq_m = audio_seq_m.unsqueeze(0)
    if cross_attn_driver_mouth_cavity is not None and cavity_idx_mouth is not None \
            and cavity_idx_mouth.shape[0] == xyz_feat_m.shape[0] \
            and int(cavity_idx_mouth.sum().item()) > 8 \
            and int((~cavity_idx_mouth).sum().item()) > 8:
        xf_lip = xyz_feat_m[~cavity_idx_mouth]
        xf_cav = xyz_feat_m[cavity_idx_mouth]
        au_lip_in = au17_window
        au_cav_in = au17_window * _CAVITY_AU_MASK_SYNTH if au17_window.dim() == 1 \
            else au17_window * _CAVITY_AU_MASK_SYNTH[None, :]
        res_lip = cross_attn_driver_mouth(xf_lip, audio_seq_m, au_lip_in)
        res_cav = cross_attn_driver_mouth_cavity(xf_cav, audio_seq_m, au_cav_in)
        N = xyz_feat_m.shape[0]
        cross_attn_residual_mouth = {}
        for k in res_lip:
            if isinstance(res_lip[k], torch.Tensor) and res_lip[k].dim() >= 1 \
                    and res_lip[k].shape[0] == xf_lip.shape[0]:
                merged = torch.zeros((N,) + tuple(res_lip[k].shape[1:]),
                                      device=res_lip[k].device, dtype=res_lip[k].dtype)
                merged[~cavity_idx_mouth] = res_lip[k]
                merged[cavity_idx_mouth]  = res_cav[k]
                cross_attn_residual_mouth[k] = merged
            else:
                cross_attn_residual_mouth[k] = res_lip[k]
    else:
        cross_attn_residual_mouth = cross_attn_driver_mouth(xyz_feat_m, audio_seq_m, au17_window)

    zero_alb = torch.zeros((gaussians.get_xyz.shape[0], 3), device="cuda")
    # D2: mouth albedo residual at inference. Uses single-frame au17 (not the
    # temporal window) to match the training call signature.
    albedo_residual_mouth = None
    if albedo_head_mouth is not None:
        audio_emb_mouth = motion_net_mouth.encode_audio(audio_raw)
        albedo_residual_mouth = albedo_head_mouth(gaussians_mouth.get_xyz,
                                                   audio_emb_mouth.squeeze(0), au17)
    # Q1/R3: predicted landmark + mouth_area amplifier for this view
    _q1_lmk = None; _r3_amp = None
    _q1_global = globals().get('_q1_landmark_predictor_fn', None)
    _r3_global = globals().get('_r3_amp_predictor_fn', None)
    _fid_now = int(view.talking_dict.get('img_id', -1))
    if _fid_now >= 0:
        if _q1_global is not None:
            _q1_lmk = _q1_global(_fid_now)
        if _r3_global is not None:
            _r3_amp = _r3_global(_fid_now)
    image, _, _ = render_fuse_v18(
        view, gaussians, motion_net,
        gaussians_mouth, motion_net_mouth, pipe, bg,
        albedo_residual=zero_alb,
        cross_attn_residual=cross_attn_residual,
        cross_attn_residual_mouth=cross_attn_residual_mouth,
        landmark=_q1_lmk,
        motion_amplifier=_r3_amp,
        albedo_residual_mouth=albedo_residual_mouth,
    )
    return image.clamp(0, 1)


def main():
    parser = ArgumentParser(description="V18 fuse synthesizer (face + mouth cross-attn)")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--ckpt_name", type=str, default="chkpnt_fuse_v18_latest.pth")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--au_window_T", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    setattr(dataset, "au_editor_mode", True)

    # Q1: set up landmark predictor as global helper for render fn
    try:
        import numpy as _np
        from models.audio_to_landmark import AudioToLipLandmark
        _q1_ckpt_p = 'models/ckpts/audio_to_landmark_obama.pth'
        if os.path.exists(_q1_ckpt_p):
            _q1_state = torch.load(_q1_ckpt_p)
            _q1_w = _q1_state.get('window', 3)
            _q1_pred = AudioToLipLandmark(window=_q1_w).cuda()
            _q1_pred.load_state_dict(_q1_state['model'])
            _q1_pred.eval()
            _q1_aud = _np.load(os.path.join(dataset.source_path, 'aud_hu.npy')).astype(_np.float32)
            def _q1_landmark_predictor_fn(fid):
                w = _q1_w; N = _q1_aud.shape[0]
                lo = max(0, fid - w); hi = min(N, fid + w + 1)
                l_pad = max(0, w - fid); r_pad = max(0, fid + w + 1 - N)
                seq = _q1_aud[lo:hi]
                if l_pad > 0: seq = _np.concatenate([_np.tile(seq[:1], (l_pad, 1, 1)), seq], axis=0)
                if r_pad > 0: seq = _np.concatenate([seq, _np.tile(seq[-1:], (r_pad, 1, 1))], axis=0)
                seq_t = torch.from_numpy(seq).unsqueeze(0).cuda()
                with torch.no_grad():
                    return _q1_pred(seq_t)[0]
            globals()['_q1_landmark_predictor_fn'] = _q1_landmark_predictor_fn
            print(f"[Q1 synth] landmark predictor loaded, window={_q1_w}")
    except Exception as _e:
        print(f"[Q1 synth] predictor load skipped: {_e}")

    # R3 (2026-05-20): audio→mouth_area predictor for motion amplification
    try:
        import numpy as _np
        from models.audio_to_mouth_area import AudioToMouthArea
        _r3_ckpt_p = 'models/ckpts/audio_to_mouth_area_obama.pth'
        if os.path.exists(_r3_ckpt_p):
            _r3_state = torch.load(_r3_ckpt_p)
            _r3_w = _r3_state.get('window', 3)
            _r3_pred = AudioToMouthArea(window=_r3_w).cuda()
            _r3_pred.load_state_dict(_r3_state['model'])
            _r3_pred.eval()
            _r3_aud = _np.load(os.path.join(dataset.source_path, 'aud_hu.npy')).astype(_np.float32)
            def _r3_amp_predictor_fn(fid):
                w = _r3_w; N = _r3_aud.shape[0]
                lo = max(0, fid - w); hi = min(N, fid + w + 1)
                l_pad = max(0, w - fid); r_pad = max(0, fid + w + 1 - N)
                seq = _r3_aud[lo:hi]
                if l_pad > 0: seq = _np.concatenate([_np.tile(seq[:1], (l_pad, 1, 1)), seq], axis=0)
                if r_pad > 0: seq = _np.concatenate([seq, _np.tile(seq[-1:], (r_pad, 1, 1))], axis=0)
                seq_t = torch.from_numpy(seq).unsqueeze(0).cuda()
                with torch.no_grad():
                    return _r3_pred(seq_t)[0]
            globals()['_r3_amp_predictor_fn'] = _r3_amp_predictor_fn
            print(f"[R3 synth] mouth_area predictor loaded, test_L1={_r3_state.get('test_l1','?'):.3f}, window={_r3_w}")
    except Exception as _e:
        print(f"[R3 synth] predictor load skipped: {_e}")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians_mouth = GaussianModel(dataset.sh_degree)

    motion_net       = MotionNetwork(args=dataset).cuda()
    motion_net_mouth = MouthMotionNetwork(args=dataset).cuda()

    ckpt_path = os.path.join(scene.model_path, args.ckpt_name)
    print(f"[v18 synth] loading {ckpt_path}")
    raw = torch.load(ckpt_path)
    gp, mp, gpm, mpm = raw[0], raw[1], raw[2], raw[3]
    extras = raw[4] if len(raw) >= 5 and isinstance(raw[4], dict) else {}
    gaussians.restore(gp, None)
    motion_net.load_state_dict(mp, strict=False)
    gaussians_mouth.restore(gpm, None)
    motion_net_mouth.load_state_dict(mpm, strict=False)

    audio_dim = motion_net.audio_dim
    cross_attn_driver = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net.in_dim,
        audio_seq_len=8, audio_dim=audio_dim, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=1e-3, residual_scale_scale=1e-3, residual_scale_rot=1e-3,
    ).cuda().eval()
    have_face_attn = 'cross_attn_driver' in extras
    if have_face_attn:
        cross_attn_driver.load_state_dict(extras['cross_attn_driver'], strict=False)
        print("[v18 synth] cross_attn_driver loaded")
    else:
        print("[v18 synth] NO face cross-attn — face renders motion_net only (paper-detail mode)")

    audio_dim_mouth = motion_net_mouth.audio_dim
    cross_attn_driver_mouth = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net_mouth.in_dim,
        audio_seq_len=8, audio_dim=audio_dim_mouth, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=1e-3, residual_scale_scale=1e-3, residual_scale_rot=1e-3,
    ).cuda().eval()
    if 'cross_attn_driver_mouth_lip' in extras:
        cross_attn_driver_mouth.load_state_dict(extras['cross_attn_driver_mouth_lip'], strict=False)
        print("[v30e synth] cross_attn_driver_mouth (lip head) loaded")
    elif 'cross_attn_driver_mouth' in extras:
        cross_attn_driver_mouth.load_state_dict(extras['cross_attn_driver_mouth'], strict=False)
        print("[v30e synth] cross_attn_driver_mouth (single head) loaded")
    else:
        print("[v30e synth] WARN: no mouth cross-attn in ckpt; rendering with random init driver")

    # V30E: cavity head + cavity_idx for dual-head inference.
    cross_attn_driver_mouth_cavity = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net_mouth.in_dim,
        audio_seq_len=8, audio_dim=audio_dim_mouth, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=2e-3, residual_scale_scale=2e-3, residual_scale_rot=2e-3,
    ).cuda().eval()
    have_cavity_head = 'cross_attn_driver_mouth_cavity' in extras
    if have_cavity_head:
        cross_attn_driver_mouth_cavity.load_state_dict(
            extras['cross_attn_driver_mouth_cavity'], strict=False)
        print("[v30e synth] cross_attn_driver_mouth_cavity loaded (dual-head ON)")
    else:
        cross_attn_driver_mouth_cavity = None
        print("[v30e synth] no cavity head in ckpt -> single-head mouth fallback")

    # Compute cavity_idx by projecting mouth Gaussians to highest-cavity train cam.
    cavity_idx_mouth = None
    if cross_attn_driver_mouth_cavity is not None:
        try:
            import glob as _glob
            _cav_files = _glob.glob(os.path.join(dataset.source_path, 'cavity_mask', '*.npy'))
            best_fid = -1; best_cav = -1
            for _f in _cav_files:
                _fid = int(os.path.splitext(os.path.basename(_f))[0])
                _s = int(np.load(_f).sum())
                if _s > best_cav:
                    best_cav = _s; best_fid = _fid
            if best_fid >= 0 and best_cav > 32:
                best_cam = None
                for _cams in (scene.getTrainCameras(), scene.getTestCameras()):
                    for _cam in _cams:
                        if int(_cam.talking_dict.get('img_id', -1)) == best_fid:
                            best_cam = _cam; break
                    if best_cam is not None: break
                if best_cam is not None:
                    if best_cam.original_image is None:
                        from utils.camera_utils import loadCamOnTheFly as _lcotf
                        import copy as _cp
                        best_cam = _lcotf(_cp.deepcopy(best_cam))
                    cav_t = torch.as_tensor(best_cam.talking_dict['cavity_mask']).cuda().bool()
                    xyz = gaussians_mouth.get_xyz.detach()
                    ones = torch.ones(xyz.shape[0], 1, device=xyz.device)
                    homog = torch.cat([xyz, ones], dim=1)
                    ndc = homog @ best_cam.full_proj_transform.cuda()
                    ndc = ndc[:, :3] / ndc[:, 3:4].clamp_min(1e-6)
                    px = ((ndc[:, 0] + 1) * best_cam.image_width  * 0.5).long()
                    py = ((ndc[:, 1] + 1) * best_cam.image_height * 0.5).long()
                    H_, W_ = cav_t.shape
                    valid = (px >= 0) & (px < W_) & (py >= 0) & (py < H_)
                    px_c = px.clamp(0, W_ - 1); py_c = py.clamp(0, H_ - 1)
                    hit = cav_t[py_c, px_c]
                    cavity_idx_mouth = valid & hit
                    n_cav = int(cavity_idx_mouth.sum().item())
                    print(f"[v30e synth] cavity_idx_mouth: {n_cav}/{xyz.shape[0]} "
                          f"(cam fid={best_fid}, cav_pixels={best_cav})")
        except Exception as _e:
            print(f"[v30e synth] cavity_idx compute failed ({_e}); single-head fallback")
            cavity_idx_mouth = None
            cross_attn_driver_mouth_cavity = None

    # D2 (2026-05-20): load albedo_head_mouth if present.
    albedo_head_mouth = None
    if 'albedo_head_mouth' in extras:
        from models.v12_heads import PerGaussianAlbedoMLP
        _r_scale = extras.get('albedo_residual_scale_mouth', 0.35)
        albedo_head_mouth = PerGaussianAlbedoMLP(
            audio_dim=audio_dim_mouth, au_dim=17, hidden=128, residual_scale=_r_scale,
        ).cuda().eval()
        albedo_head_mouth.load_state_dict(extras['albedo_head_mouth'], strict=False)
        print(f"[D2 synth] albedo_head_mouth loaded, residual_scale={_r_scale}")
    else:
        print("[D2 synth] no albedo_head_mouth in ckpt -> SH-only mouth color (legacy)")

    au_full = None
    try:
        df = pd.read_csv(os.path.join(dataset.source_path, "au.csv"))
        ids = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        cols = []
        for i in ids:
            k = ' AU' + str(i).zfill(2) + '_r'
            v = df[k].values if k in df.columns else df[k.strip()].values
            if i == 45: v = v.clip(0, 2)
            cols.append(v[:, None])
        au_full = torch.from_numpy(np.concatenate(cols, axis=-1).astype(np.float32)).cuda()
        print(f"[v18 synth] AU full loaded T={au_full.shape[0]}")
    except Exception as e:
        print(f"[v18 synth] AU pre-load skipped: {e}")

    cams = scene.getTestCameras()[: args.max_frames]
    seq_dir = os.path.join(args.output_dir, "seq_test")
    makedirs(seq_dir, exist_ok=True)

    video_frames = []
    for cam in tqdm(cams, desc="render-v18", ascii=True):
        if cam.original_image is None:
            cam = loadCamOnTheFly(copy.deepcopy(cam))
        img = render_one_v18(cam, gaussians, motion_net, gaussians_mouth, motion_net_mouth,
                             pipe,
                             cross_attn_driver if have_face_attn else None,
                             cross_attn_driver_mouth,
                             au_full, args.au_window_T,
                             cross_attn_driver_mouth_cavity=cross_attn_driver_mouth_cavity,
                             cavity_idx_mouth=cavity_idx_mouth,
                             albedo_head_mouth=albedo_head_mouth)
        gt = cam.original_image.cuda().float() / 255.0
        if gt.max() > 1.5: gt = gt / 255.0
        gt = gt.clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_np  = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_id = int(cam.talking_dict.get("img_id", 0))
        imageio.imwrite(os.path.join(seq_dir, f"frame_{img_id:04d}_pred.png"), img_np)
        imageio.imwrite(os.path.join(seq_dir, f"frame_{img_id:04d}_gt.png"), gt_np)
        video_frames.append(img_np)

    out_mp4 = os.path.join(args.output_dir, "out.mp4")
    imageio.mimwrite(out_mp4, video_frames, fps=25, quality=8, macro_block_size=1)
    print(f"[v18 synth] wrote {out_mp4}")
    print(f"[v18 synth] seq_test frames -> {seq_dir}")


if __name__ == "__main__":
    main()
