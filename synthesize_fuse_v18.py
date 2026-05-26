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


@torch.no_grad()
def render_one_v18(view, gaussians, motion_net, gaussians_mouth, motion_net_mouth,
                   pipe, cross_attn_driver, cross_attn_driver_mouth,
                   au_full, au_window_T):
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

    # Mouth cross-attn
    xyz_feat_m = motion_net_mouth.encode_x(gaussians_mouth.get_xyz, bound=motion_net_mouth.bound)
    audio_seq_m = motion_net_mouth.audio_net(audio_raw)
    if audio_seq_m.dim() == 1:
        audio_seq_m = audio_seq_m.unsqueeze(0)
    cross_attn_residual_mouth = cross_attn_driver_mouth(xyz_feat_m, audio_seq_m, au17_window)

    # R-DIAG-MOUTH-XATTN: TG_MOUTH_XATTN_GAIN scales mouth cross_attn residual.
    # Set to 0 to disable (test if cross_attn_driver_mouth causes smear at transition frames).
    import os as _os_xa
    _xa_gain = float(_os_xa.environ.get('TG_MOUTH_XATTN_GAIN', '1.0'))
    if _xa_gain != 1.0:
        for _k in list(cross_attn_residual_mouth.keys()):
            if torch.is_tensor(cross_attn_residual_mouth[_k]):
                cross_attn_residual_mouth[_k] = cross_attn_residual_mouth[_k] * _xa_gain

    # AU25-LIP-AMP (gated, inference-time only): diff-trick amplifier for AU25→mouth-open.
    # Mouth Gaussians mostly follow audio (motion_net_mouth has no AU input); AU
    # influence only enters via cross_attn_driver_mouth which has small residual scale.
    # At apex AU25 frames, mouth opens less than GT. Boost AU25's contribution to
    # cross_attn_residual_mouth['d_xyz']. Set TG_AU25_LIP_GAIN > 1 to enable.
    import os as _os
    _au25_gain = float(_os.environ.get('TG_AU25_LIP_GAIN', '1.0'))
    if _au25_gain > 1.0 and au17_window is not None:
        _au17_no25 = au17_window.clone()
        # AU index map: [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45] → AU25 at idx 14, AU26 at idx 15
        if _au17_no25.dim() == 1:
            _au17_no25[14] = 0.0; _au17_no25[15] = 0.0
        else:
            _au17_no25[..., 14] = 0.0; _au17_no25[..., 15] = 0.0
        _resid_no25 = cross_attn_driver_mouth(xyz_feat_m, audio_seq_m, _au17_no25)
        for _k in ('d_xyz', 'd_rot', 'd_scale'):
            if _k in cross_attn_residual_mouth and _k in _resid_no25:
                _diff = cross_attn_residual_mouth[_k] - _resid_no25[_k]
                cross_attn_residual_mouth[_k] = cross_attn_residual_mouth[_k] + (_au25_gain - 1.0) * _diff

    zero_alb = torch.zeros((gaussians.get_xyz.shape[0], 3), device="cuda")
    image, _, _ = render_fuse_v18(
        view, gaussians, motion_net,
        gaussians_mouth, motion_net_mouth, pipe, bg,
        albedo_residual=zero_alb,
        cross_attn_residual=cross_attn_residual,
        cross_attn_residual_mouth=cross_attn_residual_mouth,
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
    if 'cross_attn_driver_mouth' in extras:
        cross_attn_driver_mouth.load_state_dict(extras['cross_attn_driver_mouth'], strict=False)
        print("[v18 synth] cross_attn_driver_mouth loaded")
    else:
        print("[v18 synth] WARN: no mouth cross-attn in ckpt; rendering with random init driver")

    # R-DATA-5: read TG_AU_CSV so inference AU17 source matches training.
    au_full = None
    try:
        _au_csv_name = os.environ.get('TG_AU_CSV', 'au.csv')
        df = pd.read_csv(os.path.join(dataset.source_path, _au_csv_name))
        print(f"[R-DATA-5] synthesize au_full loaded from {_au_csv_name}")
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
                             au_full, args.au_window_T)
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
