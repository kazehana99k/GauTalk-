"""V30 fuse init = subsampled V30 face + V30 mouth + V17 head priors.

Composes a fuse-shape ckpt that train_fuse_v28 can consume:
    (g_face_state, motion_face, g_mouth_state, motion_mouth, extras)

The face Gaussians are opacity-top-K subsampled to face_max_pts (default
50000) to keep fuse iter time tractable; the mouth Gaussians are kept whole.
"""
import argparse
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--face_ckpt", required=True,
                   help="e.g. macrontest/macron_v28/chkpnt_face_v30_clean.pth")
    p.add_argument("--mouth_ckpt", required=True,
                   help="e.g. macrontest/macron_v28/chkpnt_mouth_v30_latest.pth")
    p.add_argument("--head_prior", default="macrontest/macron_v17/chkpnt_fuse_v17_latest.pth",
                   help="V17 fuse ckpt to lift phoneme/albedo/aperture heads from")
    p.add_argument("--out", required=True)
    p.add_argument("--face_max_pts", type=int, default=50000,
                   help="opacity-based subsample cap for face Gaussians")
    args = p.parse_args()

    print(f"[v30-init] face   <- {args.face_ckpt}")
    face_raw = torch.load(args.face_ckpt)
    print(f"[v30-init] mouth  <- {args.mouth_ckpt}")
    mouth_raw = torch.load(args.mouth_ckpt)

    g_face = list(face_raw[0])
    motion_face   = face_raw[1]
    g_mouth_state = mouth_raw[0]
    motion_mouth  = mouth_raw[1]

    # Subsample face Gaussians by opacity if too many.
    n_face = g_face[1].shape[0]
    if n_face > args.face_max_pts:
        opa_logit = g_face[7].squeeze(-1) if g_face[7].dim() == 2 else g_face[7]
        opa = torch.sigmoid(opa_logit.float())
        keep_idx = torch.topk(opa, args.face_max_pts, largest=True).indices
        keep_idx, _ = torch.sort(keep_idx)
        for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            if isinstance(g_face[i], torch.Tensor) and g_face[i].shape[0] == n_face:
                g_face[i] = g_face[i][keep_idx].contiguous()
        # Subsample optimizer state
        opt_sd = g_face[11]
        if isinstance(opt_sd, dict) and 'state' in opt_sd:
            new_state = {}
            for k, v in opt_sd['state'].items():
                nv = dict(v)
                for tk in ('exp_avg', 'exp_avg_sq'):
                    if tk in nv and isinstance(nv[tk], torch.Tensor) \
                            and nv[tk].shape[0] == n_face:
                        nv[tk] = nv[tk][keep_idx].contiguous()
                new_state[k] = nv
            opt_sd['state'] = new_state
            g_face[11] = opt_sd
        print(f"[v30-init] face Gaussians: {n_face} -> {args.face_max_pts} (top-K opacity)")
    g_face = tuple(g_face)

    # Lift V17 heads (phoneme/albedo/aperture) for warm-start.
    extras = {}
    face_extras = face_raw[4] if len(face_raw) >= 5 and isinstance(face_raw[4], dict) else {}
    mouth_extras = mouth_raw[4] if len(mouth_raw) >= 5 and isinstance(mouth_raw[4], dict) else {}
    if 'cross_attn_driver' in face_extras:
        extras['cross_attn_driver'] = face_extras['cross_attn_driver']
    if 'cross_attn_driver_lip' in mouth_extras:
        extras['cross_attn_driver_mouth'] = mouth_extras['cross_attn_driver_lip']
    elif 'cross_attn_driver' in mouth_extras:
        extras['cross_attn_driver_mouth'] = mouth_extras['cross_attn_driver']
    # E2 (2026-05-20): forward mouth-stage albedo_head_mouth to fuse extras.
    if 'albedo_head_mouth' in mouth_extras:
        extras['albedo_head_mouth'] = mouth_extras['albedo_head_mouth']
        extras['albedo_residual_scale_mouth'] = mouth_extras.get('albedo_residual_scale_mouth', 0.35)
        print('[v30-init] forwarded albedo_head_mouth from mouth ckpt')
    if args.head_prior and os.path.exists(args.head_prior):
        v17 = torch.load(args.head_prior, map_location='cpu')
        v17_extras = v17[4] if len(v17) >= 5 and isinstance(v17[4], dict) else {}
        for k in ('phoneme_head', 'albedo_head', 'aperture_aux_head',
                  'audio_dim', 'audio_dim_mouth', 'albedo_residual_scale'):
            if k in v17_extras:
                extras[k] = v17_extras[k]
        print(f"[v30-init] heads <- {args.head_prior}")

    out_tuple = (g_face, motion_face, g_mouth_state, motion_mouth, extras)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(out_tuple, args.out)
    n_face_out = g_face[1].shape[0]
    n_mouth = g_mouth_state[1].shape[0]
    print(f"[v30-init] saved -> {args.out}")
    print(f"[v30-init]   face Gaussians: {n_face_out}  mouth Gaussians: {n_mouth}")
    print(f"[v30-init]   extras: {sorted(extras.keys())}")


if __name__ == "__main__":
    main()
