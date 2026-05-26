"""V30e fuse init = V17 face foundation + V30 mouth (Gaussians + motion +
BOTH cross_attn driver heads: lip + cavity) + V17 head priors.

train_fuse_v30e expects extras to carry both 'cross_attn_driver_mouth_lip'
and 'cross_attn_driver_mouth_cavity' so the dual-head split mirrors the V30
mouth training distribution at fuse time and inference time (synthesize).
"""
import argparse
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v17_fuse", default="macrontest/macron_v17/chkpnt_fuse_v17_latest.pth",
                   help="provides V17 face Gaussians/motion + face cross_attn + heads")
    p.add_argument("--mouth_ckpt", required=True,
                   help="V30 mouth ckpt with dual-head extras "
                        "(e.g. macrontest/macron_v30/chkpnt_mouth_v30_latest.pth)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[v30e-init] face/heads <- {args.v17_fuse}")
    v17 = torch.load(args.v17_fuse)
    print(f"[v30e-init] mouth      <- {args.mouth_ckpt}")
    m30 = torch.load(args.mouth_ckpt)

    g_face       = v17[0]
    motion_face  = v17[1]
    g_mouth_new  = m30[0]
    motion_mouth_new = m30[1]
    v17_extras   = v17[4] if len(v17) >= 5 and isinstance(v17[4], dict) else {}
    m30_extras   = m30[4] if len(m30) >= 5 and isinstance(m30[4], dict) else {}

    extras = dict(v17_extras)

    # Lift both mouth heads from V30. Falls back gracefully if only one is present.
    if 'cross_attn_driver_lip' in m30_extras:
        extras['cross_attn_driver_mouth_lip'] = m30_extras['cross_attn_driver_lip']
        # Also seed the legacy single-head key with the lip head for any
        # downstream code that doesn't know about dual-head yet.
        extras['cross_attn_driver_mouth']     = m30_extras['cross_attn_driver_lip']
    elif 'cross_attn_driver' in m30_extras:
        extras['cross_attn_driver_mouth_lip'] = m30_extras['cross_attn_driver']
        extras['cross_attn_driver_mouth']     = m30_extras['cross_attn_driver']
    if 'cross_attn_driver_cavity' in m30_extras:
        extras['cross_attn_driver_mouth_cavity'] = m30_extras['cross_attn_driver_cavity']
    else:
        # Fallback: use lip head as cavity bootstrap so the cavity slot isn't empty.
        if 'cross_attn_driver_mouth_lip' in extras:
            extras['cross_attn_driver_mouth_cavity'] = extras['cross_attn_driver_mouth_lip']
            print("[v30e-init]   WARN: V30 mouth lacks cavity head; bootstrapping from lip head")

    out_tuple = (g_face, motion_face, g_mouth_new, motion_mouth_new, extras)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(out_tuple, args.out)

    n_face_pts  = g_face[1].shape[0] if hasattr(g_face[1], 'shape') else '?'
    n_mouth_pts = g_mouth_new[1].shape[0] if hasattr(g_mouth_new[1], 'shape') else '?'
    print(f"[v30e-init] saved -> {args.out}")
    print(f"[v30e-init]   face Gaussians (V17): {n_face_pts}")
    print(f"[v30e-init]   mouth Gaussians (V30): {n_mouth_pts}")
    print(f"[v30e-init]   extras: {sorted(extras.keys())}")


if __name__ == "__main__":
    main()
