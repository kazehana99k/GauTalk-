"""V17 full evaluation: PSNR / SSIM / LPIPS-alex / LPIPS-vgg / LMD on the 128
pred+gt frames written by synthesize_fuse_clean.py.

Usage:
    python scripts/eval_v17_full.py macrontest/macron_v17/render_v17/seq_test
"""
import os, sys, glob, re, numpy as np, torch, cv2
import imageio.v2 as imageio
import lpips
from skimage.metrics import structural_similarity as ssim_fn

seq_dir = sys.argv[1]
device = torch.device("cuda")

pred_files = sorted(glob.glob(os.path.join(seq_dir, "frame_*_pred.png")),
                    key=lambda p: int(re.search(r'frame_(\d+)_', p).group(1)))
gt_files = [p.replace("_pred.png", "_gt.png") for p in pred_files]
assert all(os.path.exists(g) for g in gt_files), "GT missing"

print(f"[eval] {len(pred_files)} frame pairs in {seq_dir}")

lp_alex = lpips.LPIPS(net='alex').eval().to(device)
lp_vgg  = lpips.LPIPS(net='vgg').eval().to(device)

import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

def to_torch(img):
    return torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0

psnr_v, ssim_v, alex_v, vgg_v, lmd_v, lmd_n = 0., 0., 0., 0., 0., 0
for i, (pf, gf) in enumerate(zip(pred_files, gt_files)):
    pred = imageio.imread(pf)
    gt   = imageio.imread(gf)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

    # PSNR
    mse = ((pred.astype(np.float32) - gt.astype(np.float32)) ** 2).mean() / (255 ** 2)
    psnr_v += 10 * np.log10(1.0 / max(mse, 1e-12))

    # SSIM (luminance, multichannel)
    ssim_v += ssim_fn(gt, pred, channel_axis=2, data_range=255)

    # LPIPS (need [-1, 1])
    p = to_torch(pred).to(device) * 2 - 1
    g = to_torch(gt).to(device) * 2 - 1
    with torch.no_grad():
        alex_v += lp_alex(p, g).item()
        vgg_v  += lp_vgg(p, g).item()

    # LMD on mouth (landmarks 48-68)
    try:
        lp_pts = fa.get_landmarks(pred)
        lg_pts = fa.get_landmarks(gt)
        if lp_pts is not None and lg_pts is not None:
            mp = lp_pts[-1][48:68]
            mg = lg_pts[-1][48:68]
            lmd_v += np.linalg.norm(mp - mg, axis=1).mean()
            lmd_n += 1
    except Exception:
        pass

    if (i + 1) % 32 == 0:
        print(f"  {i+1}/{len(pred_files)} done")

n = len(pred_files)
print()
print("=" * 60)
print(f"V17 (Soft Mask Boundary) — {n} test frames")
print("=" * 60)
print(f"  PSNR        : {psnr_v / n:.4f} dB")
print(f"  SSIM        : {ssim_v / n:.4f}")
print(f"  LPIPS (alex): {alex_v / n:.4f}")
print(f"  LPIPS (vgg) : {vgg_v / n:.4f}")
if lmd_n > 0:
    print(f"  LMD (mouth) : {lmd_v / lmd_n:.4f}  ({lmd_n}/{n} frames had landmarks)")
print("=" * 60)
