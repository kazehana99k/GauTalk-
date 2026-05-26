"""Lip / cavity / teeth refinement on top of EasyPortrait FPN-FP (8-class).

Outputs (per frame, .npy bool, same H,W as ori_imgs/<id>.jpg):
  <data>/lip_mask/<id>.npy        red-lips ONLY (outer + inner vermilion border)
  <data>/cavity_mask/<id>.npy     teeth + tongue + dark inner mouth
  <data>/teeth_mask_v2/<id>.npy   alias of cavity_mask, for downstream compatibility

Pipeline (per frame, no extra model deps):
  1. Run EasyPortrait FPN-FP -> seg in {0..7}; lip_raw = (seg==6), teeth_raw = (seg==7).
  2. LAB-color reclaim inside the union bbox:
       - tau_a = (mean(a* | lip_raw) + mean(a* | teeth_raw)) / 2
       - move teeth pixels with a* > tau_a back to lip (red over-classified as teeth)
       - move lip pixels with a* < tau_a AND L* > 60 to teeth (rare)
  3. Outer-lip convex hull from refined lip pixels:
       hull = convexHull(lip_refined)
       inner = hull AND NOT lip_refined
       cavity = (inner AND (teeth_refined OR L* < L_thr)) OR teeth_refined
  4. Enforce mutex: lip_mask = lip_refined AND NOT cavity
  5. Morphology: lip 1px open, cavity 2px open
  6. Optional 1-D temporal mean over a +/-K window (off by default).
"""

from argparse import ArgumentParser
import os
import glob
import re

import numpy as np
import cv2
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor


LIP_CLASS = 6
TEETH_CLASS = 7
L_DARK_THR = 35.0       # L* threshold for "dark cavity" pixels
LIP_L_HIGH = 60.0       # very bright pixels inside lip mask are likely teeth glare


def _frame_id(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def _refine_one(image_bgr, seg):
    """Take BGR image and EasyPortrait segmentation, return (lip, cavity) bool masks."""
    H, W = seg.shape
    lip_raw = (seg == LIP_CLASS)
    teeth_raw = (seg == TEETH_CLASS)

    if lip_raw.sum() == 0 and teeth_raw.sum() == 0:
        return np.zeros_like(lip_raw), np.zeros_like(lip_raw)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0  # signed a*: positive -> red

    lip_refined = lip_raw.copy()
    teeth_refined = teeth_raw.copy()

    # --- Step 2: LAB-based reclaim ---
    if lip_raw.sum() > 30 and teeth_raw.sum() > 10:
        a_lip = a[lip_raw].mean()
        a_teeth = a[teeth_raw].mean()
        tau_a = 0.5 * (a_lip + a_teeth)
        # only act when there is a real separation
        if a_lip - a_teeth > 5.0:
            reclaim_to_lip = teeth_raw & (a > tau_a)
            donate_to_teeth = lip_raw & (a < tau_a) & (L > LIP_L_HIGH)
            lip_refined = (lip_raw | reclaim_to_lip) & (~donate_to_teeth)
            teeth_refined = (teeth_raw & (~reclaim_to_lip)) | donate_to_teeth

    # --- Step 3: outer-lip convex hull -> inner cavity expansion ---
    cavity = teeth_refined.copy()
    ys, xs = np.where(lip_refined)
    if len(xs) >= 8:
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        try:
            hull = cv2.convexHull(pts)
            hull_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            hull_mask = hull_mask.astype(bool)
            inner = hull_mask & (~lip_refined)
            dark = L < L_DARK_THR
            cavity = (inner & (dark | teeth_refined)) | teeth_refined
        except cv2.error:
            pass

    # --- Step 4: mutex (cavity wins inside its own pixels; lip keeps the rest) ---
    lip_mask = lip_refined & (~cavity)

    # --- Step 5: morphological cleanup ---
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lip_mask = cv2.morphologyEx(lip_mask.astype(np.uint8), cv2.MORPH_OPEN, k1).astype(bool)
    cavity = cv2.morphologyEx(cavity.astype(np.uint8), cv2.MORPH_OPEN, k2).astype(bool)

    # final mutex re-assert (open can grow regions)
    overlap = lip_mask & cavity
    if overlap.any():
        cavity = cavity & (~overlap)

    return lip_mask, cavity


def _temporal_smooth(masks_bool, K):
    """1D box-mean smoothing along frame axis. masks: (T,H,W) bool."""
    if K <= 0:
        return masks_bool
    T = masks_bool.shape[0]
    out = np.zeros_like(masks_bool)
    f = masks_bool.astype(np.float32)
    csum = np.concatenate([np.zeros_like(f[:1]), np.cumsum(f, axis=0)], axis=0)  # (T+1,H,W)
    for t in range(T):
        lo = max(0, t - K)
        hi = min(T, t + K + 1)
        win = csum[hi] - csum[lo]
        avg = win / float(hi - lo)
        out[t] = avg > 0.5
    return out


def main():
    p = ArgumentParser()
    p.add_argument('dataset', help='dataset root, e.g. data/macron')
    p.add_argument('--config', default="./data_utils/easyportrait/local_configs/easyportrait_experiments_v2/fpn-fp/fpn-fp.py")
    p.add_argument('--checkpoint', default="./data_utils/easyportrait/fpn-fp-512.pth")
    p.add_argument('--temporal_smooth', type=int, default=0,
                   help='moving window radius (0 = off). 1 -> +/-1 frames')
    p.add_argument('--limit', type=int, default=0, help='process only first N frames (debug)')
    args = p.parse_args()

    img_dir = os.path.join(args.dataset, 'ori_imgs')
    lip_dir = os.path.join(args.dataset, 'lip_mask')
    cav_dir = os.path.join(args.dataset, 'cavity_mask')
    teeth2_dir = os.path.join(args.dataset, 'teeth_mask_v2')
    os.makedirs(lip_dir, exist_ok=True)
    os.makedirs(cav_dir, exist_ok=True)
    os.makedirs(teeth2_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')),
                   key=lambda f: _frame_id(f) if _frame_id(f) is not None else 1 << 30)
    if args.limit:
        files = files[:args.limit]
    print(f'[lip-cavity] {len(files)} frames in {img_dir}')

    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    if args.temporal_smooth > 0:
        # buffered pass: we need all masks before smoothing
        lip_buf = []
        cav_buf = []
        ids = []
        for f in tqdm(files, desc='infer'):
            seg = inference_segmentor(model, f)[0]
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            lip, cav = _refine_one(img, seg)
            lip_buf.append(lip)
            cav_buf.append(cav)
            ids.append(_frame_id(f))
        lip_arr = np.stack(lip_buf, axis=0)
        cav_arr = np.stack(cav_buf, axis=0)
        del lip_buf, cav_buf
        lip_arr = _temporal_smooth(lip_arr, args.temporal_smooth)
        cav_arr = _temporal_smooth(cav_arr, args.temporal_smooth)
        # re-assert mutex
        overlap = lip_arr & cav_arr
        if overlap.any():
            cav_arr[overlap] = False
        for i, fid in enumerate(tqdm(ids, desc='save')):
            np.save(os.path.join(lip_dir, f'{fid}.npy'), lip_arr[i])
            np.save(os.path.join(cav_dir, f'{fid}.npy'), cav_arr[i])
            np.save(os.path.join(teeth2_dir, f'{fid}.npy'), cav_arr[i])
    else:
        # streaming pass
        for f in tqdm(files, desc='infer+save'):
            seg = inference_segmentor(model, f)[0]
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            lip, cav = _refine_one(img, seg)
            fid = _frame_id(f)
            np.save(os.path.join(lip_dir, f'{fid}.npy'), lip)
            np.save(os.path.join(cav_dir, f'{fid}.npy'), cav)
            np.save(os.path.join(teeth2_dir, f'{fid}.npy'), cav)

    print(f'[lip-cavity] done. lip -> {lip_dir}, cavity -> {cav_dir}, teeth_v2 -> {teeth2_dir}')


if __name__ == '__main__':
    main()
