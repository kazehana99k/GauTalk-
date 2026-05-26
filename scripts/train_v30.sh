#!/usr/bin/env bash
# V30 = full pipeline retrain with ABC modifications baked in from FACE/MOUTH
# stages onward. Path A's V17-face foundation is replaced by a V30 face that
# incorporates the new lip/cavity mask supervision + lip 2x weighted L1 +
# NaN-safe training loop.
#
# Stages:
#   1. face_v30 (25k iter): cross-attn from iter 0 + lip 2x weighted L1 +
#      NaN guard + grad clip + tighter densify (0.0015 vs V28's 0.001).
#   2. mouth_v30 (50k iter): cavity 3x weighted L1 + cavity depth prior +
#      cavity/lip dual-head cross-attn (cavity_idx recomputed after each
#      densify_and_prune so the dual head stays valid).
#   3. fuse_v30 (10k iter): hybrid init (V30 face top-50k subsample + V30
#      mouth + V17 phoneme/albedo/aperture heads), V25 recipe losses.
#   4. render full + eval.
set -e

dataset=$1
workspace=$2
gpu_id=$3
audio_extractor='hubert'

export CUDA_VISIBLE_DEVICES=$gpu_id
export TG_LIP_CAVITY=1

# Sanity
if [ ! -d "$dataset/lip_mask" ] || [ ! -d "$dataset/cavity_mask" ]; then
    echo "[V30] ERROR: $dataset/lip_mask or cavity_mask missing." >&2
    exit 1
fi

# 1) face_v30: 25k iter (V28 face went NaN at 30k -> stop earlier with NaN guard).
python train_face_v30.py -s "$dataset" -m "$workspace" \
    --init_num 2000 --densify_grad_threshold 0.0015 \
    --audio_extractor $audio_extractor \
    --iterations 25000

# Snapshot the latest clean face ckpt (V30 has NaN guard so latest should be clean).
cp "$workspace/chkpnt_face_v30_latest.pth" "$workspace/chkpnt_face_v30_clean.pth"

# 2) mouth_v30: full 50k retrain from scratch with cavity 3x + depth + dual-head.
python train_mouth_v30.py -s "$dataset" -m "$workspace" \
    --audio_extractor $audio_extractor \
    --iterations 50000

# 3) Build hybrid fuse init: V30 face (top-50k) + V30 mouth + V17 head priors.
init_ckpt="$workspace/chkpnt_fuse_v30_init.pth"
python scripts/build_fuse_v30_init.py \
    --face_ckpt "$workspace/chkpnt_face_v30_clean.pth" \
    --mouth_ckpt "$workspace/chkpnt_mouth_v30_latest.pth" \
    --head_prior "macrontest/macron_v17/chkpnt_fuse_v17_latest.pth" \
    --face_max_pts 50000 \
    --out "$init_ckpt"

# 4) Fuse from hybrid init using train_fuse_v28 (V25 recipe, save tag v28).
python train_fuse_v28.py -s "$dataset" -m "$workspace" \
    --init_ckpt "$init_ckpt" \
    --opacity_lr 0.001 --audio_extractor $audio_extractor \
    --total_iters 10000 --au_window_T 8 --aperture_w 0.2 \
    --detail_w 0.5 --feat_anchor_w 0.005 \
    --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0

# Rename fuse ckpt to V30 tag.
if [ -f "$workspace/chkpnt_fuse_v28_latest.pth" ]; then
    cp "$workspace/chkpnt_fuse_v28_latest.pth" "$workspace/chkpnt_fuse_v30_latest.pth"
    echo "[V30] saved $workspace/chkpnt_fuse_v30_latest.pth"
fi

# 5) Render full + eval.
python synthesize_fuse_v18.py -s "$dataset" -m "$workspace" \
    --eval --audio_extractor $audio_extractor \
    --ckpt_name chkpnt_fuse_v30_latest.pth \
    --output_dir "$workspace/render_v30_full" --max_frames 9999 --au_window_T 8
python scripts/eval_v17_full.py "$workspace/render_v30_full/seq_test"
