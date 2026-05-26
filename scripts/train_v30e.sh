#!/usr/bin/env bash
# V30e = dual-head mouth cross-attn AT FUSE TIME (and at synthesis), mirroring
# the V30 mouth training distribution. Cavity Gaussians are driven by the
# cavity head (AU25/AU26 only, residual_scale 2e-3); lip Gaussians by the lip
# head (full AU17, residual_scale 1e-3). Restores temporal coherence by
# eliminating the train-vs-fuse head distribution mismatch that V30b had.
set -e

dataset=$1
workspace=$2
gpu_id=$3
audio_extractor='hubert'
fuse_iters=${4:-5000}

export CUDA_VISIBLE_DEVICES=$gpu_id
export TG_LIP_CAVITY=1

mouth_ckpt="$workspace/chkpnt_mouth_v30_latest.pth"
v17_fuse="macrontest/macron_v17/chkpnt_fuse_v17_latest.pth"

if [ ! -f "$mouth_ckpt" ]; then
    echo "[V30e] ERROR: $mouth_ckpt not found." >&2; exit 1
fi
if [ ! -f "$v17_fuse" ]; then
    echo "[V30e] ERROR: $v17_fuse not found." >&2; exit 1
fi

# 1) Build dual-head init: V17 face + V30 mouth (both heads) + V17 head priors.
init_ckpt="$workspace/chkpnt_fuse_v30e_init.pth"
python scripts/build_fuse_v30e_init.py \
    --v17_fuse "$v17_fuse" \
    --mouth_ckpt "$mouth_ckpt" \
    --out "$init_ckpt"

# 2) Fuse with dual-head mouth cross-attn (train_fuse_v30e).
python train_fuse_v30e.py -s "$dataset" -m "$workspace" \
    --init_ckpt "$init_ckpt" \
    --opacity_lr 0.001 --audio_extractor $audio_extractor \
    --total_iters $fuse_iters --au_window_T 8 --aperture_w 0.2 \
    --detail_w 0.5 --feat_anchor_w 0.005 \
    --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0

# 3) Render with dual-head synth + eval.
python synthesize_fuse_v30e.py -s "$dataset" -m "$workspace" \
    --eval --audio_extractor $audio_extractor \
    --ckpt_name chkpnt_fuse_v30e_latest.pth \
    --output_dir "$workspace/render_v30e_full" --max_frames 9999 --au_window_T 8
python scripts/eval_v17_full.py "$workspace/render_v30e_full/seq_test"
