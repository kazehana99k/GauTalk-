# GauTalk: Person-Specific 3D Gaussian Talking Head with Retrieval-Based Head Motion

> 🚧 **Work in Progress** — this is an active research project.
> APIs, training scripts, checkpoint names, and environment variables may change without notice.
> See [DESIGN.md](DESIGN.md) for the detailed design of the rendering stack.

[Paper (TBA)] | [Project (TBA)] | [Video (TBA)]

![teaser](assets/main.png)

> 🎞️ Demo video / qualitative comparisons — Coming soon.

GauTalk is a **person-specific talking-head synthesis** pipeline learned from a subject's own
video. It is trained from a per-subject recording (a few minutes) and renders a photorealistic
talking face for novel audio.

The central design choice is that head motion is **not generated — it is selected and replayed**.
For the same audio there are many plausible head trajectories, so regressing one by minimizing
error collapses toward the average and the head barely moves. Instead, GauTalk stores the
subject's own measured motion as short segments (blocks), selects among them as the audio
proceeds, and **replays actual frames from the training video**.

> **This is not the official TalkingGaussian repository.** GauTalk is built on top of
> [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian); see the
> Acknowledgement section for full attribution. A Japanese version of this README is available at
> [`README.md`](README.md).

---

## System Overview

Three driving channels. Audio determines *when* to switch and the direction of expression;
*what* the motion contains comes from the subject's own recorded material.

```
                   ┌─ offline (once, at training time) ───────────────┐
  subject video ──→│ block segmentation → motion memory (blocks +     │
                   │                       compatibility graph)       │
                   └─────────────────────────────────────────────────┘
                                       │
   audio ─┬─→ switching gate ──→ switch times ──┤
          │                                     ↓
          │                      candidate generation → motion selection
          │                                     ↓
          │              sequence of real frame indices (subject's own frames)
          │                                     ↓
          │        head 6-DoF + camera matrix & torso image (locked to that frame)
          │
          ├─→ emotion estimation ──→ expression synthesis ──→ 6 upper-face AUs
          │
          └─→ mouth (driven directly from audio inside the renderer)
                                       ↓
                    trained 3D Gaussian renderer  →  face video
```

- **Head motion (retrieve & replay)** — the subject's video is segmented into short blocks to build
  a motion memory, together with a graph that permits a transition only between blocks whose end
  and start poses are close. At inference the system selects among the legal candidates, and the
  output is **a sequence of real frame indices from the training video rather than motion
  parameters**. Camera matrices and torso images are taken from those same frames, so the motion is
  always physically realizable and stays in character.
- **Upper-face expression** — only 6 upper-face AUs are synthesized (a per-subject base, measured
  events, an audio-derived emotion direction, and blinks, clipped to the subject's own range).
  **The mouth shape is not part of this channel.**
- **Mouth** — driven directly from audio inside the renderer (existing path, unchanged).
- **Rendering** — a 3D Gaussian Splatting renderer built on TalkingGaussian.

---

## Repository Scope

At this point the repository contains the **rendering stack and data preprocessing**.

| | Status |
| --- | --- |
| 3D Gaussian renderer (face / mouth / fuse stages) | ✅ in this repository |
| Data preprocessing (3DMM tracking, parsing, masks, audio features) | ✅ in this repository |
| Training and inference scripts | ✅ in this repository |
| Head-motion planner (block segmentation, motion memory, switching gate, selection) | 🔜 to be released alongside the paper |
| Evaluation and measurement harness | 🔜 same as above |

---

## Installation

Tested on Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1.

```bash
git clone https://github.com/kazehana99k/GauTalk-.git --recursive
cd GauTalk-

conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install facenet-pytorch    # optional, for the ArcFace loss
```

If the `diff-gaussian-rasterization` / `gridencoder` builds fail, see
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[torch-ngp](https://github.com/ashawkey/torch-ngp).

### Preparation

```bash
# 3DMM + face_parsing
bash scripts/prepare.sh

# Put Basel Face Model 2009 (01_MorphableModel.mat) in data_utils/face_tracking/3DMM/
cd data_utils/face_tracking && python convert_BFM.py && cd ../..

# EasyPortrait (for teeth masks)
pip install -U openmim && mim install mmcv-full==1.7.1
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
  -O data_utils/easyportrait/fpn-fp-512.pth

# OpenFace (AU extraction) — follow https://github.com/TadasBaltrusaitis/OpenFace
```

A few dependencies are **not** in `environment.yml` but are needed by the default recipe:

```bash
pip install transformers librosa   # data_utils/hubert.py
pip install timm                   # DINOv2 perceptual loss (--dino_w)
pip install scikit-image           # scripts/eval_v17_full.py
pip install facenet-pytorch        # ArcFace loss (--arc_w; required by the default recipe)
```

**DINOv2 weights** — the default path in `models/perceptual_losses.py` is hard-coded to an
absolute path from the author's machine. If you use `--dino_w > 0`, obtain
`dinov2_vits14_pretrain.pth` and edit that path for your environment.

---

## Usage

### Important Notice

Training videos are assumed to show a **single, front-facing subject against a static background**.
Prepare `data/<ID>/<ID>.mp4` at 25 fps and 512×512.

### Video Dataset

Place the video and its derivatives under `data/<ID>/`. Preprocessing produces
`transforms_train.json`, `ori_imgs/`, `parsing/`, `torso_imgs/`, `gt_imgs/`, `au.csv`, and the
audio-feature `.npy` files.

### Pre-processing Training Video

```bash
# 1. video preprocessing (transforms.json / parsing / landmarks / background image, ...)
python data_utils/process.py data/<ID>/<ID>.mp4

# 2. extract AUs with OpenFace and save to data/<ID>/au.csv
#    make sure AU25_r is included

# 3. teeth mask
python data_utils/easyportrait/create_teeth_mask.py data/<ID>

# 4. lip / cavity 2D masks (required when using TG_LIP_CAVITY=1)
python data_utils/easyportrait/create_lip_cavity_mask.py data/<ID>
```

### Audio Pre-process

```bash
# DeepSpeech
python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>

# HuBERT (select with --audio_extractor hubert)
python data_utils/hubert.py --wav data/<ID>/aud.wav
```

### Train

The current mainline is **v30e** (dual-head mouth fuse).

> ⚠️ **Prerequisite** — the fuse stage requires a pre-trained **V17 fuse checkpoint**
> (`chkpnt_fuse_v17_latest.pth`) as a prior. This is an external asset that cannot be produced
> from the steps in this repository; `scripts/train_v30e.sh` exits with an explicit error if it
> is missing.

```bash
dataset=data/<ID>
work=output/<ID>_v30e
export TG_LIP_CAVITY=1

# A. Face
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 25000
#    →  $work/chkpnt_face_v30_latest.pth

# B. Mouth — enable Plan F / G if needed
#    Plan F: export TG_MOUTH_Z_MAX=0.05
#    Plan G: export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000
#    →  $work/chkpnt_mouth_v30_latest.pth

# C. Fuse init (build the dual-head init from the V17 prior + mouth ckpt)
python scripts/build_fuse_v30e_init.py \
  --v17_fuse <a pre-trained V17 fuse .pth> \
  --mouth_ckpt $work/chkpnt_mouth_v30_latest.pth \
  --out $work/chkpnt_fuse_v30e_init.pth

# D. Fuse training
python train_fuse_v30e.py -s $dataset -m $work \
  --init_ckpt $work/chkpnt_fuse_v30e_init.pth \
  --opacity_lr 0.001 --audio_extractor hubert --total_iters 5000 \
  --au_window_T 8 --aperture_w 0.2 --detail_w 0.5 --feat_anchor_w 0.005 \
  --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0
#    →  $work/chkpnt_fuse_v30e_latest.pth
```

Steps C–D can be run together with [`scripts/train_v30e.sh`](scripts/train_v30e.sh)
(`bash scripts/train_v30e.sh $dataset $work <gpu_id> [fuse_iters]`).
Per-stage loss weights and environment variables are documented in [DESIGN.md](DESIGN.md).

### Test

```bash
python synthesize_fuse_v30e.py -s data/<ID> -m output/<ID>_v30e \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30e_latest.pth \
  --output_dir output/<ID>_v30e/render_v30e_full --max_frames 9999 --au_window_T 8

python scripts/eval_v17_full.py output/<ID>_v30e/render_v30e_full/seq_test
```

> Render v30e weights with **`synthesize_fuse_v30e.py`**. The older `synthesize_fuse_v18.py`
> has no cavity-head branch, so loading dual-head weights into it silently drops cavity driving.

### On head-motion driving

Head-motion retrieval and replay (block selection from the motion memory and real-frame replay)
currently lives **outside this repository** and will be released alongside the paper. The
inference scripts above render the evaluation split, with head poses taken from the training
video. A path that drives head motion from arbitrary audio is not published here yet.

---

## Known Issues

Scripts accumulated during exploratory work have not been fully cleaned up. The following do not
currently work and are being fixed:

- `scripts/train_v30.sh` (the older v30 line) calls a non-existent `train_fuse_v28.py` and stops
  partway through. **Use [`scripts/train_v30e.sh`](scripts/train_v30e.sh) instead.**
- `data_utils/extract_au_openface.py` is an empty file. Run OpenFace's `FeatureExtractor`
  directly and save the result to `data/<ID>/au.csv`.
- `scripts/build_lip_mask_3d.py` imports `train_au_editor.py`, which is not in the repository, so
  it cannot run (the lip 3D mask step can be skipped for now).
- The DINOv2 weight path in `models/perceptual_losses.py`, the `--init_ckpt` default in
  `train_fuse_v30e.py`, and the V17 prior path in `scripts/train_v30e.sh` are hard-coded to
  absolute paths from the author's machine.

## Results

Quantitative results are **not published in this repository while the paper is under review**.
The evaluation protocol, baselines, and numbers will be released together with the paper.

(The preliminary rendering-stage numbers previously listed here have been withdrawn, as the scope
of the project has since changed.)

---

## Citation

```
@misc{gautalk2026,
  title  = {GauTalk: Person-Specific 3D Gaussian Talking Head with Retrieval-Based Head Motion},
  author = {anonymous},
  year   = {2026},
  note   = {Preliminary work, in progress}
}
```

## Acknowledgement

This project is built on top of [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian)
and re-uses parts of [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting),
a modified [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization),
and [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn).
Data utilities draw from [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF),
[ER-NeRF](https://github.com/Fictionarry/ER-NeRF),
[AD-NeRF](https://github.com/YudongGuo/AD-NeRF), and
[GeneFace](https://github.com/yerfor/GeneFace).
Teeth and lip masks use [EasyPortrait](https://github.com/hukenovs/easyportrait),
AU extraction uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), and
perceptual losses use [DINOv2](https://github.com/facebookresearch/dinov2) and
[facenet-pytorch](https://github.com/timesler/facenet-pytorch). Thanks to all authors.

## License

For research use only. See [LICENSE.md](LICENSE.md).
