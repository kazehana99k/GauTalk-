# GauTalk: Person-Specific Head Motion and Facial Expressions for 3D Gaussian Talking Heads

This repository contains research code under active development. APIs, training scripts,
checkpoint names, and environment variables may change without notice. The design of the
rendering stack is described in [DESIGN.md](DESIGN.md).

GauTalk is a person-specific talking-head synthesis pipeline learned from a subject's own
video. It is trained from a per-subject recording (a few minutes) and renders a photorealistic
talking face for novel audio.

The central design choice is that head motion and expression are not generated — the subject's
own measured blocks are selected and concatenated. For the same audio there are many plausible
head trajectories, so regressing one by minimizing error collapses toward the average and the
head barely moves. Instead, GauTalk collects the blocks of motion the subject actually performed
(real blocks) by type, and concatenates real blocks of the types predicted by a generic Transformer.

This is not the official TalkingGaussian repository. GauTalk is built on top of
[TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian); see the
Acknowledgement section for full attribution. A Japanese version of this README is available at
[`README.md`](README.md).

---

## System Overview

GauTalk does not generate head motion and expression; it selects and concatenates the
subject's own measured motion blocks.

1. Videos of many speakers are cut into short blocks of head motion and facial expression,
   and the blocks are grouped into types. A generic Transformer is then trained to predict
   the type of the next block from the preceding blocks and the audio.
2. The same processing is applied to the target subject's video to build a feature set,
   i.e. the subject's real blocks grouped by type, and the generic Transformer is fine-tuned
   on the subject's data.
3. At generation time, a real block of the predicted type is selected from the subject's
   feature set and appended; the resulting head motion and expression are fed, together
   with the audio, to the 3D Gaussian renderer to produce the face video.

Expression covers the six AUs around the brows and eyes (AU01, 04, 05, 06, 07, 45) and is
produced in the same framework as head motion. Blinks are not used for segmentation; they stay
inside the blocks and are replayed as is. Mouth shape is a separate path driven directly from
audio inside the renderer. Details of the method will be published with the paper. This
repository contains the renderer and data preprocessing; the planner in steps 1–3 is not yet
released (see Repository Scope).

---

## Repository Scope

At this point the repository contains the rendering stack and data preprocessing.

| Component | Status |
| --- | --- |
| 3D Gaussian renderer (face / mouth / fuse stages) | Included |
| Data preprocessing for the renderer (3DMM tracking, parsing, masks, audio features) | Included |
| Renderer training and inference scripts (rendered with the training video's head poses and AUs) | Included |
| Head-pose and expression extraction (FLAME 6-DoF via EMICA, AUs via OpenFace), block segmentation and typing | Not yet released (interface stubs only, under [`planner/`](planner/)) |
| Generic Transformer training code and weights, per-subject fine-tuning, feature-set construction | Not yet released (same as above) |
| Real-block selection and concatenation, and the path feeding generated head motion and expression into the renderer | Not yet released (same as above) |
| Evaluation and measurement harness | Not yet released (planned with the paper) |

The 3DMM tracking listed above estimates the head/camera poses used to train the renderer. The
head-pose extraction described in the paper (EMICA, based on FLAME) belongs to the planner's
preprocessing and will be released together with the planner. `planner/` holds only function
stubs that document the inputs and outputs of each stage; it contains no implementation,
weights, or configuration values.

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

A few dependencies are not in `environment.yml` but are needed by the default recipe:

```bash
pip install transformers librosa   # data_utils/hubert.py
pip install timm                   # DINOv2 perceptual loss (--dino_w)
pip install scikit-image           # scripts/eval_v17_full.py
pip install facenet-pytorch        # ArcFace loss (--arc_w; required by the default recipe)
```

DINOv2 weights — if you use `--dino_w > 0`, obtain `dinov2_vits14_pretrain.pth` and point the
`DINOV2_CKPT` environment variable at it (the loss raises an explicit error if it is unset or missing).

---

## Usage

### Important Notice

Training videos are assumed to show a single, front-facing subject against a static background.
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

Training has three stages: face → mouth → fuse (the fuse stage uses a dual-head mouth).

**Prerequisite.** The fuse stage requires a pre-trained fuse checkpoint as the face
foundation. This is an external asset that cannot be produced from the steps in this repository.
Pass its path via the `FACE_PRIOR_CKPT` environment variable; `scripts/train_v30e.sh` exits with
an explicit error if it is unset or missing. It must come from the same subject — reusing
another subject's checkpoint contaminates the auxiliary heads.

```bash
dataset=data/<ID>
work=output/<ID>_v30e
export TG_LIP_CAVITY=1

# A. Face
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 25000
#    →  $work/chkpnt_face_v30_latest.pth

# B. Mouth — optional regularizers
#    z prune:                 export TG_MOUTH_Z_MAX=0.05
#    anisotropy regularizer:  export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000
#    →  $work/chkpnt_mouth_v30_latest.pth

# C. Fuse init (build the dual-head init from the pre-trained fuse checkpoint + mouth ckpt)
python scripts/build_fuse_v30e_init.py \
  --v17_fuse $FACE_PRIOR_CKPT \
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
(`FACE_PRIOR_CKPT=<path> bash scripts/train_v30e.sh $dataset $work <gpu_id> [fuse_iters]`).
Per-stage loss weights and environment variables are documented in [DESIGN.md](DESIGN.md).

### Test

```bash
python synthesize_fuse_v30e.py -s data/<ID> -m output/<ID>_v30e \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30e_latest.pth \
  --output_dir output/<ID>_v30e/render_v30e_full --max_frames 9999 --au_window_T 8

python scripts/eval_v17_full.py output/<ID>_v30e/render_v30e_full/seq_test
```

Render v30e weights with `synthesize_fuse_v30e.py`. The older `synthesize_fuse_v18.py`
has no cavity-head branch, so loading dual-head weights into it silently drops cavity driving.

### On head-motion and expression driving

The planner (block segmentation, typing, the generic Transformer and its fine-tuning, and
real-block selection and concatenation) currently lives outside this repository and will be
released alongside the paper. The inference scripts above render the evaluation split, with head
poses and AUs taken from the training video. A path that drives head motion and expression from
arbitrary audio is not published here yet. The per-stage interfaces are documented in
[`planner/README.md`](planner/README.md).

---

## Known Issues

Scripts accumulated during exploratory work have not been fully cleaned up. The following do not
currently work and are being fixed:

- `scripts/train_v30.sh` calls a fuse-training script that is not in the
  repository and stops partway through. Use [`scripts/train_v30e.sh`](scripts/train_v30e.sh) instead.
- `data_utils/extract_au_openface.py` is an empty file. Run OpenFace's `FeatureExtractor`
  directly and save the result to `data/<ID>/au.csv`.
- `scripts/build_lip_mask_3d.py` imports a module that is not in the repository, so it cannot run
  (the lip 3D mask step can be skipped for now).
- Pre-trained checkpoint paths have no defaults: `--init_ckpt` (fuse training), `--v17_fuse`
  (fuse init) and `FACE_PRIOR_CKPT` (`scripts/train_v30e.sh`) are required, and the DINOv2 weights
  are configured via the `DINOV2_CKPT` environment variable.

## Results

Quantitative results are not listed in this repository until the paper is public.
The evaluation protocol, baselines, and numbers will be released together with the paper.

---

## Citation

```
@misc{gautalk2026,
  title  = {GauTalk++: Person-Specific Head Motion and Facial Expressions for 3D Gaussian Splatting-Based Talking Head Synthesis},
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
