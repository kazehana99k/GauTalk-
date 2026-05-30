# GauTalk: AU25-Driven 3D Gaussian Talking Head Synthesis

> 🚧 **Work in Progress** — this is an active research project.
> APIs, training scripts, checkpoint names, and environment variables may change without notice.
> See [DESIGN.md](DESIGN.md) for the detailed design and [DESIGN.md §4](DESIGN.md#4-ロードマップ) for open problems.

[Paper (TBA)] | [Project (TBA)] | [Video (TBA)]

![teaser](assets/main.png)

> 🎞️ Demo video / qualitative comparisons — Coming soon.

GauTalk is a **3D Gaussian Splatting-based talking-head synthesis** pipeline driven by audio and
Action Units. It is trained from a per-subject video (1–5 min) together with OpenFace AUs and
HuBERT / DeepSpeech audio features, and renders a photorealistic talking face for novel audio.
It combines explicit AU25 (jaw-opening) input, per-Gaussian cross-attention, a dual-head mouth
driver, and ArcFace + DINOv2 perceptual losses so that the mouth stays stable over long, arbitrary
audio.

> **This is not the official TalkingGaussian repository.** GauTalk is built on top of
> [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian); see the
> Acknowledgement section for full attribution. A Japanese version of this README is available at
> [`README.md`](README.md).

## Installation

Tested on Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1.

```bash
git clone https://github.com/kazehana99k/GauTalk-.git --recursive
cd GauTalk-

conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install facenet-pytorch    # for the ArcFace loss (optional)
```

If the `diff-gaussian-rasterization` / `gridencoder` build fails, refer to
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[torch-ngp](https://github.com/ashawkey/torch-ngp).

### Preparation

```bash
# 3DMM + face_parsing
bash scripts/prepare.sh

# Place the Basel Face Model 2009 (01_MorphableModel.mat) in data_utils/face_tracking/3DMM/
cd data_utils/face_tracking && python convert_BFM.py && cd ../..

# EasyPortrait (for teeth masks)
pip install -U openmim && mim install mmcv-full==1.7.1
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
  -O data_utils/easyportrait/fpn-fp-512.pth

# OpenFace (AU extraction) — official instructions: https://github.com/TadasBaltrusaitis/OpenFace
```

## Usage

### Important Notice

- This code is provided for research purposes only.
- Using this code for **malicious or illegal purposes is prohibited.** Comply with all applicable
  laws, and do not use it for impersonation, harassment, defamation, or similar harm.
- The authors are not responsible for any damages arising from the use of this code.

### Video Dataset

Put the training video at `data/<ID>/<ID>.mp4`. It must be **25 FPS, around 512×512, 1–5 min**, with
the speaker visible in every frame.

### Pre-processing Training Video

```bash
# 1. Video pre-processing (transforms.json / parsing / landmarks / background image, ...)
python data_utils/process.py data/<ID>/<ID>.mp4

# 2. Extract AUs with OpenFace and save to data/<ID>/au.csv
#    Make sure AU25_r is included
python data_utils/extract_au_openface.py --root data/<ID>

# 3. Teeth mask
export PYTHONPATH=./data_utils/easyportrait
python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>

# 4. 2D lip / cavity mask (required when using TG_LIP_CAVITY=1)
python data_utils/easyportrait/create_lip_cavity_mask.py ./data/<ID>

# 5. 3D lip mask (obtain per-Gaussian lip votes after the face stage)
python scripts/build_lip_mask_3d.py --root data/<ID> --ckpt <face stage ckpt>
```

### Audio Pre-process

DeepSpeech features are used for evaluation; HuBERT is also available (recommended for non-English).

```bash
# DeepSpeech
python data_utils/deepspeech_features/extract_ds_features.py --input data/<name>.wav

# HuBERT (select with --audio_extractor hubert)
python data_utils/hubert.py --wav data/<name>.wav
```

### Train

```bash
dataset=data/<ID>
work=output/<ID>_v30au25
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
export TG_LIP_CAVITY=1

# A. Face (25k iter)
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert \
  --init_num 2000 --densify_grad_threshold 0.0015 --iterations 25000
cp $work/chkpnt_face_v30_latest.pth $work/chkpnt_face_v30_clean.pth

# B. Mouth (50k iter) — enable Plan F / G if needed
#    Plan F: export TG_MOUTH_Z_MAX=0.05
#    Plan G: export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000

# C. Fuse init + D. Fuse training (10k iter)
python scripts/build_fuse_v30_init.py \
  --face_ckpt $work/chkpnt_face_v30_clean.pth \
  --mouth_ckpt $work/chkpnt_mouth_v30_latest.pth \
  --head_prior <pretrained V17 fuse .pth> \
  --face_max_pts 50000 --out $work/chkpnt_fuse_v30_init.pth
python train_fuse_v30e.py -s $dataset -m $work \
  --init_ckpt $work/chkpnt_fuse_v30_init.pth \
  --opacity_lr 0.001 --audio_extractor hubert --total_iters 10000 \
  --au_window_T 8 --aperture_w 0.2 --detail_w 0.5 --feat_anchor_w 0.005 \
  --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0
```

[`scripts/train_v30.sh`](scripts/train_v30.sh) runs the whole thing in one shot.

### Test

```bash
python synthesize_fuse_v18.py -s data/<ID> -m output/<ID>_v30au25 \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30_latest.pth \
  --output_dir output/<ID>_v30au25/render_v30_full --max_frames 9999 --au_window_T 8
```

### Inference with Specified Audio

```bash
python data_utils/hubert.py --wav new_audio.wav
python synthesize_fuse_v18.py -s data/<ID> -m output/<ID>_v30au25 \
  --use_train --audio new_audio_hu.npy \
  --ckpt_name chkpnt_fuse_v30_latest.pth
```

## Results (preliminary)

Stage-F evaluation on 3 subjects with 25 fps video (numbers are mid-tuning):

| Subject | Setting | PSNR ↑ | LMD ↓ |
| ------- | ------- | ------ | ----- |
| macron  | v30au25 (default) | **35.54** | **2.96** |
| obama   | v30au25 (default) | **35.02** | **3.65** |
| may     | v30au25 + `TG_MOUTH_Z_MAX=0.05` | 29.92 | 4.10 |

## Method Overview

GauTalk consists of the following components. See [DESIGN.md](DESIGN.md) for details:

- **AU-aware MotionNetwork** — 7-dim expression input including AU25, per-axis tanh cap,
  a y-axis HashGrid bypass and an AU25 additive branch (`au_mouth_branch`) for the mouth, and a
  lip-landmark branch.
- **Per-Gaussian Cross-Attention** — each Gaussian attends to 8 audio + 8 AU tokens and outputs
  small `d_xyz/d_rot/d_opa/d_scale` residuals. The mouth stage uses a dual-head (lip head + cavity
  head) configuration.
- **Auxiliary heads** — PhonemeAuxHead (392-class phoneme prediction), PerGaussianAlbedoMLP
  (articulation-dependent per-Gaussian RGB residual), and an aperture aux head (AU25/26 regression).
- **Perceptual losses** — ArcFace identity + DINOv2 + Sobel detail + features_dc anchor.
- **Mask pipeline** — soft mouth mask (erode + dilate), 2D lip/cavity masks, 3D lip vote, and a
  merged `face_parsing_fine` class 11+12+13.
- **Stabilisers** — anisotropy regularization, Z-prune, cavity depth prior, apex weight schedule.

## Follow-Up

- Validating `train_mouth_v2` (mouth-mask expansion).
- Improving individual controllability of 17 AUs — designed while evaluating with OpenFace
  back-measurement.
- Evaluating robustness on multilingual audio (Japanese / Chinese) with HuBERT features.

## Citation

```
@misc{gautalk2026,
  title  = {GauTalk: AU25-Driven 3D Gaussian Talking Head Synthesis},
  author = {anonymous},
  year   = {2026},
  note   = {Preliminary work, in progress}
}
```

If you use this project, please also cite the base work it is built on:

```
@inproceedings{li2024talkinggaussian,
  title={TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting},
  author={Li, Jiahe and Zhang, Jiawei and Bai, Xiao and Zheng, Jin and Ning, Xin and Zhou, Jun and Gu, Lin},
  booktitle={European Conference on Computer Vision},
  pages={127--145},
  year={2024},
  organization={Springer}
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
