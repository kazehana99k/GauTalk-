# GauTalk: AU25-Driven 3D Gaussian Talking Head Synthesis

> 🚧 **Work in Progress** — このリポジトリは現在進行中の研究プロジェクトです。
> API・学習スクリプト・ckpt 名・環境変数は予告なく変わる可能性があります。
> 詳細な設計は [DESIGN.md](DESIGN.md)、未解決の課題は [DESIGN.md §4](DESIGN.md#4-ロードマップ) を参照。

[Paper (TBA)] | [Project (TBA)] | [Video (TBA)]

![teaser](assets/main.png)

> 🎞️ Demo video / qualitative comparisons — Coming soon.

GauTalk は、音声と Action Unit から駆動される **3D Gaussian Splatting ベースの talking head 合成**
パイプラインです。被写体ごとの動画 (1〜5 分) と OpenFace の AU、HuBERT / DeepSpeech 音声特徴から
学習し、新規音声に対してフォトリアルなトーキングフェイスを描画します。
AU25 (顎の開き) の明示入力、per-Gaussian cross-attention、dual-head mouth driver、ArcFace + DINOv2
の知覚損失などを組み合わせ、長尺・任意音声でも口元が安定して動くように設計されています。

> 英語版 README は [`README_en.md`](README_en.md) を参照してください。

## Installation

Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1 で動作確認しています。

```bash
git clone https://github.com/kazehana99k/GauTalk-.git --recursive
cd GauTalk-

conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install facenet-pytorch    # ArcFace 損失用 (任意)
```

`diff-gaussian-rasterization` / `gridencoder` のビルドが失敗する場合は
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) と
[torch-ngp](https://github.com/ashawkey/torch-ngp) を参照してください。

### Preparation

```bash
# 3DMM + face_parsing
bash scripts/prepare.sh

# Basel Face Model 2009 (01_MorphableModel.mat) を data_utils/face_tracking/3DMM/ に置く
cd data_utils/face_tracking && python convert_BFM.py && cd ../..

# EasyPortrait (歯マスク用)
pip install -U openmim && mim install mmcv-full==1.7.1
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
  -O data_utils/easyportrait/fpn-fp-512.pth

# OpenFace (AU 抽出) — 公式手順 https://github.com/TadasBaltrusaitis/OpenFace
```

## Usage

### Important Notice

- This code is provided for research purposes only.
- 本コードを **悪意のある用途・違法な用途** に使用することを禁じます。
  使用にあたっては適用される法令を遵守し、なりすまし・誹謗中傷・名誉毀損などに
  繋がる利用は行わないでください。
- 本コード使用により生じたいかなる損害についても、著者は責任を負いません。

### Video Dataset

`data/<ID>/<ID>.mp4` に学習動画を置きます。**25 FPS、解像度 約 512×512、1〜5 分** で、
全フレームに話者が映っていることが前提です。

### Pre-processing Training Video

```bash
# 1. 動画前処理 (transforms.json / parsing / landmarks / 背景画像など)
python data_utils/process.py data/<ID>/<ID>.mp4

# 2. OpenFace で AU を抽出し data/<ID>/au.csv に保存
#    AU25_r が含まれていることを必ず確認
python data_utils/extract_au_openface.py --root data/<ID>

# 3. 歯マスク
export PYTHONPATH=./data_utils/easyportrait
python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>

# 4. lip / cavity 2D マスク (TG_LIP_CAVITY=1 を使う場合に必須)
python data_utils/easyportrait/create_lip_cavity_mask.py ./data/<ID>

# 5. lip 3D マスク (face stage 後に Gaussian 側の lip 投票を取得)
python scripts/build_lip_mask_3d.py --root data/<ID> --ckpt <face stage ckpt>
```

### Audio Pre-process

評価には DeepSpeech 特徴を使っていますが、HuBERT も利用可能 (英語以外推奨)。

```bash
# DeepSpeech
python data_utils/deepspeech_features/extract_ds_features.py --input data/<name>.wav

# HuBERT (--audio_extractor hubert で指定)
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

# B. Mouth (50k iter) — 必要なら Plan F / G を有効化
#    Plan F: export TG_MOUTH_Z_MAX=0.05
#    Plan G: export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000

# C. Fuse 初期化 + D. Fuse 学習 (10k iter)
python scripts/build_fuse_v30_init.py \
  --face_ckpt $work/chkpnt_face_v30_clean.pth \
  --mouth_ckpt $work/chkpnt_mouth_v30_latest.pth \
  --head_prior <事前学習済み V17 fuse の .pth> \
  --face_max_pts 50000 --out $work/chkpnt_fuse_v30_init.pth
python train_fuse_v30e.py -s $dataset -m $work \
  --init_ckpt $work/chkpnt_fuse_v30_init.pth \
  --opacity_lr 0.001 --audio_extractor hubert --total_iters 10000 \
  --au_window_T 8 --aperture_w 0.2 --detail_w 0.5 --feat_anchor_w 0.005 \
  --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0
```

[`scripts/train_v30.sh`](scripts/train_v30.sh) でワンショットでも実行可能です。

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

3 被験者・25fps 動画でのステージ F 評価値（チューニング途中の数値です）：

| 被験者 | 設定 | PSNR ↑ | LMD ↓ |
| ------ | ---- | ------ | ----- |
| macron | v30au25 (default) | **35.54** | **2.96** |
| obama  | v30au25 (default) | **35.02** | **3.65** |
| may    | v30au25 + `TG_MOUTH_Z_MAX=0.05` | 29.92 | 4.10 |

## Method Overview

GauTalk は次の要素で構成されています。詳細は [DESIGN.md](DESIGN.md)：

- **AU-aware MotionNetwork** — AU25 を加えた 7 次元 expression 入力、per-axis tanh cap、
  mouth 用 y-axis HashGrid bypass と AU25 加算分岐 (`au_mouth_branch`)、唇ランドマーク分岐。
- **Per-Gaussian Cross-Attention** — 各 Gaussian が 8 音声 + 8 AU トークンにアテンションを
  かけ、`d_xyz/d_rot/d_opa/d_scale` の小さな残差を出力。Mouth stage では lip head と
  cavity head の dual-head 構成。
- **Auxiliary heads** — PhonemeAuxHead（392 クラス音素予測）、PerGaussianAlbedoMLP
  （発音状態依存の per-Gaussian RGB 残差）、aperture aux head（AU25/26 回帰）。
- **Perceptual losses** — ArcFace identity + DINOv2 + Sobel detail + features_dc anchor。
- **Mask pipeline** — soft mouth mask (erode + dilate)、2D lip/cavity マスク、
  3D lip vote、`face_parsing_fine` の class 11+12+13 統合。
- **Stabilisers** — 異方性正則化、Z-prune、cavity depth prior、apex weight schedule。

## Follow-Up

- `train_mouth_v2` (mouth mask 拡張) を検証予定。
- 17-AU 個別可制御性の改善 — OpenFace 逆計測で評価しながら設計中。
- 多言語音声 (日本語 / 中国語) でのロバスト性を HuBERT 特徴で評価中。

## Citation

```
@misc{gautalk2026,
  title  = {GauTalk: AU25-Driven 3D Gaussian Talking Head Synthesis},
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

For research use only. See LICENSE.md.
