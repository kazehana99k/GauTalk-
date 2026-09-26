# GauTalk: 本人らしい頭部運動・表情を持つ人物特化 3D Gaussian トーキングヘッド

本リポジトリは研究開発中のコードを公開するものです。API・学習スクリプト・checkpoint 名・環境変数は
予告なく変更されることがあります。描画スタックの設計は [DESIGN.md](DESIGN.md) に記載しています。

GauTalk は、本人の映像から学習した人物特化のトーキングヘッド合成パイプラインです。
被写体ごとの動画（数分）から学習し、新規音声に対してフォトリアルな顔動画を描画します。

本手法の中心は、頭部運動と表情を「生成」せず「本人の実測ブロックを選んで連結する」という設計です。
同じ音声に対して自然な頭の動きは複数あり得るため、誤差最小化で回帰すると平均に潰れて
ほとんど動かなくなります。GauTalk は代わりに、本人が実際に行った動きのブロック（実ブロック）を
種類ごとに集めておき、汎用 Transformer が予測した種類の実ブロックを選んでつなぎます。

英語版は [README_en.md](README_en.md) を参照してください。

---

## System Overview

GauTalk は、頭部運動と表情を「生成」ではなく「本人の実測ブロックの選択・連結」で作ります。

1. 多人物の動画から頭部運動と表情の動きを短いブロックに切り、種類に分けます。そのうえで、
   直前までの動きと音声から次に来る動きの種類を予測する汎用 Transformer を学習します。
2. 対象人物の動画にも同じ処理を行い、種類ごとに本人の実ブロックを集めた特徴セットを作り、
   汎用 Transformer を対象人物のデータで微調整します。
3. 生成時は、予測された種類の実ブロックを本人の特徴セットから選んでつなぎ、得られた
   頭部運動・表情と音声を 3D Gaussian 描画モデルに入力して顔動画を描画します。

表情は眉と目に関わる 6 つの AU（AU01・04・05・06・07・45）を頭部運動と同じ枠組みで作ります。
まばたきは区切りには使わず、ブロックの中に残したまま再生されます。口の形は描画モデル内で
音声から直接駆動される別経路です。手法の詳細は論文で公開します。本リポジトリに含まれるのは
描画モデルとデータ前処理で、上記 1〜3 のプランナは未公開です（Repository Scope を参照）。

---

## Repository Scope

現時点で本リポジトリに含まれるのは描画スタックとデータ前処理です。

| 構成要素 | 状態 |
| --- | --- |
| 3D Gaussian 描画モデル（face / mouth / fuse ステージ）| 含む |
| 描画モデル用のデータ前処理（3DMM トラッキング・parsing・マスク・音声特徴）| 含む |
| 描画モデルの学習・推論スクリプト（学習動画の頭部姿勢・AU で描画）| 含む |
| 頭部姿勢・表情の抽出（EMICA による FLAME 6DoF、OpenFace による AU）とブロック分割・種類分け | 未公開（[`planner/`](planner/) にインタフェースの雛形のみ） |
| 汎用 Transformer の学習コード・重み、対象人物での微調整、特徴セットの構築 | 未公開（同上） |
| 実ブロックの選択・連結と、生成した頭部運動・表情を描画モデルへ入力する経路 | 未公開（同上） |
| 評価・計測基盤 | 未公開（論文公開時に追加予定） |

表中の 3DMM トラッキングは描画モデルの学習に使う頭部姿勢（カメラ姿勢）を推定するものです。
論文に記載した頭部姿勢の抽出（FLAME を用いる EMICA）はプランナ側の前処理で、プランナと
合わせて公開予定です。`planner/` には各段階の入出力を示す関数の雛形だけを置いており、
実装・重み・設定値は含みません。

---

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

`environment.yml` に含まれていない追加依存があります（既定のレシピで必要になります）：

```bash
pip install transformers librosa   # data_utils/hubert.py
pip install timm                   # DINOv2 知覚損失 (--dino_w)
pip install scikit-image           # scripts/eval_v17_full.py
pip install facenet-pytorch        # ArcFace 損失 (--arc_w、既定レシピでは必須)
```

DINOv2 の重み — `--dino_w > 0` を使う場合は `dinov2_vits14_pretrain.pth` を用意し、
環境変数 `DINOV2_CKPT` にそのパスを設定してください（未設定・不在の場合は明示的にエラー終了します）。

---

## Usage

### Important Notice

学習用動画は単一人物・正面向き・背景静止を前提としています。25fps・512×512 に揃えた
`data/<ID>/<ID>.mp4` を用意してください。

### Video Dataset

`data/<ID>/` 以下に動画と派生物を置きます。前処理を通すと `transforms_train.json`、
`ori_imgs/`、`parsing/`、`torso_imgs/`、`gt_imgs/`、`au.csv`、音声特徴 `.npy` が生成されます。

### Pre-processing Training Video

```bash
# 1. 動画前処理 (transforms.json / parsing / landmarks / 背景画像など)
python data_utils/process.py data/<ID>/<ID>.mp4

# 2. OpenFace で AU を抽出し data/<ID>/au.csv に保存
#    AU25_r が含まれていることを必ず確認

# 3. 歯マスク
python data_utils/easyportrait/create_teeth_mask.py data/<ID>

# 4. lip / cavity 2D マスク (TG_LIP_CAVITY=1 を使う場合に必須)
python data_utils/easyportrait/create_lip_cavity_mask.py data/<ID>
```

### Audio Pre-process

```bash
# DeepSpeech
python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>

# HuBERT (--audio_extractor hubert で指定)
python data_utils/hubert.py --wav data/<ID>/aud.wav
```

### Train

学習は face → mouth → fuse の 3 段です（fuse は dual-head mouth 構成）。

**前提条件.** fuse ステージは事前学習済みの fuse checkpoint を face 側の土台として
必要とします。これは本リポジトリの手順からは生成できない外部資産です。パスは環境変数
`FACE_PRIOR_CKPT` で渡し、`scripts/train_v30e.sh` は未設定・不在の場合に明示的にエラー終了します。
必ず同一被写体の checkpoint を使ってください（別被写体のものを流用すると補助ヘッドが汚染されます）。

```bash
dataset=data/<ID>
work=output/<ID>_v30e
export TG_LIP_CAVITY=1

# A. Face
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 25000
#    →  $work/chkpnt_face_v30_latest.pth

# B. Mouth — 必要なら以下を有効化
#    z prune:      export TG_MOUTH_Z_MAX=0.05
#    異方性正則化: export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000
#    →  $work/chkpnt_mouth_v30_latest.pth

# C. Fuse 初期化（事前学習済み fuse checkpoint + mouth ckpt から dual-head init を構築）
python scripts/build_fuse_v30e_init.py \
  --v17_fuse $FACE_PRIOR_CKPT \
  --mouth_ckpt $work/chkpnt_mouth_v30_latest.pth \
  --out $work/chkpnt_fuse_v30e_init.pth

# D. Fuse 学習
python train_fuse_v30e.py -s $dataset -m $work \
  --init_ckpt $work/chkpnt_fuse_v30e_init.pth \
  --opacity_lr 0.001 --audio_extractor hubert --total_iters 5000 \
  --au_window_T 8 --aperture_w 0.2 --detail_w 0.5 --feat_anchor_w 0.005 \
  --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0
#    →  $work/chkpnt_fuse_v30e_latest.pth
```

C〜D は [`scripts/train_v30e.sh`](scripts/train_v30e.sh) でまとめて実行できます
（`FACE_PRIOR_CKPT=<path> bash scripts/train_v30e.sh $dataset $work <gpu_id> [fuse_iters]`）。
各ステージの損失重み・環境変数は [DESIGN.md](DESIGN.md) を参照してください。

### Test

```bash
python synthesize_fuse_v30e.py -s data/<ID> -m output/<ID>_v30e \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30e_latest.pth \
  --output_dir output/<ID>_v30e/render_v30e_full --max_frames 9999 --au_window_T 8

python scripts/eval_v17_full.py output/<ID>_v30e/render_v30e_full/seq_test
```

v30e の重みは `synthesize_fuse_v30e.py` で描画してください。旧 `synthesize_fuse_v18.py`
には cavity head の分岐がないため、dual-head で学習した重みを読ませると cavity 駆動が失われます。

### 頭部運動・表情の駆動について

頭部運動・表情のプランナ（ブロック分割、種類分け、汎用 Transformer とその微調整、実ブロックの
選択・連結）は現時点で本リポジトリの外にあり、論文公開に合わせてリリースします。上記の推論
スクリプトは評価スプリットを描画するもので、頭部姿勢と AU は学習動画のものを使います。
任意音声から頭部運動・表情まで含めて駆動する経路は、現状このリポジトリでは公開していません。
各段階のインタフェースは [`planner/README.md`](planner/README.md) を参照してください。

---

## Known Issues

試行錯誤期に増えたスクリプトが整理しきれておらず、以下は現状動作しません。順次修正します。

- `scripts/train_v30.sh` — リポジトリに含まれない fuse 学習スクリプトを呼ぶため途中で停止します。
  [`scripts/train_v30e.sh`](scripts/train_v30e.sh) を使ってください。
- `data_utils/extract_au_openface.py` — 空ファイルです。AU 抽出は OpenFace の
  `FeatureExtractor` を直接実行し、`data/<ID>/au.csv` に保存してください。
- `scripts/build_lip_mask_3d.py` — リポジトリに含まれないモジュールを import するため実行できません
  （lip 3D マスクの手順は現状省略可能です）。
- 事前学習済み checkpoint のパスに既定値はありません。`--init_ckpt`（fuse 学習）、`--v17_fuse`
  （fuse 初期化）、`FACE_PRIOR_CKPT`（`scripts/train_v30e.sh`）は必須指定、DINOv2 重みは環境変数
  `DINOV2_CKPT` で指定してください。

## Results

定量評価は論文公開まで本リポジトリでは掲載しません。
評価プロトコル・比較手法・数値は、論文の公開に合わせて掲載します。

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
