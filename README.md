# GauTalk: 検索再生型の頭部動作を持つ人物特化 3D Gaussian トーキングヘッド

> 🚧 **Work in Progress** — このリポジトリは現在進行中の研究プロジェクトです。
> API・学習スクリプト・ckpt 名・環境変数は予告なく変わる可能性があります。
> 描画スタックの詳細設計は [DESIGN.md](DESIGN.md) を参照してください。

[Paper (TBA)] | [Project (TBA)] | [Video (TBA)]

![teaser](assets/main.png)

> 🎞️ Demo video / qualitative comparisons — Coming soon.

GauTalk は、**本人の映像から学習した人物特化のトーキングヘッド合成**パイプラインです。
被写体ごとの動画（数分）から学習し、新規音声に対してフォトリアルな顔動画を描画します。

本手法の中心は、**頭部動作を「生成」せず「選んで再生する」**という設計です。
同じ音声に対して自然な頭の動きは複数あり得るため、誤差最小化で回帰すると平均に潰れて
ほとんど動かなくなります。GauTalk は代わりに、本人の実測動作を断片（ブロック）として
記憶しておき、音声に応じてその中から選び、**学習動画の実フレームをそのまま再生**します。

> 英語版 README は [`README_en.md`](README_en.md) を参照してください。

---

## System Overview

駆動チャンネルは 3 つに分かれています。音声は「いつ切り替えるか」と「感情の方向」を担い、
「どう動くか」の中身は本人の記憶側が担います。

```
                  ┌─ オフライン（学習時に一度だけ）────────────────┐
  本人動画 ───────→│ ブロック分割 →  動作メモリ（ブロック＋接続グラフ）│
                  └──────────────────────────────────────────────┘
                                      │
   音声 ─┬─→ 切替ゲート ──→ 切替時刻 ──┤
         │                             ↓
         │                     候補生成 →  動作選択
         │                             ↓
         │           実フレーム番号列（本人動画の実測値そのまま）
         │                             ↓
         │        頭部 6DoF ＋ カメラ行列・胴体画像（同一フレームに固定）
         │
         ├─→ 感情推定 ──→ 表情合成 ──→ 上半顔 6 AU
         │
         └─→ 口形（描画モデル内で音声から直接駆動）
                                      ↓
                  学習済み 3D Gaussian 描画モデル  →  顔動画
```

- **頭部動作（検索・再生）** — 本人動画を短い断片に分割して動作メモリを構築し、終端姿勢と
  始端姿勢が近い組だけを接続可能とするグラフを作ります。推論時は合法な候補の中から選び、
  出力は**動作パラメータではなく学習動画の実フレーム番号列**です。カメラ行列と胴体画像も
  そのフレームのものを固定で使うため、動きは常に物理的に成立し、本人らしさが保たれます。
- **上半顔の表情** — 上半顔 6 AU のみを合成します（土台＋実測イベント＋音声由来の感情方向
  ＋まばたき、本人の可動域にクリップ）。**口の形は含まれません**。
- **口形** — 描画モデル内で音声から直接駆動される別経路です（既存経路、変更なし）。
- **描画** — TalkingGaussian ベースの 3D Gaussian Splatting 描画モデル。

---

## Repository Scope

現時点で本リポジトリに含まれるのは**描画スタックとデータ前処理**です。

| | 状態 |
| --- | --- |
| 3D Gaussian 描画モデル（face / mouth / fuse ステージ）| ✅ 本リポジトリに含む |
| データ前処理（3DMM トラッキング・parsing・マスク・音声特徴）| ✅ 本リポジトリに含む |
| 学習・推論スクリプト | ✅ 本リポジトリに含む |
| 頭部動作プランナ（ブロック分割・動作メモリ・切替ゲート・動作選択）| 🔜 論文公開に合わせて順次リリース |
| 評価・計測基盤 | 🔜 同上 |

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

**DINOv2 の重み** — `models/perceptual_losses.py` の既定パスが作者環境の絶対パスに
ハードコードされています。`--dino_w > 0` を使う場合は
`dinov2_vits14_pretrain.pth` を用意し、同ファイルのパスを自環境に合わせて書き換えてください。

---

## Usage

### Important Notice

学習用動画は**単一人物・正面向き・背景静止**を前提としています。25fps・512×512 に揃えた
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

現行のメインラインは **v30e**（dual-head mouth fuse）です。

> ⚠️ **前提条件** — fuse ステージは事前学習済みの **V17 fuse checkpoint**
> (`chkpnt_fuse_v17_latest.pth`) を prior として必要とします。これは本リポジトリの
> 手順からは生成できない外部資産です。`scripts/train_v30e.sh` は見つからない場合に
> 明示的にエラー終了します。

```bash
dataset=data/<ID>
work=output/<ID>_v30e
export TG_LIP_CAVITY=1

# A. Face
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 25000
#    →  $work/chkpnt_face_v30_latest.pth

# B. Mouth — 必要なら Plan F / G を有効化
#    Plan F: export TG_MOUTH_Z_MAX=0.05
#    Plan G: export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000
#    →  $work/chkpnt_mouth_v30_latest.pth

# C. Fuse 初期化（V17 prior + mouth ckpt から dual-head init を構築）
python scripts/build_fuse_v30e_init.py \
  --v17_fuse <事前学習済み V17 fuse の .pth> \
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
（`bash scripts/train_v30e.sh $dataset $work <gpu_id> [fuse_iters]`）。
各ステージの損失重み・環境変数は [DESIGN.md](DESIGN.md) を参照してください。

### Test

```bash
python synthesize_fuse_v30e.py -s data/<ID> -m output/<ID>_v30e \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30e_latest.pth \
  --output_dir output/<ID>_v30e/render_v30e_full --max_frames 9999 --au_window_T 8

python scripts/eval_v17_full.py output/<ID>_v30e/render_v30e_full/seq_test
```

> v30e の重みは **`synthesize_fuse_v30e.py`** で描画してください。旧 `synthesize_fuse_v18.py`
> には cavity head の分岐がないため、dual-head で学習した重みを読ませると cavity 駆動が失われます。

### 頭部動作の駆動について

頭部動作の検索・再生（動作メモリからのブロック選択と実フレーム再生）は現時点で
**本リポジトリの外**にあり、論文公開に合わせてリリースします。上記の推論スクリプトは
評価スプリットを描画するもので、頭部姿勢は学習動画のものを使います。
任意音声から頭部動作まで含めて駆動する経路は、現状このリポジトリでは公開していません。

---

## Known Issues

試行錯誤期に増えたスクリプトが整理しきれておらず、以下は現状動作しません。順次修正します。

- `scripts/train_v30.sh`（旧 v30 系）— 存在しない `train_fuse_v28.py` を呼ぶため途中で停止します。
  **v30e 系の [`scripts/train_v30e.sh`](scripts/train_v30e.sh) を使ってください。**
- `data_utils/extract_au_openface.py` — 空ファイルです。AU 抽出は OpenFace の
  `FeatureExtractor` を直接実行し、`data/<ID>/au.csv` に保存してください。
- `scripts/build_lip_mask_3d.py` — リポジトリに含まれない `train_au_editor.py` を import
  するため実行できません（lip 3D マスクの手順は現状省略可能です）。
- `models/perceptual_losses.py` の DINOv2 重みパス、`train_fuse_v30e.py` の `--init_ckpt` 既定値、
  `scripts/train_v30e.sh` の V17 prior パスが作者環境の絶対パスにハードコードされています。

## Results

定量評価は**論文投稿中のため本リポジトリでは公開していません**。
評価プロトコル・比較手法・数値は、論文の公開に合わせて掲載します。

（以前のバージョンに掲載していた描画ステージのみの暫定値は、対象範囲が変わったため取り下げました。）

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
