# GauTalk: AU25 駆動 + Cross-Attention 3D Gaussian Talking Head

> ⚠️ **本リポジトリは現在進行中の研究プロジェクトです (Work In Progress)。**
> API・学習スクリプト・チェックポイント名・環境変数は予告なく変更される可能性があります。
> 結果も最終形ではありません — 既知の問題と次の TODO は [§9](#9-未解決の課題--ロードマップ) を参照してください。

---

## 0. これは何か

GauTalk は、音声と Action Unit から駆動される **3D Gaussian Splatting ベースの talking head 合成**
パイプラインです。被写体ごとの動画 (1〜5 分) と OpenFace の AU、HuBERT / DeepSpeech 音声特徴から
学習し、新規音声に対してフォトリアルなトーキングフェイスを描画します。

主要な構成要素：

- **AU25 (顎の開き) を MotionNetwork の明示入力に組み込む** ことで、未知音声でも口内が安定して開く。
- **Per-Gaussian Cross-Attention** が speech ⇄ AU ⇄ Gaussian の関係を動的に学習する。
- **Mouth は dual-head Cross-Attention** で唇 (lip) と口内 (cavity) を分離して駆動する。
- **音素・アパーチャ・per-Gaussian アルベド** の補助ヘッドが、識別可能な内部表現を後押しする。
- **ArcFace + DINOv2 知覚損失** で identity drift と高周波ディテールを同時に保つ。
- **Soft mouth mask / 2D lip+cavity mask / 3D lip vote** で唇と口内の境界 seam を抑える。
- **Per-axis tanh cap / y-axis HashGrid bypass / 異方性正則化 / Z-prune** で長尺音声でも口元
  Gaussian を安定化する。

オリジナル英語版の参考用 README は [`README_en.md`](README_en.md) に残しています。

---

## 1. アーキテクチャ全体図

```
                ┌──────────────────────────────────────────────┐
                │           Audio (HuBERT / DeepSpeech)        │
                └────────────────────┬─────────────────────────┘
                                     │
       ┌──────────────────┐          │           ┌──────────────────────┐
       │ AU17 (OpenFace)  │──────────┤           │ Lip Landmark (Q1)    │
       │  + AU25 jaw drop │          │           └────────┬─────────────┘
       └──────────────────┘          │                    │
                │                    ▼                    ▼
                │       ┌────────────────────────────────────────┐
                │       │       MotionNetwork (Face / Mouth)     │
                │       │  - exp_in_dim = 7 (AU01..07 + AU25)    │
                │       │  - eye_dim    = 7 (AU45, AU25 raw)     │
                │       │  - per-axis tanh cap on d_xyz/d_scale  │
                │       │  - mouth: y-axis HashGrid bypass       │
                │       │  - mouth: au_mouth_branch (AU25 add)   │
                │       │  - mouth: lmk_proj+lmk_to_dxyz         │
                │       └─────────────────┬──────────────────────┘
                │                         │
                │           ┌─────────────┴───────────────┐
                │           ▼                             ▼
       ┌────────┴──────────────────┐         ┌──────────────────────────┐
       │ GaussianCrossAttnDriver   │         │   Auxiliary heads        │
       │  - per-Gaussian Q         │         │  - PhonemeAuxHead        │
       │  - 8 audio + 8 AU tokens  │         │  - PerGaussianAlbedoMLP  │
       │  - 4 heads, d_model=128   │         │  - aperture head         │
       │  - residual_scale 1e-3    │         │    (predict AU25/26)     │
       │  - Mouth dual-head:       │         └──────────────────────────┘
       │      lip head + cavity head
       └────────────┬──────────────┘
                    │
                    ▼
       ┌───────────────────────────────────────────┐
       │   3D Gaussian Splatting                   │
       │  - override_xyz / opacity / rotation      │
       │  - alpha-pass reconstruction              │
       └───────────────────────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────────────────────┐
       │ Losses                                    │
       │  L1 + SSIM + LPIPS                        │
       │  + ArcFace identity                       │
       │  + DINOv2 perceptual                      │
       │  + Sobel detail + features_dc anchor      │
       │  + cavity depth + apex weight schedule    │
       │  + R2 lip-y landmark constraint           │
       │  + temporal d_xyz smoothness (gated)      │
       └───────────────────────────────────────────┘
```

---

## 2. コンポーネント詳細

### 2.1 MotionNetwork ([`scene/motion_net.py`](scene/motion_net.py))

`MotionNetwork` は HashGrid 空間特徴と音声・AU から、各 Gaussian の差分姿勢
(`d_xyz, d_rot, d_opa, d_scale`) を回帰する MLP です。本実装の構成は：

- 表情入力 `au_exp` は **7 次元**で、`[AU01, AU04, AU05, AU06, AU07, AU45, AU25]` を渡す。
- `eye_dim = 7` で、AU45 と AU25 は **末尾の 2 スロットに raw のまま** 保持する
  (HashGrid 駆動の閉眼 / 開口を独立に保つため)。
- **Face MotionNetwork** は `d_xyz` に per-axis tanh cap (`[0.015, 0.06, 0.025]`) と
  `d_scale` の tanh cap (`0.5`、`exp` 比に直すと約 `[0.6, 1.7]`) を適用する。
- **Mouth MotionNetwork** には以下の分岐を持たせる：
  - **`y_bypass_proj`** — `(audio_emb, y_rel)` から d_y を直接生成して加算する小さな MLP。
    HashGrid の空間平滑がもたらす「上下唇が一緒に動いてしまう」問題を回避する。
  - **`au_mouth_branch`** — zero-init MLP で `(xyz_feat, AU25)` → 3D d_xyz の加算残差を出す。
    AU25 が音声経由ではなく直接 mouth Gaussian に伝わる経路を確保する。
  - **`lmk_proj` + `lmk_to_dxyz`** — 任意で渡せる 20 点の唇ランドマーク (40 次元) から
    32 次元の埋め込みを作り、Gaussian 特徴に concat して d_xyz の加算残差を生成する。
  - 出力には Face と同じ per-axis tanh cap (`[0.025, 0.15, 0.04]`) を最後に適用する。
- 推論時に `TG_AU45_EYE_GAIN`, `TG_AU25_LIP_GAIN`, `TG_MOUTH_D_XYZ_GAIN`,
  `TG_MOUTH_XATTN_GAIN` で運動振幅を後付けでスケールできる（学習に手を入れず A/B 評価する用途）。

### 2.2 Per-Gaussian Cross-Attention ([`models/cross_attn_driver.py`](models/cross_attn_driver.py))

`GaussianCrossAttnDriver` は各 Gaussian を独立のクエリとして、音声トークン列と AU トークン列に
アテンションをかけるモジュールです。

- 入力は **8 個の音声トークン + 8 個の AU トークン = 16 トークン**。
- 4 heads / `d_model = 128` / positional embedding でモダリティを識別。
- AU は単フレーム入力 (17-D を 8 トークンに分割) と **時間窓入力 (`au_window_T=8`、過去 4 + 未来 4 フレーム)** の両方をサポート。
- 出力は `(d_xyz, d_rot, d_opa, d_scale)` の **小さな残差**で、`residual_scale = 1e-3`。MotionNetwork の出力に**加算**して使う。
- **Face stage** では 1 個の driver を使う。
- **Mouth stage** では `cavity_idx` で Gaussian を 2 グループに分け、**lip head と cavity head の 2 個**を使う。
  `cavity_idx` は densify/prune ごとに再計算され、dual-head の整合性が保たれる。

### 2.3 補助ヘッド ([`models/v12_heads.py`](models/v12_heads.py) ほか)

- **`PhonemeAuxHead`** — 音声埋め込みから **392 クラスの音素事後確率** を予測する 2 層 MLP。
  クロスエントロピーで音声埋め込みを「音素として意味がある」方向に押す。推論コストはゼロ。
- **`PerGaussianAlbedoMLP`** — `(canonical xyz hashgrid, audio_emb, AU17)` → 各 Gaussian の **RGB 残差**。
  SH の DC 係数に小さく加算され、発音状態依存の影・しわを表現する。
- **Aperture aux head** — `train_fuse_v30e` 内に inline 定義。口元音声埋め込みから AU25/AU26 を直接回帰し、
  音声側を aperture-discriminative にする補助ロス。

### 2.4 知覚損失 ([`models/perceptual_losses.py`](models/perceptual_losses.py))

- **`ArcFaceIdentityLoss`** — InceptionResnetV1 (VGGFace2 学習) の 512-D 顔埋め込みを cos 距離で比較する。
  identity drift や色シフトを直接ペナルティ。
- **`DinoV2PerceptualLoss`** — DINOv2 ViT-S/14 のパッチ埋め込み (384-D × 256 patch) を L1 で比較。
  自然画像で自己教師あり学習されており、高周波ディテールに敏感。
- どちらも eval-only (パラメータ凍結)。fuse stage の既定は `arc_w = 0.1` / `dino_w = 0.5`。

### 2.5 データ・マスク経路

- [`scene/dataset_readers.py`](scene/dataset_readers.py)
  - `au_exp` (7 次元、`[1,4,5,6,7,45,25]`) を構築。
  - 眉領域の矩形 `ldmks_brow` を導出（将来の眉駆動用）。
  - `TG_LIP_CAVITY=1` のとき [`utils/lip_cavity_masks.py`](utils/lip_cavity_masks.py) 経由で
    `lip_mask` / `cavity_mask` を統合ロードする。
  - `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` で長尺シーケンスの frame 数を切り詰めて RAM を節約。
  - `TG_AU_CSV` で AU CSV ファイルを切り替え可能（補正済み AU の差し替え用）。
- [`utils/camera_utils.py`](utils/camera_utils.py)
  - `TG_LIP_CAVITY=1` で `lip_mask` / `cavity_mask` を優先ロード。
  - `TG_MOUTH_MASK_FULL=1` で口マスクを `face_parsing_fine` の `class 11 + 12 + 13`
    (口内 + 上唇 + 下唇) まで拡張。
  - `face_parsing_fine` を使って lazy-load パスでも `fp_eye_mask` を生成する。
- [`utils/soft_mask_utils.py`](utils/soft_mask_utils.py) — `mouth_core = erode(mouth_mask, 3)` と
  `mouth_overlap = dilate(mouth_mask, 3)` を導出。境界 1〜2 px の取り合いを緩和し、
  唇 ↔ 口内の seam・leak を防ぐ。alpha hard-constraint は `mouth_core` (mouth 側) と
  `~mouth_overlap` (face 側) にのみ適用する。
- [`scripts/build_lip_mask_3d.py`](scripts/build_lip_mask_3d.py) — 120 視点をサンプリングし、
  `face_parsing` の `class 11 + 12 + 13` に投票することで **per-Gaussian の lip mask 3D** を構築する。
- [`data_utils/easyportrait/create_lip_cavity_mask.py`](data_utils/easyportrait/create_lip_cavity_mask.py)
  — EasyPortrait と FP の組み合わせで 2D lip / cavity マスクを生成する。
- [`data_utils/extract_au_openface.py`](data_utils/extract_au_openface.py) — OpenFace `FeatureExtraction` の呼び出しラッパ。

### 2.6 Rasterizer 経路 ([`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py))

- `render()` / `render_motion()` / `render_motion_mouth()` に
  `override_xyz / override_scaling / override_rotation / override_opacity` を追加し、
  外部で計算したデフォーム後 Gaussian を直接差し込めるようにする (複数 deformer の合成に必要)。
- 使用する `diff-gaussian-rasterization` の版が `(image, radii, depth)` の 3-tuple しか返さない場合は、
  **白色 precomputed colors と同じ opacity でもう一度ラスタライズ**して alpha を再構成する。
  ダウンストリームの face/mouth alpha-blend 合成に必要な alpha チャンネルを常に保証する。
- `TG_ALPHA_GRAD=1` でこの alpha pass にも勾配を流す (既定 OFF、回帰確認後にのみ ON にする)。

### 2.7 Face stage ([`train_face_v30.py`](train_face_v30.py))

25k iter、HuBERT 音声特徴、`densify_grad_threshold = 0.0015`、NaN-safe guard + grad clip。

学習中に積まれるロス：

- 緑背景に対する **L1 + SSIM**。
- **唇 ROI LPIPS** (`0.01`) と **Patchified LPIPS** (`0.2`)。
- 各種微小 L1 正則化 — MotionNetwork の `d_xyz/d_rot/d_opa/d_scale` および cross-attn 残差 (`1e-5`)。
- **Mouth alpha loss** (`1e-2`) — 口内領域の不透明度を抑える。
- **Head mask alpha hinge** (`1e-3`) — 頭領域外で alpha がリークしないようにする。
- **P1 FaceLipRelease** (apex-aware) — apex フレームで face_alpha が唇を覆ってしまうのを抑えるペナルティ。
- **Patch landmark loss** — render 内の唇 landmark を GT に合わせる L1。
- **`features_dc` anchor** (`0.005`) — 初期 SH 係数からの drift 抑制。
- **`R-SUP-4` (gated)** — AU25 / AU45 の apex フレーム oversampling。
- **`R-SUP-3` (gated)** — d_xyz の時間平滑。

Cross-attn driver は **iter 0 から** 学習に参加する。

### 2.8 Mouth stage ([`train_mouth_v30.py`](train_mouth_v30.py))

50k iter、緑背景 + cavity 3× の重み付き L1、緑除外 SSIM。

主要な追加要素：

- **dual-head cross-attn (lip / cavity)**。densify ごとに `cavity_idx` を再計算。
- **Cavity depth prior** — `z_cavity > z_lip_median + 0.005` を満たさない Gaussian にペナルティ
  (口内が前に出ないようにする)。
- **Apex weight schedule** — AU25 / AU45 apex フレームでロスを `apex_w` 倍する。
- **R2 lip-y constraint** — render 内の上唇 / 下唇 y 位置を GT ランドマーク y に合わせる L1。
- **`R-ANISO-REG` (`TG_ANISO_REG_W`)** — `scaling.max() / scaling.min() > TG_ANISO_THR` の Gaussian にペナルティ。
- **`R-Z-PRUNE` (`TG_MOUTH_Z_MAX`)** — z scale が閾値超の Gaussian を強制 prune (Plan F)。
- **violent green prune を無効化 (noPrune)** — 真っ当な口元 Gaussian も巻き込んで殺していたため、
  soft prune に切り替え。
- **`features_dc` anchor** (`0.005`) と **cross-attn residual 微小 L1** (`1e-5`)。

### 2.9 Fuse stage ([`train_fuse_v30e.py`](train_fuse_v30e.py), [`scripts/build_fuse_v30_init.py`](scripts/build_fuse_v30_init.py))

10k iter で face + mouth + head priors を統合する段。

- **Hybrid 初期化** — face stage の top-50k Gaussian + mouth stage の全 Gaussian + 既存 V17 fuse の
  head priors (phoneme / albedo / aperture) を融合した `chkpnt_fuse_v30_init.pth` を作る。
- **Dual-head mouth cross-attn** (`cross_attn_driver_mouth_lip` + `cross_attn_driver_mouth_cavity`)。
- **PhonemeAuxHead** (`phoneme_w`)、**PerGaussianAlbedoMLP** (`albedo_lr` / `albedo_residual_scale`)、
  **Aperture aux head** (`aperture_w = 0.2`)。
- **AU sliding window** (`au_window_T = 8`) で cross-attn に時間文脈を渡す。
- **ArcFace identity** (`arc_w = 0.1`) + **DINOv2 perceptual** (`dino_w = 0.5`)
  + **Sobel detail** (`detail_w = 0.5`) + **features_dc anchor** (`feat_anchor_w = 0.005`)。

### 2.10 推論側 ([`synthesize_fuse_v18.py`](synthesize_fuse_v18.py), [`synthesize_fuse_v30e.py`](synthesize_fuse_v30e.py))

- 学習時と同一の AU sliding window と Cross-Attn 駆動を使用する。
- 後述の `TG_*` 環境変数で挙動を制御可能（学習せず推論時に運動振幅をスケールしたい用途）。
- dual-head 用の推論は `synthesize_fuse_v30e.py` を使う (lip head / cavity head を `cavity_idx` で切り分ける)。

---

## 3. 環境変数一覧

| 変数 | 既定 | 効果 |
| ---- | ---- | ---- |
| `TG_LIP_CAVITY` | `0` | `1` で lip / cavity 2D マスクを優先 |
| `TG_MOUTH_MASK_FULL` | `0` | `1` で口マスクを class 11+12+13 まで拡張 |
| `TG_MOUTH_Z_MAX` | 無効 | 口元 Gaussian の z scale 上限 (例 `0.05`) — Plan F |
| `TG_ANISO_REG_W` | `0` | 口元 Gaussian の異方性正則化重み (例 `0.001`) — Plan G |
| `TG_ANISO_THR` | `50` | 異方性閾値 (`scaling.max() / scaling.min()`) |
| `TG_AU45_EYE_GAIN` | `1.0` | 推論時 AU45 → 瞼運動アンプ |
| `TG_AU25_LIP_GAIN` | `1.0` | 推論時 AU25 → 口元 cross-attn 残差アンプ |
| `TG_MOUTH_XATTN_GAIN` | `1.0` | 推論時 mouth cross-attn 残差スケール |
| `TG_MOUTH_D_XYZ_GAIN` | `1.0` | 推論時 mouth d_xyz pre-cap ゲイン |
| `TG_AU_CSV` | `au.csv` | AU CSV ファイル切り替え |
| `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` | `0` | 長尺データの frame 数制限 (RAM 節約) |
| `TG_ALPHA_GRAD` | `0` | rasterizer の alpha pass に勾配を通す |
| `TG_USE_D2` / `TG_RSUP` | OFF | 進行中の D2 / R-SUP 実験フラグ群 |

---

## 4. 現在の結果（途中経過）

3 被験者（25fps、約 1〜3 分）で、ステージ F 評価値：

| 被験者 | 設定 | PSNR ↑ | LMD ↓ | 視覚 |
| ------ | ---- | ------ | ----- | ---- |
| macron | `v30au25` 既定 | **35.54** | **2.96** | 安定 |
| obama  | `v30au25` 既定 | **35.02** | **3.65** | 安定 |
| may    | `v30au25` + `TG_MOUTH_Z_MAX=0.05` | 29.92 | 4.10 | apex / 過渡フレームに残 artifact（**未解決**） |

これは固定 baseline ではなく**途中経過のチューニング数値**です。may は Plan G (`TG_ANISO_REG_W`)
を試験中。

---

## 5. インストール

Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1 で動作確認しています。

```bash
git clone https://github.com/kazehana99k/GauTalk-.git --recursive
cd GauTalk-
conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
# ArcFace / DINOv2 損失を使う場合
pip install facenet-pytorch
# DINOv2 は torch.hub から自動取得される
```

`diff-gaussian-rasterization` / `gridencoder` のビルドが失敗する場合は
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) と
[torch-ngp](https://github.com/ashawkey/torch-ngp) を参照してください。

### 前処理用モデルの取得

```bash
# 3DMM + face_parsing
bash scripts/prepare.sh

# Basel Face Model 2009 (01_MorphableModel.mat) を data_utils/face_tracking/3DMM/ に置く
cd data_utils/face_tracking && python convert_BFM.py && cd ../..

# EasyPortrait（歯マスク用）
pip install -U openmim && mim install mmcv-full==1.7.1
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
  -O data_utils/easyportrait/fpn-fp-512.pth

# OpenFace（AU 抽出）— 公式手順 https://github.com/TadasBaltrusaitis/OpenFace
```

---

## 6. データ前処理

```bash
# 1. 動画を 25fps / 約 512x512 で data/<ID>/<ID>.mp4 に配置
python data_utils/process.py data/<ID>/<ID>.mp4

# 2. OpenFace で AU を抽出し data/<ID>/au.csv に保存
#    AU25_r が含まれていることを必ず確認
python data_utils/extract_au_openface.py --root data/<ID>

# 3. 歯マスク
export PYTHONPATH=./data_utils/easyportrait
python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>

# 4. lip / cavity 2D マスク (TG_LIP_CAVITY=1 を使う場合に必須)
python data_utils/easyportrait/create_lip_cavity_mask.py ./data/<ID>

# 5. lip 3D マスク（mouth stage 後に Gaussian 側の lip 投票を取得）
python scripts/build_lip_mask_3d.py --root data/<ID> --ckpt <face stage ckpt>

# 6. 音声特徴
python data_utils/hubert.py --wav data/<ID>/aud.wav
# あるいは
python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>/aud.wav

# (任意) 音素事後確率 — PhonemeAuxHead 用
#   data/<ID>/aud_phoneme.npy に 392 クラスのフレーム別 posterior を置く
```

---

## 7. 学習（v30au25 既定パイプライン）

[`scripts/train_v30.sh`](scripts/train_v30.sh) または [`scripts/train_v30e.sh`](scripts/train_v30e.sh)
がワンショットで全段を実行します。段ごとの分解は以下：

```bash
dataset=data/<ID>
work=output/<ID>_v30au25
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
export TG_LIP_CAVITY=1   # lip/cavity マスクを使用

# A. Face stage (25k iter) — cross-attn from iter 0
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert \
  --init_num 2000 --densify_grad_threshold 0.0015 --iterations 25000
cp $work/chkpnt_face_v30_latest.pth $work/chkpnt_face_v30_clean.pth

# B. Mouth stage (50k iter) — dual-head cross-attn
#    必要に応じて Plan F / Plan G を有効化:
#      Plan F : export TG_MOUTH_Z_MAX=0.05
#      Plan G : export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000

# C. Fuse 初期化 (face + mouth + head priors を融合)
python scripts/build_fuse_v30_init.py \
  --face_ckpt $work/chkpnt_face_v30_clean.pth \
  --mouth_ckpt $work/chkpnt_mouth_v30_latest.pth \
  --head_prior <事前学習済み V17 fuse の .pth> \
  --face_max_pts 50000 --out $work/chkpnt_fuse_v30_init.pth

# D. Fuse 学習 (10k iter) — phoneme + albedo + aperture + ArcFace + DINOv2
python train_fuse_v30e.py -s $dataset -m $work \
  --init_ckpt $work/chkpnt_fuse_v30_init.pth \
  --opacity_lr 0.001 --audio_extractor hubert --total_iters 10000 \
  --au_window_T 8 --aperture_w 0.2 --detail_w 0.5 --feat_anchor_w 0.005 \
  --arc_w 0.1 --dino_w 0.5 --lpips_w 0.0

# F. 推論・評価
python synthesize_fuse_v18.py -s $dataset -m $work \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30_latest.pth \
  --output_dir $work/render_v30_full --max_frames 9999 --au_window_T 8
```

---

## 8. 任意音声での推論

```bash
python data_utils/hubert.py --wav new_audio.wav
python synthesize_fuse_v18.py -s data/<ID> -m output/<ID>_v30au25 \
  --use_train --audio new_audio_hu.npy \
  --ckpt_name chkpnt_fuse_v30_latest.pth
```

---

## 9. 未解決の課題 / ロードマップ

- [ ] **may の apex frame artifact** — `TG_MOUTH_Z_MAX=0.05` で部分改善するが完全解消には至っていない。
  Plan G (`TG_ANISO_REG_W=0.001`) と Plan F の同時併用を検証中。
- [ ] **`train_mouth_v2` 構想 (未マージ)** — `mouth_mask` を `lips_rect` + alpha hinge ≥ 0.95
  に拡張し、mouth init Gaussian を 350 → 1000 に増やす案。`face_parsing` の `mouth_mask` が
  apex で 52〜563 px しかないため口中心 alpha が 0 に潰れる根因への正面対処。
- [ ] **17-AU 個別可制御性の検証** — OpenFace 逆計測で AU の独立可制御性が低い状況を確認しており、
  クロス AU を許容しつつ目標 AU を保証する監督方式が必要。
- [ ] **論文用評価プロトコルの確定** — clean プロトコル (GT fallback なし) と α 後処理込みの数値が
  乖離している。論文には clean 路線を採用予定。
- [ ] **モデル ZOO** — checkpoint 公開と再現スクリプトの整備。
- [ ] **コード整理** — 試行錯誤期間に増えた `train_*.py` / `synthesize_*.py` の重複を順次淘汰し、
  v30au25 と train_fuse_v30e を正式パイプラインとする。
- [ ] **多言語音声でのロバスト性検証** — HuBERT 特徴で日本語・中国語の評価を進行中。

---

## 10. 既知の制限

- AU25 の数値スケールは OpenFace の出力に依存する。`au.csv` の `AU25_r` が概ね 0〜3 の範囲に
  収まっていることを学習前に確認してください。
- 英語以外の音声でも動作しますが、現状ロバスト性を確認しているのは HuBERT 特徴で駆動した場合のみ
  (DeepSpeech は英語推奨)。
- 長尺データの取り扱い時には `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` を設定しないと
  RAM が枯渇する可能性がある。
- `diff-gaussian-rasterization` の新版 (3-tuple 返却) を使う場合、alpha 再構成のため 1 iter
  あたり追加で 1 回ラスタライズが走る (`TG_ALPHA_GRAD=0` 既定では `no_grad` で安価)。

---

## 11. 謝辞

本プロジェクトは [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian)
のコードベースを土台に開発を始めました。3D Gaussian Splatting 関連の基盤ライブラリ
(gaussian-splatting, diff-gaussian-rasterization, simple-knn) と、データ前処理 / 補助モデル
(RAD-NeRF, ER-NeRF, EasyPortrait, OpenFace, AD-NeRF, GeneFace, DINOv2, facenet-pytorch)
の作者の皆様に感謝します。

---

## 12. ライセンス

研究用途のみ。LICENSE.md を参照してください。
