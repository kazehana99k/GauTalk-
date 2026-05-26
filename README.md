# TalkingGaussian-AU25 / GauTalk: AU25 駆動 + Cross-Attention 3D Gaussian Talking Head

> ⚠️ **本リポジトリは現在進行中の研究プロジェクトです (Work In Progress)。**
> API・学習スクリプト・チェックポイント名・環境変数は予告なく変更される可能性があります。
> 結果も最終形ではありません — 既知の問題と次の TODO は [§9](#9-未解決の課題--ロードマップ) を参照。

---

## 0. これは何か

オリジナル [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian)
は、顔のジオメトリと口元のジオメトリを 2 つの独立した 3D Gaussian で表現し、
音声特徴で MotionNetwork を駆動するパイプラインです。本リポジトリはこれを土台に、
以下を体系的に積み上げています。

- **AU25 (顎の開き) を MotionNetwork の明示入力に追加**し、長尺・分布外音声でも口内が「黒抜け」しないようにする。
- **Per-Gaussian Cross-Attention** で speech ⇄ AU ⇄ Gaussian の対応を動的に学習させる。
- **Dual-Head Mouth Cross-Attention** で唇 (lip) と口内 (cavity) を別ヘッドで駆動する。
- **音素・アパーチャ・アルベド** の補助ヘッド (PhonemeAuxHead / aperture head / PerGaussianAlbedoMLP) で
  暗黙的な制約を追加する。
- **ArcFace + DINOv2 ペアの知覚損失**で識別性とディテールを同時に保つ。
- **Soft mouth mask / lip / cavity 2D・3D マスク**で境界の seam・leak を抑える。
- **MotionNetwork 出力に tanh per-axis cap / y-axis HashGrid bypass / 異方性正則化 / Z-prune**
  を入れ、長尺音声で破綻していた口元 Gaussian を安定化する。

オリジナルの英語 README は [`README_en.md`](README_en.md) を参照してください。

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
                │       │  - exp_in_dim 6→7 (+AU25)              │
                │       │  - eye_dim    6→7 (AU45,AU25 raw)      │
                │       │  - per-axis tanh cap on d_xyz/d_scale  │
                │       │  - mouth: y-axis HashGrid bypass       │
                │       │  - mouth: au_mouth_branch (AU25 add)   │
                │       │  - mouth: lmk_proj+lmk_to_dxyz (lndmk) │
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
       │   3D Gaussian Splatting (patched)         │
       │  - override_xyz / opacity / rotation      │
       │  - alpha-pass reconstruction for new      │
       │    INRIA rasterizer (3-tuple return)      │
       └───────────────────────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────────────────────┐
       │ Losses                                    │
       │  L1 + SSIM + LPIPS                        │
       │  + ArcFace identity (face/fuse)           │
       │  + DINOv2 perceptual                      │
       │  + Sobel detail + features_dc anchor      │
       │  + cavity depth + apex apex_w schedule    │
       │  + R2 lip-y landmark constraint           │
       │  + temporal d_xyz smoothness (gated)      │
       └───────────────────────────────────────────┘
```

---

## 2. オリジナル → 本実装の差分（コンポーネント別）

### 2.1 MotionNetwork ([`scene/motion_net.py`](scene/motion_net.py))

| 項目 | オリジナル | 本実装 |
| ---- | ---------- | ------ |
| 表情入力 `au_exp` | 6 (AU01,04,05,06,07,45) | **7 (+ AU25)** |
| `eye_dim` | 6 (AU45 のみ raw) | **7 (AU45 + AU25 を末尾 raw)** |
| Face `d_xyz` cap | なし | **tanh per-axis `[0.015, 0.06, 0.025]`** |
| Face `d_scale` cap | なし | **tanh `0.5` (exp ratio 約 [0.6, 1.7])** |
| 推論時 AU45 → 眼運動アンプ | なし | **`TG_AU45_EYE_GAIN`（zero-AU45 reference との diff を eye_att でマスクして増幅）** |
| Mouth `d_xyz` cap | なし | **tanh per-axis `[0.025, 0.15, 0.04]`** |
| Mouth `y_bypass_proj` | なし | **音声 + y_rel → d_y 直接注入 (HashGrid 空間平滑のバイパス)** |
| Mouth `au_mouth_branch` | なし | **zero-init MLP: (xyz_feat + AU25) → d_xyz 残差** |
| Mouth ランドマーク条件 | なし | **`lmk_proj`(20×2→32) + `lmk_to_dxyz`(in+32→3) — Q1 加算分岐** |
| 推論時口元 d_xyz ゲイン | なし | **`TG_MOUTH_D_XYZ_GAIN`** |

### 2.2 Cross-Attention 駆動 ([`models/cross_attn_driver.py`](models/cross_attn_driver.py))

オリジナルには **存在しない** 新規モジュール。

- 各 Gaussian の HashGrid 特徴をクエリにし、**音声 8 トークン + AU 8 トークンの計 16 トークン**にアテンションをかける。
- 4 heads、d_model=128、positional embedding でモダリティ識別。
- 出力は `d_xyz / d_rot / d_opa / d_scale` の **小さな残差** (`residual_scale = 1e-3`)。
  既存 MotionNetwork の出力に**加算**する。
- AU 入力は **時間窓** (`au_window_T=8`) で渡せる（フレームごとに per-AU を時間方向に集約）。
- Face stage では 1 個、**Mouth stage では 2 個 (`lip head` と `cavity head`)** を使用し、
  `cavity_idx` で Gaussian を分割。`cavity_idx` は densify/prune ごとに再計算される。
- 推論時に **`TG_MOUTH_XATTN_GAIN`** で口元 cross-attn residual を増幅可能。
- **`TG_AU25_LIP_GAIN`** で「AU25=0 と現在 AU25 の residual 差分」を増幅する diagnostic モード。

### 2.3 補助ヘッド ([`models/v12_heads.py`](models/v12_heads.py))

- **`PhonemeAuxHead`** — 音声埋め込みから音素事後確率 (392 クラス) を予測。
  CE ロスで音声埋め込みを「音素として意味がある」方向に押す。推論コストはゼロ。
- **`PerGaussianAlbedoMLP`** — `(canonical xyz hashgrid + audio_emb + AU17)` →
  per-Gaussian の RGB 残差。SH DC に小さく加算する。発音状態依存の影・しわを補う。
- **Aperture aux head** (`train_fuse_v30e` 内 inline) — 口元音声埋め込みから AU25/AU26 を直接回帰。
  音声側を aperture-discriminative にする補助ロス。

### 2.4 知覚損失 ([`models/perceptual_losses.py`](models/perceptual_losses.py))

- **`ArcFaceIdentityLoss`** — InceptionResnetV1 (VGGFace2 学習) の 512-D 埋め込みを cos 距離で比較。
  identity drift や色シフトを直接ペナルティ。
- **`DinoV2PerceptualLoss`** — DINOv2 ViT-S/14 の 384-D × 256 patch 埋め込みを L1。
  VGG-LPIPS より高周波を保つ。
- どちらも eval-only (フリーズ済み)。fuse 段で `arc_w=0.1` / `dino_w=0.5` を既定としている。

### 2.5 データ・マスク経路

- [`scene/dataset_readers.py`](scene/dataset_readers.py)
  - `au_exp` 7 次元化 (`[1,4,5,6,7,45,25]`)。
  - `ldmks_brow` を新規追加 (将来の眉駆動用)。
  - lip/cavity マスク統合ロード経路を追加。
  - `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` で長尺シーケンスの RAM 切り詰め。
  - `TG_AU_CSV` で AU csv ファイルを切り替え可能（補正済み AU の差し替え用）。
- [`utils/camera_utils.py`](utils/camera_utils.py)
  - `TG_LIP_CAVITY=1` で `lip_mask` / `cavity_mask` を優先ロード。
  - `TG_MOUTH_MASK_FULL=1` で口マスクを `class 11 + 12 + 13`（口内＋上下唇）まで拡張。
  - `face_parsing_fine` クラスを使い fp_eye_mask を lazy-load パスでも生成。
- [`utils/lip_cavity_masks.py`](utils/lip_cavity_masks.py) — lip / cavity / teeth マスク統合ヘルパ。
- [`utils/soft_mask_utils.py`](utils/soft_mask_utils.py) — `mouth_core = erode(mouth_mask, 3)` と
  `mouth_overlap = dilate(mouth_mask, 3)` を導出。境界 1〜2 px の取り合いを緩和し、
  唇 ↔ 口内の seam / leak を防ぐ。
- [`scripts/build_lip_mask_3d.py`](scripts/build_lip_mask_3d.py) — 120 視点サンプリングと FP クラス
  投票で **per-Gaussian の lip mask 3D** を構築 (`face_parsing` の class 11+12+13)。
- [`data_utils/easyportrait/create_lip_cavity_mask.py`](data_utils/easyportrait/create_lip_cavity_mask.py)
  — EasyPortrait + FP の組み合わせで 2D lip/cavity マスクを生成。
- [`data_utils/extract_au_openface.py`](data_utils/extract_au_openface.py) — OpenFace 呼び出しラッパ。

### 2.6 Rasterizer 互換 ([`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py))

- `override_xyz / override_scaling / override_rotation / override_opacity` を追加。
  外部からデフォーム後の Gaussian を直接差し込めるようにし、複数 deformer の合成を容易化。
- 新 INRIA 版 `diff-gaussian-rasterization` が `(image, radii, depth)` の **3-tuple** しか返さない場合に、
  **白色 precomputed colors と同じ opacity でもう一度ラスタライズ**して alpha を再構成。
  ダウンストリームの face/mouth alpha-blend を温存。
- **`TG_ALPHA_GRAD=1`** で alpha pass に勾配を流す（既定 OFF、回帰確認後にのみ ON）。

### 2.7 Loss / Regularization (Face stage — [`train_face_v30.py`](train_face_v30.py))

- L1 + SSIM ベースに加えて：
  - **唇領域 LPIPS** + **Patchified LPIPS** (`0.01` / `0.2` 係数)。
  - `d_xyz / d_rot / d_opa / d_scale` の微小 L1 正則化 (`1e-5`)。
  - cross-attn residual の微小 L1 (`1e-5`)。
  - **mouth alpha loss** — `alpha[:, mouth_core].mean() * 1e-2`。
  - **head mask alpha hinge** — 頭領域外で alpha を抑える (`1e-3`)。
  - **P1 FaceLipRelease** — apex フレームの唇領域で face_alpha を抑えるペナルティ
    (apex における face → lip の遮蔽を解放)。
  - **Patch landmark loss** — render 内の lip landmark を GT に合わせる。
  - **`features_dc` anchor** — 初期 SH 係数から離れすぎないよう拘束 (`0.005`)。
  - **`R-SUP-4` (gated)** — AU45 / AU25 apex フレームの oversampling。
  - **`R-SUP-3` (gated)** — d_xyz の時間平滑 (`TEMPORAL_SMOOTH_W`)。
- 25k iter、`densify_grad_threshold = 0.0015` (オリジナル 0.001 より tighter)、NaN-safe guard + grad clip。

### 2.8 Loss / Regularization (Mouth stage — [`train_mouth_v30.py`](train_mouth_v30.py))

- 50k iter、緑背景での重み付き L1 (cavity 3×) + 緑除外 SSIM。
- **Apex weight schedule** — AU25/AU45 apex フレームでロスを `apex_w` 倍。
- **`R-ANISO-REG` (`TG_ANISO_REG_W`)** — 口元 Gaussian の異方性ペナルティ。
  `scaling.max() / scaling.min() > TG_ANISO_THR (既定 50)` の Gaussian を抑制。
- **`R-Z-PRUNE` (`TG_MOUTH_Z_MAX`)** — z scale が閾値超の Gaussian を強制 prune (Plan F)。
- **`R2 lip-y` constraint** — render の上唇/下唇 y 位置を GT landmark に合わせる L1。
- **cavity depth prior** — `z_cavity > z_lip_median + 0.005` を強制 (口内が前に出ないように)。
- **`features_dc` anchor**、**cross-attn residual 微小 L1**、**near-green prune は無効化 (noPrune)** など、
  個別の sink optimum 回避手当を導入。
- densify ごとに `cavity_idx` を再計算し、dual-head cross-attn を整合させる。

### 2.9 Fuse stage ([`train_fuse_v30e.py`](train_fuse_v30e.py), [`scripts/build_fuse_v30_init.py`](scripts/build_fuse_v30_init.py))

- **Hybrid 初期化** — V30 face (top-50k subsample) + V30 mouth + V17 fuse head priors を融合した
  `chkpnt_fuse_v30_init.pth` を構築。
- 10k iter で fuse 学習：
  - dual-head mouth cross-attn (`cross_attn_driver_mouth_lip` + `cross_attn_driver_mouth_cavity`)。
  - PhonemeAuxHead (`phoneme_w`)、PerGaussianAlbedoMLP (`albedo_lr` / `albedo_residual_scale`)、
    aperture aux head (`aperture_w=0.2`)。
  - **AU sliding window** (`au_window_T=8`) で cross-attn に時間文脈を渡す。
  - ArcFace identity (`arc_w=0.1`) + DINOv2 perceptual (`dino_w=0.5`) + Sobel detail (`detail_w=0.5`)
    + features_dc anchor (`feat_anchor_w=0.005`)。

### 2.10 推論側 ([`synthesize_fuse_v18.py`](synthesize_fuse_v18.py), [`synthesize_fuse_v30e.py`](synthesize_fuse_v30e.py))

- 学習時と同一の AU sliding window を用いるよう統一。
- 上記すべての `TG_*` 環境変数で挙動を制御可能（学習せず推論時にだけ強める用途）。
- dual-head 用は `synthesize_fuse_v30e.py` (lip head / cavity head を `cavity_idx` で切り分け)。

---

## 3. 環境変数一覧

| 変数 | 既定 | 効果 |
| ---- | ---- | ---- |
| `TG_LIP_CAVITY` | `0` | `1` で lip / cavity 2D マスクを優先 |
| `TG_MOUTH_MASK_FULL` | `0` | `1` で口マスクを class 11+12+13 まで拡張 |
| `TG_MOUTH_Z_MAX` | 無効 | 口元 Gaussian の z scale 上限 (例 `0.05`) — Plan F |
| `TG_ANISO_REG_W` | `0` | 口元 Gaussian の異方性正則化重み (例 `0.001`) — Plan G |
| `TG_ANISO_THR` | `50` | 異方性閾値 (scaling.max() / scaling.min()) |
| `TG_AU45_EYE_GAIN` | `1.0` | 推論時 AU45→瞼運動アンプ |
| `TG_AU25_LIP_GAIN` | `1.0` | 推論時 AU25→口元 cross-attn 残差アンプ |
| `TG_MOUTH_XATTN_GAIN` | `1.0` | 推論時 mouth cross-attn 残差スケール |
| `TG_MOUTH_D_XYZ_GAIN` | `1.0` | 推論時 mouth d_xyz pre-cap ゲイン |
| `TG_AU_CSV` | `au.csv` | AU CSV ファイル切り替え (補正済み AU 等) |
| `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` | `0` | 長尺データの frame 数制限 (RAM 節約) |
| `TG_ALPHA_GRAD` | `0` | 新版 rasterizer の alpha pass に勾配を通す |
| `TG_USE_D2` / `TG_RSUP` | OFF | 進行中の D2 / R-SUP 実験フラグ群 |

---

## 4. 現在の結果（途中経過）

3 被験者（25fps, 約 1〜3 分）で、ステージ F 評価値：

| 被験者 | 設定 | PSNR ↑ | LMD ↓ | 視覚 |
| ------ | ---- | ------ | ----- | ---- |
| macron | `v30au25` 既定 | **35.54** | **2.96** | 安定 |
| obama  | `v30au25` 既定 | **35.02** | **3.65** | 安定 |
| may    | `v30au25` + `TG_MOUTH_Z_MAX=0.05` | 29.92 | 4.10 | apex / 過渡フレームに残 artifact（**未解決**） |

- AU25 を明示入力したことで、**学習に出てこない長尺・任意音声でも口内が真っ黒にならない**ことを確認。
  オリジナル系の v29.7 系では `cam.original_image` を後処理で貼り直して見かけを保っていたのに対し、
  本実装は新規音声に対しても妥当な口内が描画される。
- 口の開閉幅は AU25 にほぼ線形に追従。Z-prune と異方性正則化で高アスペクト比な "飛ぶ Gaussian" を抑制。

**注:** これは固定 baseline ではなく、まだチューニング中の数字です。
特に may は Plan G (`TG_ANISO_REG_W`) を試験中。

---

## 5. インストール

オリジナル TalkingGaussian と同じ要件 (Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1) です。

```bash
git clone https://github.com/kazehana99k/GauTalk-.git --recursive
cd GauTalk-
conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
# ArcFace / DINOv2 損失を使う場合
pip install facenet-pytorch
# DINOv2 は torch.hub から自動取得されます
```

`diff-gaussian-rasterization` / `gridencoder` のビルドが失敗する場合は
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) と
[torch-ngp](https://github.com/ashawkey/torch-ngp) を参照してください。

### 前処理用モデルの取得

```bash
# AD-NeRF 由来の 3DMM + face_parsing
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
#    AU25_r が含まれていることを必ず確認（本実装の必須入力）
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
#   data/<ID>/aud_phoneme.npy として 392 クラスのフレーム別 posterior を置く
```

---

## 7. 学習（v30au25 既定パイプライン）

[`scripts/train_v30.sh`](scripts/train_v30.sh) または [`scripts/train_v30e.sh`](scripts/train_v30e.sh)
がワンショットで全段を実行します。段ごとの分解は以下。

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

# C. Fuse 初期化 (face + mouth + 既存 V17 head 事前知識を融合)
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
  Plan G (`TG_ANISO_REG_W=0.001`) と Plan F を同時併用する組み合わせを検証中。
- [ ] **`train_mouth_v2` 構想** (本実装では未マージ) — `mouth_mask` を `lips_rect` + alpha hinge ≥ 0.95
  に拡張し、mouth init Gaussian を 350 → 1000 に増やす案。`face_parsing` の `mouth_mask` が
  apex で 52〜563 px しかないため口中心 alpha が 0 に潰れる根因への正面対処。
- [ ] **17-AU 個別可制御性** — OpenFace 逆計測で各 AU の独立可制御性が低いことを確認しており、
  クロス AU を許容しつつ目標 AU を保証する監督方式が必要。
- [ ] **論文用評価プロトコルの確定** — TG-native clean（GT fallback なし）の数値と
  v29.7 soft α 系の数値が乖離している。論文には clean 路線を採用予定。
- [ ] **モデル ZOO** — checkpoint 公開と再現スクリプトの整備。
- [ ] **コード整理** — 試行錯誤期間に増えた `train_*.py` / `synthesize_*.py` の重複は
  順次淘汰する（現在は v30au25 と train_fuse_v30e を正式パイプラインとする）。
- [ ] **多言語音声でのロバスト性検証** — HuBERT 特徴で日本語・中国語の評価を進行中。

---

## 10. 既知の制限

- AU25 の数値スケールは OpenFace の出力に依存します。`au.csv` の `AU25_r` が
  概ね 0〜3 の範囲に収まっていることを学習前に確認してください。
- 本リポジトリは英語以外の音声でも動作しますが、HuBERT 特徴量を使う場合のみ
  ロバスト性を確認しています（DeepSpeech は英語推奨）。
- 学習・推論ともに長尺データを取り扱う場合、 `TG_MAX_TRAIN_FRAMES` /
  `TG_MAX_TEST_FRAMES` を設定しないと RAM が枯渇する可能性があります。
- 新版 INRIA `diff-gaussian-rasterization` を使う場合、alpha 再構成のため
  1 iter あたり追加で 1 回ラスタライズが走ります（`TG_ALPHA_GRAD=0` 既定では
  no_grad で安価）。

---

## 11. 謝辞

本実装は [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian)
を起点に、口元の AU 駆動と cross-attention 化を中心に拡張しています。
元コードベース・依存ライブラリ群（gaussian-splatting, diff-gaussian-rasterization,
simple-knn, RAD-NeRF, ER-NeRF, EasyPortrait, OpenFace, AD-NeRF, GeneFace,
DINOv2, facenet-pytorch 等）の作者の皆様に感謝します。

---

## 12. ライセンス

研究用途のみ。元 TalkingGaussian の LICENSE.md を継承します。
