# GauTalk — 設計ドキュメント (Work In Progress)

README は使い方中心です。**何がどう動いているのか**の詳細はこの DESIGN.md に集約します。

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

## 2. コンポーネント詳細

### 2.1 MotionNetwork ([`scene/motion_net.py`](scene/motion_net.py))

`MotionNetwork` は HashGrid 空間特徴と音声・AU から、各 Gaussian の差分姿勢
(`d_xyz, d_rot, d_opa, d_scale`) を回帰する MLP。

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
  `TG_MOUTH_XATTN_GAIN` で運動振幅を後付けでスケールできる。

### 2.2 Per-Gaussian Cross-Attention ([`models/cross_attn_driver.py`](models/cross_attn_driver.py))

`GaussianCrossAttnDriver` は各 Gaussian を独立のクエリとして、音声トークン列と AU トークン列に
アテンションをかけるモジュール。

- 入力は **8 個の音声トークン + 8 個の AU トークン = 16 トークン**。
- 4 heads / `d_model = 128` / positional embedding でモダリティを識別。
- AU は単フレーム入力と **時間窓入力 (`au_window_T=8`、過去 4 + 未来 4 フレーム)** の両方をサポート。
- 出力は `(d_xyz, d_rot, d_opa, d_scale)` の **小さな残差** (`residual_scale = 1e-3`)。
  MotionNetwork の出力に**加算**して使う。
- **Face stage** では 1 個の driver を使う。
- **Mouth stage** では `cavity_idx` で Gaussian を 2 グループに分け、
  **lip head と cavity head の 2 個**を使う。`cavity_idx` は densify/prune ごとに再計算される。

### 2.3 補助ヘッド ([`models/v12_heads.py`](models/v12_heads.py) ほか)

- **`PhonemeAuxHead`** — 音声埋め込みから **392 クラスの音素事後確率** を予測する 2 層 MLP。
  クロスエントロピーで音声埋め込みを「音素として意味がある」方向に押す。推論コストはゼロ。
- **`PerGaussianAlbedoMLP`** — `(canonical xyz hashgrid, audio_emb, AU17)` →
  各 Gaussian の RGB 残差。SH の DC 係数に小さく加算され、発音状態依存の影・しわを表現する。
- **Aperture aux head** — `train_fuse_v30e` 内に inline 定義。口元音声埋め込みから AU25/AU26 を直接回帰し、
  音声側を aperture-discriminative にする補助ロス。

### 2.4 知覚損失 ([`models/perceptual_losses.py`](models/perceptual_losses.py))

- **`ArcFaceIdentityLoss`** — InceptionResnetV1 (VGGFace2 学習) の 512-D 顔埋め込みを cos 距離で比較する。
- **`DinoV2PerceptualLoss`** — DINOv2 ViT-S/14 のパッチ埋め込み (384-D × 256 patch) を L1 で比較。
- どちらも eval-only。fuse stage の既定は `arc_w = 0.1` / `dino_w = 0.5`。

### 2.5 データ・マスク経路

- [`scene/dataset_readers.py`](scene/dataset_readers.py)
  - `au_exp` (7 次元、`[1,4,5,6,7,45,25]`) を構築。
  - 眉領域の矩形 `ldmks_brow` を導出（将来の眉駆動用）。
  - `TG_LIP_CAVITY=1` のとき [`utils/lip_cavity_masks.py`](utils/lip_cavity_masks.py) 経由で
    `lip_mask` / `cavity_mask` を統合ロードする。
  - `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` で frame 数を切り詰めて RAM 節約。
  - `TG_AU_CSV` で AU CSV ファイルを切り替え可能。
- [`utils/camera_utils.py`](utils/camera_utils.py)
  - `TG_LIP_CAVITY=1` で `lip_mask` / `cavity_mask` を優先ロード。
  - `TG_MOUTH_MASK_FULL=1` で口マスクを `face_parsing_fine` の `class 11 + 12 + 13`
    (口内 + 上唇 + 下唇) まで拡張。
  - `face_parsing_fine` を使って lazy-load パスでも `fp_eye_mask` を生成する。
- [`utils/soft_mask_utils.py`](utils/soft_mask_utils.py) — `mouth_core = erode(mouth_mask, 3)` と
  `mouth_overlap = dilate(mouth_mask, 3)` を導出し、唇 ↔ 口内の seam・leak を防ぐ。
- [`scripts/build_lip_mask_3d.py`](scripts/build_lip_mask_3d.py) — 120 視点をサンプリングし、
  `face_parsing` の class 11+12+13 に投票して **per-Gaussian の lip mask 3D** を構築する。
- [`data_utils/easyportrait/create_lip_cavity_mask.py`](data_utils/easyportrait/create_lip_cavity_mask.py)
  — EasyPortrait と FP の組み合わせで 2D lip / cavity マスクを生成する。
- [`data_utils/extract_au_openface.py`](data_utils/extract_au_openface.py) — OpenFace `FeatureExtraction` の呼び出しラッパ。

### 2.6 Rasterizer 経路 ([`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py))

- `render()` / `render_motion()` / `render_motion_mouth()` に
  `override_xyz / override_scaling / override_rotation / override_opacity` を追加し、
  外部で計算したデフォーム後 Gaussian を直接差し込めるようにする。
- 使用する `diff-gaussian-rasterization` の版が `(image, radii, depth)` の 3-tuple しか返さない場合は、
  **白色 precomputed colors と同じ opacity でもう一度ラスタライズ**して alpha を再構成する。
- `TG_ALPHA_GRAD=1` でこの alpha pass にも勾配を流す (既定 OFF)。

### 2.7 Face stage ([`train_face_v30.py`](train_face_v30.py))

25k iter、HuBERT 音声特徴、`densify_grad_threshold = 0.0015`、NaN-safe guard + grad clip。

- 緑背景に対する L1 + SSIM。
- 唇 ROI LPIPS (`0.01`) と Patchified LPIPS (`0.2`)。
- MotionNet `d_xyz/d_rot/d_opa/d_scale` および cross-attn 残差の微小 L1 (`1e-5`)。
- Mouth alpha loss (`1e-2`) — 口内領域の不透明度を抑える。
- Head mask alpha hinge (`1e-3`) — 頭領域外で alpha がリークしないようにする。
- P1 FaceLipRelease (apex-aware) — apex フレームで face_alpha が唇を覆ってしまうのを抑える。
- Patch landmark loss — render 内の唇 landmark を GT に合わせる L1。
- `features_dc` anchor (`0.005`) — 初期 SH 係数からの drift 抑制。
- `R-SUP-4` (gated) — AU25 / AU45 の apex フレーム oversampling。
- `R-SUP-3` (gated) — d_xyz の時間平滑。

Cross-attn driver は iter 0 から学習に参加する。

### 2.8 Mouth stage ([`train_mouth_v30.py`](train_mouth_v30.py))

50k iter、緑背景 + cavity 3× の重み付き L1、緑除外 SSIM。

- dual-head cross-attn (lip / cavity)。densify ごとに `cavity_idx` を再計算。
- Cavity depth prior — `z_cavity > z_lip_median + 0.005` を満たさない Gaussian にペナルティ。
- Apex weight schedule — AU25 / AU45 apex フレームでロスを `apex_w` 倍する。
- R2 lip-y constraint — render 内の上唇 / 下唇 y 位置を GT ランドマーク y に合わせる L1。
- `R-ANISO-REG` (`TG_ANISO_REG_W`) — `scaling.max() / scaling.min() > TG_ANISO_THR` の Gaussian にペナルティ。
- `R-Z-PRUNE` (`TG_MOUTH_Z_MAX`) — z scale が閾値超の Gaussian を強制 prune (Plan F)。
- violent green prune を無効化 (noPrune) — 正常な口元 Gaussian も巻き込んで殺していたため、soft prune に切替。
- `features_dc` anchor (`0.005`) と cross-attn residual 微小 L1 (`1e-5`)。

### 2.9 Fuse stage ([`train_fuse_v30e.py`](train_fuse_v30e.py), [`scripts/build_fuse_v30_init.py`](scripts/build_fuse_v30_init.py))

10k iter で face + mouth + head priors を統合する段。

- Hybrid 初期化 — face stage の top-50k Gaussian + mouth stage の全 Gaussian + 既存 V17 fuse の
  head priors を融合した `chkpnt_fuse_v30_init.pth` を作る。
- Dual-head mouth cross-attn (`cross_attn_driver_mouth_lip` + `cross_attn_driver_mouth_cavity`)。
- PhonemeAuxHead (`phoneme_w`)、PerGaussianAlbedoMLP、Aperture aux head (`aperture_w = 0.2`)。
- AU sliding window (`au_window_T = 8`) で cross-attn に時間文脈を渡す。
- ArcFace identity (`arc_w = 0.1`) + DINOv2 perceptual (`dino_w = 0.5`) + Sobel detail (`detail_w = 0.5`)
  + features_dc anchor (`feat_anchor_w = 0.005`)。

### 2.10 推論側 ([`synthesize_fuse_v18.py`](synthesize_fuse_v18.py), [`synthesize_fuse_v30e.py`](synthesize_fuse_v30e.py))

- 学習時と同一の AU sliding window と Cross-Attn 駆動を使用。
- `TG_*` 環境変数で挙動を制御可能（学習せず推論時に運動振幅をスケールしたい用途）。
- dual-head 用の推論は `synthesize_fuse_v30e.py` を使う。

## 3. 環境変数一覧

| 変数 | 既定 | 効果 |
| ---- | ---- | ---- |
| `TG_LIP_CAVITY` | `0` | `1` で lip / cavity 2D マスクを優先 |
| `TG_MOUTH_MASK_FULL` | `0` | `1` で口マスクを class 11+12+13 まで拡張 |
| `TG_MOUTH_Z_MAX` | 無効 | 口元 Gaussian の z scale 上限 (例 `0.05`) |
| `TG_ANISO_REG_W` | `0` | 口元 Gaussian の異方性正則化重み (例 `0.001`) |
| `TG_ANISO_THR` | `50` | 異方性閾値 (`scaling.max() / scaling.min()`) |
| `TG_AU45_EYE_GAIN` | `1.0` | 推論時 AU45 → 瞼運動アンプ |
| `TG_AU25_LIP_GAIN` | `1.0` | 推論時 AU25 → 口元 cross-attn 残差アンプ |
| `TG_MOUTH_XATTN_GAIN` | `1.0` | 推論時 mouth cross-attn 残差スケール |
| `TG_MOUTH_D_XYZ_GAIN` | `1.0` | 推論時 mouth d_xyz pre-cap ゲイン |
| `TG_AU_CSV` | `au.csv` | AU CSV ファイル切り替え |
| `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` | `0` | 長尺データの frame 数制限 (RAM 節約) |
| `TG_ALPHA_GRAD` | `0` | rasterizer の alpha pass に勾配を通す |
| `TG_USE_D2` / `TG_RSUP` | OFF | 進行中の D2 / R-SUP 実験フラグ群 |

## 4. ロードマップ

- [ ] **may の apex frame artifact** — `TG_MOUTH_Z_MAX=0.05` で部分改善するが完全解消には至っていない。
  Plan G (`TG_ANISO_REG_W=0.001`) と Plan F の同時併用を検証中。
- [ ] **`train_mouth_v2` 構想 (未マージ)** — `mouth_mask` を `lips_rect` + alpha hinge ≥ 0.95
  に拡張し、mouth init Gaussian を 350 → 1000 に増やす案。
- [ ] **17-AU 個別可制御性の検証**。
- [ ] **論文用評価プロトコルの確定** — clean プロトコル (GT fallback なし) を採用予定。
- [ ] **モデル ZOO** — checkpoint 公開と再現スクリプトの整備。
- [ ] **コード整理** — 試行錯誤期間に増えた `train_*.py` / `synthesize_*.py` の重複を順次淘汰。
- [ ] **多言語音声でのロバスト性検証** (HuBERT 特徴で日本語・中国語を評価中)。
