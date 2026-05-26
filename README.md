# TalkingGaussian-AU25: AU25 駆動 3D Gaussian Talking Head

本リポジトリは ECCV 2024 [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian)
をベースに、**Action Unit 25（顎の開き）を motion network に明示的に組み込む**ことで
口の開閉表現を強化し、長尺・任意音声でも口内が「黒抜け」しないように改良したものです。

オリジナルの英語 README は [`README_en.md`](README_en.md) を参照してください。

---

## 1. 本実装の主な変更点

オリジナル TalkingGaussian は、表情入力として `[AU01, AU04, AU05, AU06, AU07, AU45]` の
6 次元のみを `MotionNetwork` に渡しており、**顎の開き (AU25) を明示的に扱っていません**。
そのため、新規音声に対する推論時に口の内側が描画されず、合成段で `cam.original_image`
を貼り直して見かけ上ごまかすという挙動になっていました。

本実装では下記の点を変更しています。

| 項目 | オリジナル | 本実装 |
| ---- | ---------- | ------ |
| MotionNet 表情入力 `au_exp` | 6 次元 | **7 次元（末尾に AU25 を追加）** |
| MotionNet `eye_dim` | 6（AU45 のみ raw） | **7（AU45・AU25 を raw 出力末尾に保持）** |
| 口元 Gaussian の駆動 | 音声のみ | **音声 + AU25 加算分岐 (`au_mouth_branch`)** |
| 口元 `d_xyz` の発散対策 | なし | **tanh per-axis cap (`[0.025, 0.15, 0.04]`)** |
| 上下唇の対称化抑制 | HashGrid 平滑のみ | **`y_bypass_proj` で y 軸を直接注入** |
| Mouth 用マスク | parsing 灰色 + teeth | **lip_mask / cavity_mask 統合（V28）** |
| Mouth Gaussian の過密抑制 | なし | **Z-prune (`TG_MOUTH_Z_MAX`) と aniso reg (`TG_ANISO_REG_W`) を環境変数で切替** |
| 推論時 alpha チャンネル | rasterizer 依存 | **新 INRIA 版 (3 値返却) との互換パッチ** |

主要コードの修正点：

- [`scene/dataset_readers.py`](scene/dataset_readers.py) — `au_exp` を 7 次元化し AU25 を追加。
  眉領域 (`brow_rect`) と lip/cavity マスクの統合経路を追加。
- [`scene/motion_net.py`](scene/motion_net.py) — Face/Mouth 両 MotionNet に
  AU25 経路・tanh cap・y バイパス・推論時 AU45 アンプを追加。
- [`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py) —
  `override_xyz / override_scaling / override_rotation / override_opacity` を追加し、
  新版 `diff-gaussian-rasterization` が 3 値しか返さない場合の alpha 互換パスを実装。
- [`utils/camera_utils.py`](utils/camera_utils.py) — `TG_LIP_CAVITY=1` で lip/cavity
  マスクを優先ロード。`TG_MOUTH_MASK_FULL=1` で口マスクを上下唇まで拡張。
- [`utils/lip_cavity_masks.py`](utils/lip_cavity_masks.py) — lip / cavity / teeth
  マスクを統合する薄いヘルパ（新規）。
- [`utils/soft_mask_utils.py`](utils/soft_mask_utils.py) — soft mouth mask（新規）。
- v30 ステージ系スクリプト一式
  ([`train_face_v30.py`](train_face_v30.py),
   [`train_mouth_v30.py`](train_mouth_v30.py),
   [`scripts/build_fuse_v30_init.py`](scripts/build_fuse_v30_init.py),
   [`train_fuse_v30e.py`](train_fuse_v30e.py),
   [`synthesize_fuse_v18.py`](synthesize_fuse_v18.py))。

---

## 2. 達成した効果

3 被験者（25fps, 約 1〜3 分）で、ステージ F (`synthesize_fuse_v18.py`) の評価値：

| 被験者 | 設定 | PSNR ↑ | LMD ↓ | 視覚 |
| ------ | ---- | ------ | ----- | ---- |
| macron | v30au25 既定 | **35.54** | **2.96** | 安定 |
| obama  | v30au25 既定 | **35.02** | **3.65** | 安定 |
| may    | v30au25 + Z-prune (`TG_MOUTH_Z_MAX=0.05`) | 29.92 | 4.10 | apex / 過渡フレームに残アーチファクト |

- AU25 を明示入力したことで、**学習に出てこない長尺音声でも口内が真っ黒にならない**ことを確認。
- 口の開閉幅は AU25 にほぼ線形に追従し、Z-prune・aniso 正則化により高アスペクト比な
  「飛ぶ Gaussian」を抑制。

---

## 3. インストール

オリジナル TalkingGaussian と同じ要件 (Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1) です。

```bash
git clone <this-repo> --recursive
conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
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

## 4. データ前処理

```bash
# 1. 動画を 25fps / 約 512x512 で data/<ID>/<ID>.mp4 に配置
python data_utils/process.py data/<ID>/<ID>.mp4

# 2. OpenFace で AU を抽出し data/<ID>/au.csv に保存
#    AU25_r が含まれることを必ず確認（本実装の必須入力）

# 3. 歯マスク
export PYTHONPATH=./data_utils/easyportrait
python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>

# 4. lip / cavity 3D マスク（V28 以降に必須）
python scripts/build_lip_mask_3d.py --root data/<ID>

# 5. 音声特徴
python data_utils/hubert.py --wav data/<ID>/aud.wav
# あるいは
python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>/aud.wav
```

---

## 5. 学習（v30au25 既定パイプライン）

`scripts/train_v30.sh` または `scripts/train_v30e.sh` がワンショットで全段を実行します。
ここでは段ごとに分解して示します。

```bash
dataset=data/<ID>
work=output/<ID>_v30au25
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
export TG_LIP_CAVITY=1   # lip/cavity マスクを使用

# A. Face stage (25k iter)
python train_face_v30.py -s $dataset -m $work --audio_extractor hubert \
  --init_num 2000 --densify_grad_threshold 0.0015 --iterations 25000
cp $work/chkpnt_face_v30_latest.pth $work/chkpnt_face_v30_clean.pth

# B. Mouth stage (50k iter)
#    必要に応じて Plan F / Plan G を有効化:
#      Plan F : export TG_MOUTH_Z_MAX=0.05
#      Plan G : export TG_ANISO_REG_W=0.001
python train_mouth_v30.py -s $dataset -m $work --audio_extractor hubert --iterations 50000

# C. Fuse 初期化（face + mouth + 既存 V17 head 事前知識を融合）
python scripts/build_fuse_v30_init.py \
  --face_ckpt $work/chkpnt_face_v30_clean.pth \
  --mouth_ckpt $work/chkpnt_mouth_v30_latest.pth \
  --head_prior <事前学習済み V17 fuse の .pth> \
  --face_max_pts 50000 --out $work/chkpnt_fuse_v30_init.pth

# D. Fuse 学習 (10k iter)
python train_fuse_v30e.py -s $dataset -m $work \
  --init_ckpt $work/chkpnt_fuse_v30_init.pth \
  --opacity_lr 0.001 --audio_extractor hubert --total_iters 10000

# F. 推論・評価
python synthesize_fuse_v18.py -s $dataset -m $work \
  --eval --audio_extractor hubert \
  --ckpt_name chkpnt_fuse_v30_latest.pth \
  --output_dir $work/render_v30_full --max_frames 9999 --au_window_T 8
```

### 環境変数一覧（本実装で追加）

| 変数 | 既定 | 用途 |
| ---- | ---- | ---- |
| `TG_LIP_CAVITY` | `0` | `1` で lip/cavity マスクを優先使用 |
| `TG_MOUTH_MASK_FULL` | `0` | `1` で口マスクを class 11+12+13（口内+上下唇）に拡張 |
| `TG_MOUTH_Z_MAX` | `0`（無効） | 口元 Gaussian の z scale 上限。例: `0.05` |
| `TG_ANISO_REG_W` | `0`（無効） | 口元 Gaussian の異方性正則化重み。例: `0.001` |
| `TG_AU45_EYE_GAIN` | `1.0` | 推論時の AU45→瞼運動アンプ（>1 で強調） |
| `TG_MOUTH_D_XYZ_GAIN` | `1.0` | 推論時の口元 d_xyz pre-cap ゲイン |
| `TG_AU_CSV` | `au.csv` | 別 AU csv ファイルを指定したい場合 |
| `TG_MAX_TRAIN_FRAMES` / `TG_MAX_TEST_FRAMES` | `0`（無効） | 長尺の学習時にフレームを切り詰めて RAM を節約 |
| `TG_ALPHA_GRAD` | `0` | 新版 rasterizer での alpha-pass に勾配を流す場合 `1` |

---

## 6. 任意音声での推論

```bash
python data_utils/hubert.py --wav new_audio.wav
python synthesize_fuse_v18.py -s data/<ID> -m output/<ID>_v30au25 \
  --use_train --audio new_audio_hu.npy \
  --ckpt_name chkpnt_fuse_v30_latest.pth
```

---

## 7. 既知の制限

- macron / obama では問題ないが、**may では apex 付近のフレームに僅かな artifact が残る**。
  `TG_MOUTH_Z_MAX=0.05` の Z-prune で改善するが完全解消には至っておらず、
  `TG_ANISO_REG_W=0.001` を併用する Plan G を試験中です。
- AU25 の数値スケールは OpenFace の出力に依存します。`au.csv` の `AU25_r` が
  概ね 0〜3 の範囲に収まっていることを学習前に確認してください。
- 本リポジトリは英語以外の音声でも動作しますが、HuBERT 特徴量を使う場合のみ
  ロバスト性を確認しています（DeepSpeech は英語推奨）。

---

## 8. 謝辞

本実装は [TalkingGaussian (ECCV 2024)](https://github.com/Fictionarry/TalkingGaussian)
を起点に、口元の AU 駆動を中心に改造しています。元コードベース・依存ライブラリ群
（gaussian-splatting, diff-gaussian-rasterization, simple-knn, RAD-NeRF, ER-NeRF,
EasyPortrait, OpenFace, AD-NeRF, GeneFace 等）の作者の皆様に感謝します。

---

## 9. ライセンス

研究用途のみ。元 TalkingGaussian の LICENSE.md を継承します。
