# planner/ — 頭部運動・表情プランナ（インタフェースの雛形）

本ディレクトリは **未公開部分のインタフェースだけ** を示す雛形です。各関数は入出力の形を
docstring に書き、本体は `NotImplementedError` を送出します。実装・学習済み重み・設定値
（ブロック長、種類数、学習条件など）は論文公開時に追加します。

論文で用いている名前をそのまま使います。

| 用語 | 意味 |
| --- | --- |
| 運動ブロック | 頭部運動または表情の短い区間 |
| 実ブロック | 対象人物が実際に行った動きのブロック |
| 種類 | ブロックを分類したグループ |
| 特徴セット | ブロックを種類ごとにまとめたもの（対象人物のものは実ブロックから作る） |
| 汎用 Transformer | 多人物データで学習した、次のブロックの種類を予測するモデル |
| 微調整 | 汎用 Transformer を対象人物のデータに合わせること |

## 段階と対応ファイル

| 段階 | ファイル | 入力 → 出力 |
| --- | --- | --- |
| 頭部姿勢の抽出（EMICA / FLAME 6DoF） | [`pose_extraction.py`](pose_extraction.py) | 動画 → 毎フレームの 6DoF |
| 表情の抽出（OpenFace AU01・04・05・06・07・45） | [`au_extraction.py`](au_extraction.py) | 動画 → 毎フレームの AU |
| 人物ごとの正規化と復元 | [`normalization.py`](normalization.py) | 6DoF / AU → 正規化した系列、およびその逆 |
| ブロック分割 | [`segmentation.py`](segmentation.py) | 系列 → ブロック境界 |
| 種類分け（VQ モデル） | [`vq_typing.py`](vq_typing.py) | ブロック → 種類番号 |
| 特徴セットの構築 | [`feature_set.py`](feature_set.py) | ブロックと種類 → 種類ごとの集合 |
| 汎用 Transformer | [`generic_transformer.py`](generic_transformer.py) | 直前の種類列・現在の姿勢・音声特徴 → 次の種類 |
| 微調整 | [`finetune.py`](finetune.py) | 汎用 Transformer ＋ 対象人物のデータ → 対象人物用モデル |
| 実ブロックの選択・連結 | [`selection.py`](selection.py) | 予測した種類 ＋ 特徴セット → 頭部運動・表情の系列 |
| 描画モデルへの入力 | [`render_interface.py`](render_interface.py) | 頭部運動・表情・音声 → 描画モデルの入力形式 |
| 一連の実行 | [`run_pipeline.py`](run_pipeline.py) | 上記を順に呼ぶ骨組み |

## 描画モデルとの関係

本リポジトリの描画モデル（`train_*.py` / `synthesize_*.py`）は、学習動画から得た頭部姿勢
（`transforms_train.json`）と AU（`au.csv`）で描画します。プランナが生成した頭部運動・表情を
描画モデルに渡す経路は `render_interface.py` の雛形にあたり、未公開です。
