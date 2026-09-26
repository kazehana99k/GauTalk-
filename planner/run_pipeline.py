"""一連の実行の骨組み — 雛形。

    多人物データ ─┬─ 抽出 → 正規化 → 分割 → 種類分け → 特徴セット ─→ 汎用 Transformer の学習
                  │
    対象人物の動画 ┴─ 同じ処理 → 本人の特徴セット ─→ 微調整
    生成: 種類を予測 → 本人の実ブロックを選んで連結 → 大きさを戻す → 描画モデルへ
"""
import argparse

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool_dir", help="多人物データのディレクトリ")
    p.add_argument("--person_dir", help="対象人物の動画・特徴のディレクトリ")
    p.add_argument("--audio", help="生成に使う音声ファイル")
    p.add_argument("--out_dir", help="描画モデル入力の出力先")
    p.parse_args()
    raise NotImplementedError(NOT_RELEASED)


if __name__ == "__main__":
    main()
