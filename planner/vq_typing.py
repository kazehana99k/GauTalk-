"""種類分け（VQ モデル）— 雛形。

論文: VQ モデルは、長さの違うブロックを同じ長さの特徴に変換し、学習した代表のうち最も近い
ものの番号を種類とするニューラルネットワーク。代表は、割り当てられたブロックの平均へ動かす
ことを繰り返して学習する（k-means と同じ考え方）。代表の数（種類数）は未公開。
"""
from typing import Any

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


class BlockVQ:
    """ブロック → 種類番号。"""

    def __init__(self, n_types: int, n_points: int, feat_dim: int):
        self.n_types, self.n_points, self.feat_dim = n_types, n_points, feat_dim

    def fit(self, blocks: Any) -> "BlockVQ":
        """多人物のブロックで代表を学習する。"""
        raise NotImplementedError(NOT_RELEASED)

    def encode(self, block: Any) -> int:
        """1 ブロックを同じ長さにそろえ、最も近い代表の番号（種類）を返す。"""
        raise NotImplementedError(NOT_RELEASED)
