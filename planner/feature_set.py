"""特徴セットの構築 — 雛形。

論文: ブロックを種類ごとにまとめたものを特徴セットと呼ぶ。対象人物の動画にも同じ VQ モデルを
そのまま使い、実ブロックを種類ごとにまとめた対象人物の特徴セットを得る。
"""
from typing import Any, Dict, List

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def build_feature_set(blocks: List[Any], types: List[int]) -> Dict[int, List[Any]]:
    """種類番号 → その種類のブロック一覧。"""
    raise NotImplementedError(NOT_RELEASED)


def build_person_feature_set(person_blocks: List[Any], vq: Any) -> Dict[int, List[Any]]:
    """対象人物の実ブロックを同じ VQ モデルで分類し、本人の特徴セットを作る。"""
    raise NotImplementedError(NOT_RELEASED)
