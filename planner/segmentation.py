"""ブロック分割 — 雛形。

論文: 頭部運動は頭部回転の角速度が最も小さい所（頭がほぼ止まる所）で区切る。表情は
まばたき（AU45）を除いた AU の変化が最も遅くなる点で区切り、境界がまばたきの区間に
入らないようにする。ブロック長の範囲などの設定値は未公開。
"""
from typing import Any, List, Tuple

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def head_angular_speed(pose6: Any, fps: float) -> Any:
    """回転 3 成分から毎フレームの頭部回転の角速度を求める。"""
    raise NotImplementedError(NOT_RELEASED)


def segment_head(pose6: Any, fps: float, min_len: int, max_len: int) -> List[Tuple[int, int]]:
    """角速度の谷で区切り、[min_len, max_len] フレームのブロック境界を返す。"""
    raise NotImplementedError(NOT_RELEASED)


def segment_expression(au: Any, blink_intervals: Any, fps: float, min_len: int, max_len: int) -> List[Tuple[int, int]]:
    """AU45 を除いた AU の変化が最も遅い点で区切る。境界はまばたき区間を避ける。"""
    raise NotImplementedError(NOT_RELEASED)
