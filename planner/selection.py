"""実ブロックの選択・連結 — 雛形。

論文: 予測した種類の実ブロックを本人の特徴セットから優先して選び、現在の姿勢に滑らかに
つながるものを直前のブロックの後ろにつなぐ。ブロックは開始時の姿勢からの変化量として持つので、
境界で位置は途切れない。まばたきはブロックの中に残したまま再生される。
"""
from typing import Any, Dict, List

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def select_block(pred_type: int, person_set: Dict[int, List[Any]], pooled_set: Dict[int, List[Any]], current_pose: Any) -> Any:
    """予測した種類の実ブロックを本人の特徴セットから優先して 1 つ選ぶ。"""
    raise NotImplementedError(NOT_RELEASED)


def concatenate(blocks: List[Any], start_pose: Any) -> Any:
    """選んだブロックを開始姿勢から順につなぎ、頭部運動（または表情）の系列を返す。"""
    raise NotImplementedError(NOT_RELEASED)
