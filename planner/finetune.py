"""汎用 Transformer の微調整 — 雛形。

論文: 学習済みの汎用モデルを、対象人物の動画から得た本人のデータで微調整する。
入力の種類は汎用モデルと同じで、中身は本人のデータ。手順・条件は未公開。
"""
from typing import Any

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def finetune_for_person(generic_model: Any, person_sequences: Any) -> Any:
    """対象人物用に調整したモデルを返す。"""
    raise NotImplementedError(NOT_RELEASED)
