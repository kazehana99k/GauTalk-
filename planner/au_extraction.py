"""表情の抽出（OpenFace の Action Unit）— 雛形。

論文: OpenFace で顔の特徴点と見た目から AU の強さを推定し、眉と目に関わる
AU01・04・05・06・07・45 を使う。
"""
from typing import Any, Sequence

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"
AU_COLUMNS: Sequence[str] = ("AU01_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU45_r")


def extract_au(video_path: str, out_csv: str) -> Any:
    """OpenFace で毎フレームの AU 強度を抽出し、AU_COLUMNS の列を返す。"""
    raise NotImplementedError(NOT_RELEASED)
