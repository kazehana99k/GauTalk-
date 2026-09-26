"""頭部姿勢の抽出（EMICA / FLAME 6DoF）— 雛形。

論文: 映像から 3D 頭部モデル FLAME のパラメータを推定するツール EMICA で、各フレームの顔に
FLAME を当てはめ、回転 3・並進 3 の 6 自由度（6DoF）を得る。
"""
from typing import Any

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def extract_head_pose(video_path: str, out_path: str, fps: float) -> Any:
    """動画から毎フレームの頭部姿勢 6DoF を抽出する。

    Args:
        video_path: 入力動画のパス。
        out_path: 出力先（フレーム数 x 6 の配列を保存）。
        fps: 動画のフレームレート。
    Returns:
        フレーム数 x 6 の配列（回転 3・並進 3）。
    """
    raise NotImplementedError(NOT_RELEASED)
