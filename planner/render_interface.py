"""生成した頭部運動・表情を描画モデルへ渡す経路 — 雛形。

現状の描画スクリプト（synthesize_*.py）は学習動画の頭部姿勢（transforms_train.json）と
AU（au.csv）で描画する。ここでは、プランナの出力を同じ形式に変換して渡す入口だけを示す。
"""
from typing import Any

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def to_renderer_inputs(head_pose6: Any, au: Any, audio_feature_path: str, out_dir: str) -> None:
    """頭部運動（6DoF 系列）と AU 系列を、描画モデルが読む transforms / au.csv の形式に書き出す。"""
    raise NotImplementedError(NOT_RELEASED)
