"""人物ごとの正規化と、生成後に本人の大きさへ戻す処理 — 雛形。

論文: 6DoF と AU は人物ごとに正規化し、動きの大きさや偏りの違いを取り除く。汎用モデルは
正規化した動きで学習し、つないだ動きは対象人物の大きさに戻す。
"""
from typing import Any, Tuple

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


def fit_person_stats(pose6: Any, au: Any) -> Any:
    """対象人物の正規化に使う統計量を求める。"""
    raise NotImplementedError(NOT_RELEASED)


def normalize(pose6: Any, au: Any, stats: Any) -> Tuple[Any, Any]:
    """人物ごとの統計量で 6DoF と AU を正規化する。"""
    raise NotImplementedError(NOT_RELEASED)


def denormalize(pose6_norm: Any, au_norm: Any, stats: Any) -> Tuple[Any, Any]:
    """正規化した動きを対象人物の大きさに戻す。"""
    raise NotImplementedError(NOT_RELEASED)
