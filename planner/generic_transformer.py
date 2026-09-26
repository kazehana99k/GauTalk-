"""汎用 Transformer（運動予測モデル）— 雛形。

論文: 入力は直前までの運動ブロックの列、現在の頭部姿勢（6DoF）、音声特徴で、出力は次の
運動ブロックの種類。多人物データで学習し、どの動きの後にどの動きが続きやすいか（時間）と、
どの姿勢からどの動きが起こりやすいか（空間）の規則性を学習する。構造・学習条件は未公開。
"""
from typing import Any, Optional

NOT_RELEASED = "未公開: 論文公開時に追加します / not released yet"


class GenericTransformer:
    def __init__(self, n_types: int, pose_dim: int, audio_dim: Optional[int]):
        self.n_types, self.pose_dim, self.audio_dim = n_types, pose_dim, audio_dim

    def train(self, pooled_sequences: Any) -> "GenericTransformer":
        """多人物データのブロック列・姿勢・音声特徴で学習する。"""
        raise NotImplementedError(NOT_RELEASED)

    def predict_next_type(self, prev_types: Any, current_pose: Any, audio_feat: Optional[Any]) -> Any:
        """次のブロックの種類の分布を返す。"""
        raise NotImplementedError(NOT_RELEASED)
