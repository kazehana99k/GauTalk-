"""
GaussianCrossAttnDriver — Per-Gaussian Cross-Attention Speech-Visual Decoder.

Each face Gaussian queries a tokenized speech context (audio frames + AU sub-groups)
and attends to the relevant tokens. This resolves the fundamental ambiguity of
audio→mouth-shape mapping (silent open mouth vs voiced open mouth) by letting
each Gaussian dynamically select which audio formants / AU dimensions matter.

Output is a SMALL motion residual added to the existing motion_net output.
Audio + AU are tokenised into 16 tokens; positional embeddings disambiguate them.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianCrossAttnDriver(nn.Module):
    def __init__(self,
                 xyz_feat_dim: int = 36,    # tri-plane hashgrid output dim
                 audio_seq_len: int = 8,
                 audio_dim: int = 32,        # encoded audio dim from existing motion_net.encode_audio
                 au_dim: int = 17,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_au_tokens: int = 8,
                 residual_scale_xyz: float = 1e-3,
                 residual_scale_scale: float = 1e-3,
                 residual_scale_rot: float = 1e-3):
        super().__init__()
        self.audio_seq_len = audio_seq_len
        self.n_au_tokens = n_au_tokens
        self.n_tokens = audio_seq_len + n_au_tokens          # 8 + 8 = 16
        self.d_model = d_model
        self.residual_scale_xyz = residual_scale_xyz
        self.residual_scale_scale = residual_scale_scale
        self.residual_scale_rot = residual_scale_rot

        # Tokenizers
        self.audio_token_proj = nn.Linear(audio_dim, d_model)
        # AU sub-group tokenizer (single-frame mode, kept for backward compat)
        self.au_tokenizer = nn.Linear(au_dim, n_au_tokens * d_model)
        # AU temporal-window tokenizer: project a per-frame 17-D AU into one
        # d_model token. Used when the caller passes a [T, au_dim] AU sequence
        # so the transformer attends over an AU sliding window instead of
        # a value sub-grouping of a single frame.
        self.au_temporal_proj = nn.Linear(au_dim, d_model)
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.randn(self.n_tokens, d_model) * 0.02)

        # Q-projection from per-Gaussian xyz feat
        self.q_proj = nn.Linear(xyz_feat_dim, d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Output head: 11 dims (3 xyz + 4 rot + 1 opa + 3 scale)
        self.out_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11),
        )
        # Zero-init final layer so identity at start (V14 motion preserved).
        with torch.no_grad():
            self.out_head[-1].weight.zero_()
            self.out_head[-1].bias.zero_()

    def forward(self,
                xyz_feat: torch.Tensor,           # [N, xyz_feat_dim]
                audio_seq_emb: torch.Tensor,      # [audio_seq_len, audio_dim] from motion_net.audio_net (sequence)
                au17: torch.Tensor                # [au_dim] (single-frame) or [T, au_dim] (temporal window)
                ) -> dict:
        N = xyz_feat.shape[0]

        # Tokenize speech
        if audio_seq_emb.dim() == 1:
            audio_seq_emb = audio_seq_emb.unsqueeze(0).repeat(self.audio_seq_len, 1)
        elif audio_seq_emb.shape[0] != self.audio_seq_len:
            # mean-pool to expected length if needed
            audio_seq_emb = audio_seq_emb.mean(dim=0, keepdim=True).repeat(self.audio_seq_len, 1)
        audio_tokens = self.audio_token_proj(audio_seq_emb)            # [8, d_model]

        # Two AU-tokenisation modes:
        #   - single frame  (au17.dim()==1): old sub-group split via au_tokenizer
        #   - temporal seq  (au17.dim()==2): per-frame projection via au_temporal_proj
        if au17.dim() == 2:
            T = au17.shape[0]
            if T < self.n_au_tokens:
                # left-pad with the earliest frame
                pad = au17[:1].expand(self.n_au_tokens - T, -1)
                au17 = torch.cat([pad, au17], dim=0)
            elif T > self.n_au_tokens:
                # centre-crop to expected length
                start = (T - self.n_au_tokens) // 2
                au17 = au17[start:start + self.n_au_tokens]
            au_tokens = self.au_temporal_proj(au17)                      # [n_au_tokens, d_model]
        else:
            au_tokens = self.au_tokenizer(au17.flatten()).view(self.n_au_tokens, self.d_model)  # [8, d_model]

        speech_tokens = torch.cat([audio_tokens, au_tokens], dim=0)    # [16, d_model]
        speech_tokens = speech_tokens + self.pos_emb                   # add positional

        # Per-Gaussian Q
        Q = self.q_proj(xyz_feat).unsqueeze(1)                         # [N, 1, d_model]
        # Broadcast K, V to [N, 16, d_model]
        K = V = speech_tokens.unsqueeze(0).expand(N, -1, -1)

        attended, attn_weights = self.cross_attn(Q, K, V, need_weights=False)  # [N, 1, d_model]
        attended = self.norm1(attended.squeeze(1))                     # [N, d_model]

        # Output
        h = self.out_head(attended)                                    # [N, 11]
        d_xyz   = h[..., :3]   * self.residual_scale_xyz
        d_rot   = h[..., 3:7]  * self.residual_scale_rot
        d_opa   = h[..., 7:8]  * self.residual_scale_scale
        d_scale = h[..., 8:11] * self.residual_scale_scale

        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
        }

    def get_params(self, lr: float = 5e-4, wd: float = 1e-6):
        return [
            {'params': list(self.parameters()), 'lr': lr, 'weight_decay': wd},
        ]
