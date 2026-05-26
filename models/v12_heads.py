"""
V12 modules: PhonemeAuxHead + PerGaussianAlbedoMLP.

Purpose
-------
Add two orthogonal innovations on top of V10 fuse, without touching face_motion_net
or mouth_motion_net architectures (avoids re-training face/mouth from scratch).

1. PhonemeAuxHead
   - Auxiliary classifier that maps the encoded audio embedding back to a phoneme
     posterior.  Cross-entropy against the dataset's `aud_phoneme.npy` posteriors
     forces the audio embedding to be informative about phonemes — gives lip-sync
     a discrete linguistic prior without changing inference cost.

2. PerGaussianAlbedoMLP
   - Speech-conditioned per-Gaussian color residual.
   - Inputs: canonical xyz hashgrid + audio embedding + AU embedding
   - Output: small RGB residual added on the SH DC of every face Gaussian before
     rasterization.
   - Captures shading/wrinkle details that vary with speech state (e.g. dimples
     when smiling, nasolabial fold when speaking certain phonemes).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonemeAuxHead(nn.Module):
    """Predict phoneme posterior from an audio embedding.

    audio_emb -> Linear -> ReLU -> Linear -> log-softmax over n_phonemes.

    Used as an auxiliary loss only — no impact on inference.
    """

    def __init__(self, audio_dim: int = 32, hidden: int = 128, n_phonemes: int = 392):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_phonemes),
        )

    def forward(self, audio_emb: torch.Tensor) -> torch.Tensor:
        # audio_emb: [B, audio_dim] OR [audio_dim]; returns [B, n_phonemes] log-probs.
        if audio_emb.dim() == 1:
            audio_emb = audio_emb.unsqueeze(0)
        logits = self.net(audio_emb)
        return F.log_softmax(logits, dim=-1)


def phoneme_aux_loss(log_probs: torch.Tensor, target_post: torch.Tensor) -> torch.Tensor:
    """KL divergence between predicted log-probs and target posterior.
    log_probs : [B, K]  log-softmax output of PhonemeAuxHead
    target_post : [B, K] or [K]  ground-truth phoneme posterior (from aud_phoneme.npy).
    """
    if target_post.dim() == 1:
        target_post = target_post.unsqueeze(0)
    # Normalise target posterior in case it isn't already a distribution.
    t = target_post.clamp_min(1e-8)
    t = t / t.sum(dim=-1, keepdim=True)
    # KL(target || pred) = sum t * (log t - log_p)
    return (t * (t.clamp_min(1e-8).log() - log_probs)).sum(dim=-1).mean()


class PerGaussianAlbedoMLP(nn.Module):
    """Speech-conditioned per-Gaussian color residual.

    Architecture
    ------------
    xyz       --hashgrid_3plane-->  [N, 36]   (re-uses MotionNetwork-style tri-plane)
    audio     --Linear-->           [128]
    AU17      --Linear-->           [32]
    cond_global = repeat(audio + AU, N)
    h = MLP([xyz_feat, cond_global]) -> [N, 3] tanh-bounded residual
    Final color = SH_DC + tanh(residual) * scale     (default scale = 0.15)
    """

    def __init__(self, audio_dim: int = 32, au_dim: int = 17,
                 hidden: int = 128, residual_scale: float = 0.15,
                 bound: float = 0.15):
        super().__init__()
        from gridencoder import GridEncoder

        self.bound = bound
        self.residual_scale = residual_scale

        # Tri-plane hashgrid like MouthMotionNetwork (cheap, fast, ~12 levels).
        self.encoder_xy = GridEncoder(input_dim=2, num_levels=12, level_dim=1,
                                      base_resolution=16, log2_hashmap_size=17,
                                      desired_resolution=int(256 * bound))
        self.encoder_yz = GridEncoder(input_dim=2, num_levels=12, level_dim=1,
                                      base_resolution=16, log2_hashmap_size=17,
                                      desired_resolution=int(256 * bound))
        self.encoder_xz = GridEncoder(input_dim=2, num_levels=12, level_dim=1,
                                      base_resolution=16, log2_hashmap_size=17,
                                      desired_resolution=int(256 * bound))
        in_dim_xyz = self.encoder_xy.output_dim * 3

        self.audio_proj = nn.Linear(audio_dim, 64)
        self.au_proj    = nn.Linear(au_dim, 32)

        cond_dim = 64 + 32
        self.mlp = nn.Sequential(
            nn.Linear(in_dim_xyz + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        # Zero-init final layer so the network outputs 0 at start (identity).
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [N, 3]. Tri-plane hashgrid.
        xy, yz, xz = xyz[:, :-1], xyz[:, 1:], torch.cat([xyz[:, :1], xyz[:, -1:]], dim=-1)
        f = torch.cat([
            self.encoder_xy(xy, bound=self.bound),
            self.encoder_yz(yz, bound=self.bound),
            self.encoder_xz(xz, bound=self.bound),
        ], dim=-1)
        return f

    def forward(self, xyz: torch.Tensor, audio_emb: torch.Tensor,
                au: torch.Tensor) -> torch.Tensor:
        """Return per-Gaussian RGB residual of shape [N, 3], values in [-residual_scale, +residual_scale]."""
        N = xyz.shape[0]
        feat_xyz = self.encode_xyz(xyz)                    # [N, 36]
        a = self.audio_proj(audio_emb.flatten().unsqueeze(0)) if audio_emb.dim() <= 1 else \
            self.audio_proj(audio_emb.mean(dim=0, keepdim=True) if audio_emb.dim() == 2 else
                            audio_emb.mean(dim=(0, 1), keepdim=True).squeeze(0))
        u = self.au_proj(au.flatten().unsqueeze(0))        # [1, 32]
        cond = torch.cat([a, u], dim=-1).expand(N, -1)     # [N, 96]
        h = self.mlp(torch.cat([feat_xyz, cond], dim=-1))  # [N, 3]
        return torch.tanh(h) * self.residual_scale

    def get_params(self, lr: float = 5e-4, wd: float = 1e-6):
        return [
            {'params': list(self.encoder_xy.parameters()) +
                       list(self.encoder_yz.parameters()) +
                       list(self.encoder_xz.parameters()),
             'lr': lr, 'weight_decay': 0.0},
            {'params': list(self.audio_proj.parameters()) +
                       list(self.au_proj.parameters()) +
                       list(self.mlp.parameters()),
             'lr': lr, 'weight_decay': wd},
        ]
