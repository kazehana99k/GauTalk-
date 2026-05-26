"""
Soft Mask Boundary helpers (V17).

Goal: resolve the hard-cut seam between mouth-interior and lip Gaussians.

The original TG protocol partitions the head into a hard mouth_mask (BiSeNet
inside-mouth + EasyPortrait teeth). Lip pixels right next to that seam are
exclusively assigned to the *face* branch, and inside-mouth pixels exclusively
to the *mouth* branch. Whenever lip motion crosses the seam (yawning, voicing
"o", lower lip pulled inward) one branch loses pixels it actually should
render, leaving an under-filled lip. Likewise the mouth branch sometimes has a
sliver of lip leaking into its mask which it has to render with no lip prior.

Soft Mask Boundary fixes this with two derived masks:
- mouth_core    = erode(mouth_mask, k=3)   # interior we KEEP exclusive to mouth
- mouth_overlap = dilate(mouth_mask, k=3)  # extended region BOTH branches learn

Both branches train with GT colour inside `mouth_overlap`; alpha hard-constraint
is only applied in `mouth_core` (mouth) and outside `mouth_overlap` (face).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _to_float_chw(mask: torch.Tensor) -> torch.Tensor:
    """Normalize bool/float HxW mask to a [1,1,H,W] float tensor."""
    if mask.dim() == 2:
        m = mask[None, None].float()
    elif mask.dim() == 3:
        m = mask[None].float()
    else:
        m = mask.float()
    return m


def dilate_mask(mask: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """Binary dilation via max-pool with stride 1, returns bool same shape as input."""
    m = _to_float_chw(mask)
    pool = torch.nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)
    out = pool(m)
    out = out.view(*mask.shape).bool()
    return out


def erode_mask(mask: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """Binary erosion = -dilate(-x), via max-pool on negated mask."""
    m = _to_float_chw(mask)
    pool = torch.nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)
    out = -pool(-m)
    out = out.view(*mask.shape).bool()
    return out


def build_soft_mouth_masks(mouth_mask: torch.Tensor,
                           erode_k: int = 3,
                           dilate_k: int = 5):
    """Return (mouth_core, mouth_overlap).

    mouth_core    : erode(mouth_mask, erode_k)    — exclusive interior
    mouth_overlap : dilate(mouth_mask, dilate_k)  — face+mouth shared band

    The shared band (mouth_overlap & ~mouth_core) is the soft seam — both
    branches receive GT supervision there but neither has an alpha
    hard-constraint, so they cooperate freely on that ring.
    """
    if mouth_mask.dtype != torch.bool:
        mouth_mask = mouth_mask.bool()
    mouth_core = erode_mask(mouth_mask, erode_k) if erode_k > 1 else mouth_mask.clone()
    mouth_overlap = dilate_mask(mouth_mask, dilate_k) if dilate_k > 1 else mouth_mask.clone()
    # safety: core ⊆ original ⊆ overlap
    mouth_core = mouth_core & mouth_mask
    mouth_overlap = mouth_overlap | mouth_mask
    return mouth_core, mouth_overlap
