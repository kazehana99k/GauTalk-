"""V25 perceptual losses — ArcFace identity + DINOv2 feature matching.

Two non-GAN losses that replace LPIPS for the face stage / fuse stage of the
3DGS talking-head pipeline:

1. ArcFaceIdentityLoss
   Cosine distance between the rendered and GT face's 512-D ArcFace-style
   embedding (InceptionResnetV1 pretrained on VGGFace2 — the closest open-
   weight equivalent to ArcFace).  Penalises identity drift / colour shift /
   excessive smoothing that wipes out face-discriminative features.

2. DinoV2PerceptualLoss
   L1 between rendered and GT DINOv2 patch embeddings (ViT-S/14, 384-D x 256
   patches at 224x224 input).  DINOv2 is trained self-supervised on natural
   images and preserves higher-frequency detail than VGG-LPIPS.

Both are eval-only (parameters frozen).  Forward = single inference pass per
loss per iter, ~10-25 ms on A6000.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_for_face(img_chw: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a 3xHxW or Bx3xHxW image to size×size for the face encoder."""
    if img_chw.dim() == 3:
        img_chw = img_chw.unsqueeze(0)
    return F.interpolate(img_chw, size=(size, size), mode="bilinear", align_corners=False)


class ArcFaceIdentityLoss(nn.Module):
    """Cos-distance between InceptionResnetV1(VGGFace2) face embeddings."""

    def __init__(self, device: torch.device = torch.device("cuda")):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        # facenet_pytorch expects [-1, 1] images at 160x160 (its model
        # internally does fixed_image_standardization).  We feed normalised
        # images already in [0, 1] so just rescale.
        self.input_size = 160

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        x = _resize_for_face(img, self.input_size)
        # facenet_pytorch normalisation: (x - 127.5) / 128 → for [0,1] in: x * 2 - 1
        x = x.clamp(0, 1) * 2 - 1
        return self.encoder(x)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # We can't use @torch.no_grad on pred (we need gradients for training)
        x_p = _resize_for_face(pred, self.input_size).clamp(0, 1) * 2 - 1
        e_p = self.encoder(x_p)
        with torch.no_grad():
            x_g = _resize_for_face(gt, self.input_size).clamp(0, 1) * 2 - 1
            e_g = self.encoder(x_g)
        # cosine distance = 1 - cos_sim
        return (1.0 - F.cosine_similarity(e_p, e_g, dim=-1)).mean()


class DinoV2PerceptualLoss(nn.Module):
    """L1 on DINOv2 ViT-S/14 patch features at 224x224."""

    def __init__(self,
                 ckpt_path: str = "/home/labliu/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
                 device: torch.device = torch.device("cuda")):
        super().__init__()
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                              pretrained=False, img_size=224, num_classes=0)
        sd = torch.load(ckpt_path, map_location="cpu")
        # Interpolate native 518x518 pos_embed (37x37 patches + 1 cls) down to
        # 224x224 (16x16 patches + 1 cls) so the model accepts 224 inputs.
        pe = sd["pos_embed"]
        cls, patch = pe[:, :1], pe[:, 1:]
        gs = int(patch.shape[1] ** 0.5)
        patch = patch.reshape(1, gs, gs, -1).permute(0, 3, 1, 2)
        target_gs = 224 // 14
        patch_r = F.interpolate(patch, size=(target_gs, target_gs),
                                 mode="bicubic", align_corners=False)
        patch_r = patch_r.permute(0, 2, 3, 1).reshape(1, target_gs * target_gs, -1)
        sd["pos_embed"] = torch.cat([cls, patch_r], dim=1)
        m.load_state_dict(sd, strict=False)
        self.encoder = m.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        # ImageNet normalisation that DINOv2 was trained with
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img.clamp(0, 1), size=(224, 224),
                            mode="bilinear", align_corners=False)
        return (img - self.mean) / self.std

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        x_p = self._prep(pred)
        f_p = self.encoder.forward_features(x_p)  # [1, 257, 384]
        with torch.no_grad():
            x_g = self._prep(gt)
            f_g = self.encoder.forward_features(x_g)
        return (f_p - f_g).abs().mean()
