#
# train_fuse_v15.py — V14 + (E) Per-Gaussian Cross-Attention Speech-Visual Driver.
#
# V15 = V14 (D + A) + GaussianCrossAttnDriver as a RESIDUAL on motion_net output.
# Each face Gaussian queries a tokenized speech context (8 audio tokens + 8 AU
# tokens with positional embeddings) and attends to the relevant tokens.
# Output is a small additive residual to d_xyz, d_scale, d_rot, d_opa.
#
# Why this is novel: prior 3DGS talking-head broadcasts a single global speech
# embedding to every Gaussian via multiplicative channel attention. We use
# structured cross-attention so different Gaussians can attend to different
# tokens (lip-corner → AU12 token; upper-lip → vowel audio token). Resolves
# audio→mouth ambiguity (silent open mouth vs voiced open mouth) by giving
# Gaussians access to BOTH audio and AU at token level.
#
# Init from V14 ckpt; CrossAttn final layer zero-init so iter 0 = V14 verbatim.
# Finetune 5k iter.
#
# (Original V14 docstring below)
# train_fuse_v14.py — V4 baseline + (D) PhonemeAuxHead + (A) Speech-Conditioned Albedo
#                      via colors_precomp (gradient-correct).
#
# IMPORTANT: V14 trains the FULL 10k iter from V4 ckpt — not a finetune of V10.
# All previous "V12/V13 finetune V10 5k" experiments drifted because re-fine-tuning
# already-finetuned features_dc with raster bg [0,1,0] for 5k more iterations
# pushed face Gaussian colors into a green-biased local minimum.  V14 follows the
# original V4 protocol (10k iter from face_50k+mouth_50k+ V3 fuse path) but adds
# the two innovations correctly so they cannot corrupt training.
#
# Innovations:
# D. PhonemeAuxHead — auxiliary loss with audio_emb DETACHED so backward never
#                     reaches motion_net (which is anyway frozen after iter 0).
# A. Speech-Conditioned Albedo MLP — RGB residual added DURING rasterization via
#                     colors_precomp pathway. Gradient flows correctly to albedo
#                     head via colors_precomp tensor. Does NOT touch face Gaussian
#                     features_dc directly — features_dc is only updated through
#                     the standard image-loss backward path.

import os, copy, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim
from utils.sh_utils import eval_sh
from gaussian_renderer import render_motion_mouth
from scene import Scene, GaussianModel, MotionNetwork, MouthMotionNetwork
from utils.general_utils import safe_state
import lpips, uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import loadCamOnTheFly
from models.perceptual_losses import ArcFaceIdentityLoss, DinoV2PerceptualLoss
from models.v12_heads import PhonemeAuxHead, PerGaussianAlbedoMLP, phoneme_aux_loss
from models.cross_attn_driver import GaussianCrossAttnDriver

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# Reimplementation of render_motion that injects per-Gaussian color residual
# via colors_precomp. Mirrors gaussian_renderer.render_motion but allows
# albedo_residual to be added to the precomputed RGB *with full gradient flow*
# back to albedo_head (and not into face Gaussian features_dc).
def render_motion_with_albedo(viewpoint_camera, pc, motion_net, pipe, bg_color,
                                albedo_residual: torch.Tensor,
                                cross_attn_residual=None):
    """V15: optionally apply cross_attn_residual on top of motion_net output.

    cross_attn_residual: dict with keys d_xyz/d_rot/d_opa/d_scale (per-Gaussian),
                         OR None to skip.
    """
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat   = viewpoint_camera.talking_dict["au_exp"].cuda()
    ind_code   = None  # mirrors gaussian_renderer.render_motion default

    motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code)

    # V15: add cross-attention residual to motion before applying.
    d_xyz   = motion_preds['d_xyz']
    d_scale = motion_preds['d_scale']
    d_rot   = motion_preds['d_rot']
    if cross_attn_residual is not None:
        d_xyz   = d_xyz   + cross_attn_residual['d_xyz']
        d_scale = d_scale + cross_attn_residual['d_scale']
        d_rot   = d_rot   + cross_attn_residual['d_rot']

    means3D  = pc.get_xyz + d_xyz
    means2D  = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda") + 0
    opacity  = pc.get_opacity        # NOTE: fuse-stage protocol — no d_opa applied
    scales   = pc.scaling_activation(pc._scaling + d_scale)
    rotations = pc.rotation_activation(pc._rotation + d_rot)

    # Compute SH → RGB in Python (so we can add albedo residual cleanly).
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    base_color = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # V14-A: add speech-conditioned albedo residual (already in [-scale, +scale]).
    final_color = (base_color + albedo_residual).clamp(0.0, 1.0)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=float(np.tan(viewpoint_camera.FoVx * 0.5)),
        tanfovy=float(np.tan(viewpoint_camera.FoVy * 0.5)),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        antialiasing=getattr(pipe, "antialiasing", True),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    outputs = rasterizer(
        means3D=means3D, means2D=means2D, shs=None, colors_precomp=final_color,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        rendered_image, radii, rendered_depth = outputs
        # CRITICAL: alpha pass MUST use BLACK bg [0,0,0] so the rendered image is
        # exactly α·white + 0·(1-α) = α. The original fallback used bg_color (green
        # in fuse training), then subtracted bg_mean — that is mathematically wrong
        # for green bg because the per-channel bg contribution differs from mean.
        # Using black bg avoids this entirely and gives a true alpha channel.
        with torch.no_grad():
            black_bg = torch.zeros(3, device=bg_color.device, dtype=bg_color.dtype)
            black_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=float(np.tan(viewpoint_camera.FoVx * 0.5)),
                tanfovy=float(np.tan(viewpoint_camera.FoVy * 0.5)),
                bg=black_bg,
                scale_modifier=1.0,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=pc.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=getattr(pipe, "debug", False),
                antialiasing=getattr(pipe, "antialiasing", True),
            )
            black_rasterizer = GaussianRasterizer(raster_settings=black_settings)
            white = torch.ones_like(final_color)
            alpha_outputs = black_rasterizer(
                means3D=means3D, means2D=means2D.detach(), shs=None,
                colors_precomp=white, opacities=opacity,
                scales=scales, rotations=rotations, cov3D_precomp=None,
            )
            rendered_alpha = alpha_outputs[0].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = outputs
    return {"render": rendered_image, "viewspace_points": means2D,
            "visibility_filter": radii > 0, "depth": rendered_depth,
            "alpha": rendered_alpha, "radii": radii, "motion": motion_preds}


def render_motion_mouth_v18(viewpoint_camera, pc, motion_net_mouth, pipe, bg_color,
                              cross_attn_residual=None, albedo_residual=None):
    """V18: render the mouth Gaussian set with an extra cross-attn residual on
    top of MouthMotionNetwork's d_xyz.  Mirrors render_motion_mouth from
    gaussian_renderer/__init__.py but adds d_rot/d_scale/d_opa from the
    cross-attn driver, giving the mouth branch the same speech+AU conditioning
    that face already has in V15/V17.
    """
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    motion_preds = motion_net_mouth(pc.get_xyz, audio_feat)

    d_xyz   = motion_preds['d_xyz']
    d_rot   = torch.zeros_like(pc._rotation)
    d_scale = torch.zeros_like(pc._scaling)
    d_opa   = torch.zeros_like(pc._opacity)
    if cross_attn_residual is not None:
        d_xyz   = d_xyz + cross_attn_residual['d_xyz']
        d_rot   = d_rot + cross_attn_residual['d_rot']
        d_scale = d_scale + cross_attn_residual['d_scale']
        d_opa   = d_opa + cross_attn_residual['d_opa']

    means3D   = pc.get_xyz + d_xyz
    means2D   = torch.zeros_like(pc.get_xyz, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except Exception:
        pass
    opacity   = pc.get_opacity
    scales    = pc.scaling_activation(pc._scaling + d_scale)
    rotations = pc.rotation_activation(pc._rotation + d_rot)

    # E2: audio/AU-conditioned per-Gaussian RGB residual on mouth.
    if albedo_residual is not None:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        base_color = torch.clamp_min(sh2rgb + 0.5, 0.0)
        final_color = (base_color + albedo_residual).clamp(0.0, 1.0)
        _shs_arg, _color_arg = None, final_color
    else:
        _shs_arg, _color_arg = pc.get_features, None

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=float(np.tan(viewpoint_camera.FoVx * 0.5)),
        tanfovy=float(np.tan(viewpoint_camera.FoVy * 0.5)),
        bg=bg_color, scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=getattr(pipe, "debug", False),
        antialiasing=getattr(pipe, "antialiasing", True),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    outputs = rasterizer(means3D=means3D, means2D=means2D, shs=_shs_arg,
                         colors_precomp=_color_arg, opacities=opacity,
                         scales=scales, rotations=rotations, cov3D_precomp=None)

    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        rendered_image, radii, rendered_depth = outputs
        with torch.no_grad():
            black_bg = torch.zeros(3, device=bg_color.device, dtype=bg_color.dtype)
            black_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=float(np.tan(viewpoint_camera.FoVx * 0.5)),
                tanfovy=float(np.tan(viewpoint_camera.FoVy * 0.5)),
                bg=black_bg, scale_modifier=1.0,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center,
                prefiltered=False, debug=getattr(pipe, "debug", False),
                antialiasing=getattr(pipe, "antialiasing", True),
            )
            black_raster = GaussianRasterizer(raster_settings=black_settings)
            white = torch.ones((means3D.shape[0], 3), dtype=means3D.dtype, device=means3D.device)
            alpha_outputs = black_raster(means3D=means3D, means2D=means2D.detach(), shs=None,
                                         colors_precomp=white, opacities=opacity,
                                         scales=scales, rotations=rotations, cov3D_precomp=None)
            rendered_alpha = alpha_outputs[0].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = outputs

    return {"render": rendered_image, "viewspace_points": means2D,
            "visibility_filter": radii > 0, "depth": rendered_depth,
            "alpha": rendered_alpha, "radii": radii, "motion": motion_preds}


def render_fuse_v18(viewpoint_cam, gaussians, motion_net,
                    gaussians_mouth, motion_net_mouth, pipe, background,
                    albedo_residual,
                    cross_attn_residual=None,
                    cross_attn_residual_mouth=None,
                    albedo_residual_mouth=None):
    """V18: V15 face composite + V18 mouth with its own cross-attn residual."""
    pkg_face = render_motion_with_albedo(viewpoint_cam, gaussians, motion_net, pipe,
                                           background, albedo_residual,
                                           cross_attn_residual=cross_attn_residual)
    pkg_mouth = render_motion_mouth_v18(viewpoint_cam, gaussians_mouth, motion_net_mouth,
                                          pipe, background,
                                          cross_attn_residual=cross_attn_residual_mouth,
                                          albedo_residual=albedo_residual_mouth)
    alpha = pkg_face["alpha"]
    alpha_mouth = pkg_mouth["alpha"]
    mouth_image = pkg_mouth["render"] - background[:, None, None] * (1.0 - alpha_mouth) \
                  + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha_mouth)
    image = pkg_face["render"] - background[:, None, None] * (1.0 - alpha) + mouth_image * (1.0 - alpha)
    return image, pkg_face, pkg_mouth


def render_fuse_v15(viewpoint_cam, gaussians, motion_net,
                    gaussians_mouth, motion_net_mouth, pipe, background,
                    albedo_residual, cross_attn_residual=None):
    """V4-style fuse compositing + V14 albedo + V15 cross-attn motion residual."""
    pkg_face = render_motion_with_albedo(viewpoint_cam, gaussians, motion_net, pipe,
                                           background, albedo_residual,
                                           cross_attn_residual=cross_attn_residual)
    pkg_mouth = render_motion_mouth(viewpoint_cam, gaussians_mouth, motion_net_mouth, pipe, background)
    alpha = pkg_face["alpha"]
    alpha_mouth = pkg_mouth["alpha"]
    mouth_image = pkg_mouth["render"] - background[:, None, None] * (1.0 - alpha_mouth) \
                  + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha_mouth)
    image = pkg_face["render"] - background[:, None, None] * (1.0 - alpha) + mouth_image * (1.0 - alpha)
    return image, pkg_face, pkg_mouth


def training(dataset, opt, pipe, debug_from,
             # base losses (V4 protocol)
             lpips_w=1.0, lpips_net="vgg", patch_min=64, patch_max=80,
             # V14 D / A
             phoneme_w=0.3,
             albedo_lr=1e-3, albedo_residual_scale=0.05,
             # full-protocol iters
             init_ckpt="macrontest/macron_v28/chkpnt_fuse_v30e_init.pth",
             total_iters=8000,
             au_window_T=8,
             aperture_w=0.2,
             freeze_face=False,
             detail_w=0.5,
             feat_anchor_w=0.0,
             pixel_start_iter=0,
             arc_w=0.0,
             dino_w=0.0,
             aperture_geom_w=0.0):
    opt.iterations = total_iters
    opt.densify_until_iter = 0
    bg_iter = opt.densify_until_iter
    saving_iterations = [opt.iterations]
    checkpoint_iterations = [opt.iterations]
    lpips_start_iter = 5000   # V4 schedule: LPIPS kicks in at half-way
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    setattr(dataset, "au_editor_mode", True)
    scene = Scene(dataset, gaussians)
    gaussians_mouth = GaussianModel(dataset.sh_degree)
    with torch.no_grad():
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda()
        motion_net = MotionNetwork(args=dataset).cuda()

    gaussians.training_setup(opt)
    gaussians_mouth.training_setup(opt)

    if init_ckpt and os.path.exists(init_ckpt):
        raw = torch.load(init_ckpt)
        gp, mp, gpm, mpm = raw[0], raw[1], raw[2], raw[3]
        # If V14 ckpt has 5th element (extras dict with phoneme/albedo heads),
        # load those into our fresh head instances so we start exactly at V14.
        v14_extras = raw[4] if len(raw) >= 5 and isinstance(raw[4], dict) else None
        gaussians.restore(gp, opt)
        motion_net.load_state_dict(mp, strict=False)
        gaussians_mouth.restore(gpm, opt)
        motion_net_mouth.load_state_dict(mpm, strict=False)
        print(f"[v15] init from V14 ckpt: {init_ckpt}")
    else:
        raise RuntimeError(f"init_ckpt not found: {init_ckpt}")

    audio_dim = motion_net.audio_dim
    audio_dim_mouth = motion_net_mouth.audio_dim
    phoneme_head = PhonemeAuxHead(audio_dim=audio_dim, hidden=128, n_phonemes=392).cuda()
    albedo_head  = PerGaussianAlbedoMLP(audio_dim=audio_dim, au_dim=17,
                                         hidden=128, residual_scale=albedo_residual_scale).cuda()
    # E2 (2026-05-20): mouth-side albedo carried over from mouth stage. Gated.
    _use_d2 = os.environ.get('TG_USE_D2', '0') == '1'
    albedo_head_mouth = PerGaussianAlbedoMLP(audio_dim=audio_dim_mouth, au_dim=17,
                                              hidden=128, residual_scale=0.35).cuda()
    if not _use_d2:
        print('[D2] disabled (TG_USE_D2=0)')
    # V15: per-Gaussian cross-attention residual driver.
    # xyz_feat_dim = motion_net.in_dim (tri-plane hashgrid output)
    cross_attn_driver = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net.in_dim,
        audio_seq_len=8,
        audio_dim=audio_dim,
        au_dim=17,
        d_model=128,
        n_heads=4,
        n_au_tokens=8,
        residual_scale_xyz=1e-3,
        residual_scale_scale=1e-3,
        residual_scale_rot=1e-3,
    ).cuda()
    # V30E: dual-head mouth cross-attn at fuse time, mirroring train_mouth_v30.
    # Lip head sees full AU17 with residual_scale 1e-3; cavity head sees only
    # AU25/AU26 (zeroed elsewhere) with residual_scale 2e-3 for stronger
    # teeth/tongue motion. cavity_idx (over mouth Gaussians) is computed once
    # at iter 0 and frozen; this matches the V30 mouth training distribution.
    cross_attn_driver_mouth = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net_mouth.in_dim,
        audio_seq_len=8, audio_dim=audio_dim_mouth, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=1e-3, residual_scale_scale=1e-3, residual_scale_rot=1e-3,
    ).cuda()
    cross_attn_driver_mouth_cavity = GaussianCrossAttnDriver(
        xyz_feat_dim=motion_net_mouth.in_dim,
        audio_seq_len=8, audio_dim=audio_dim_mouth, au_dim=17,
        d_model=128, n_heads=4, n_au_tokens=8,
        residual_scale_xyz=2e-3, residual_scale_scale=2e-3, residual_scale_rot=2e-3,
    ).cuda()
    # AU mask for cavity head: only AU25 (idx 14) + AU26 (idx 15).
    _CAVITY_AU_MASK = torch.zeros(17, device='cuda')
    _CAVITY_AU_MASK[14] = 1.0
    _CAVITY_AU_MASK[15] = 1.0
    # V18: aperture aux head — predict AU25/26 from the mouth audio embedding.
    # Trains motion_net_mouth.audio_net to be aperture-discriminative, which
    # then feeds the mouth cross-attn driver via shared audio_seq tokens.
    aperture_aux_head = nn.Sequential(
        nn.Linear(audio_dim_mouth, 32), nn.ReLU(inplace=True),
        nn.Linear(32, 2),
    ).cuda()
    print(f"[v15] PhonemeAuxHead audio_dim={audio_dim}, KL weight={phoneme_w}")
    print(f"[v15] PerGaussianAlbedoMLP residual_scale={albedo_residual_scale}, lr={albedo_lr}")
    print(f"[v15] CrossAttnDriver in_dim={motion_net.in_dim}, d_model=128, heads=4, tokens=16")
    print(f"[v18] CrossAttnDriverMouth in_dim={motion_net_mouth.in_dim}, audio_dim={audio_dim_mouth}, residual=2e-3")
    print(f"[v18] aperture aux head w={aperture_w} (predict AU25/AU26 from mouth audio_emb)")

    # Load V14/V15 head weights if present (continues from prior, not scratch).
    if v14_extras is not None:
        if 'phoneme_head' in v14_extras:
            phoneme_head.load_state_dict(v14_extras['phoneme_head'])
            print('[v17] loaded prior phoneme_head')
        if 'albedo_head' in v14_extras:
            albedo_head.load_state_dict(v14_extras['albedo_head'])
            print('[v17] loaded prior albedo_head')
        if 'albedo_head_mouth' in v14_extras:
            albedo_head_mouth.load_state_dict(v14_extras['albedo_head_mouth'])
            print('[E2] loaded prior albedo_head_mouth from mouth stage')
        if 'cross_attn_driver' in v14_extras:
            # V17: keep most layers from V15 cross-attn driver but skip the
            # newly-added au_temporal_proj (it's randomly init, will train).
            sd = v14_extras['cross_attn_driver']
            missing, unexpected = cross_attn_driver.load_state_dict(sd, strict=False)
            print(f"[v17] loaded prior cross_attn_driver (missing={missing}, unexpected={unexpected})")
        # V30E: prefer the V30-mouth dual-head pair if present.
        if 'cross_attn_driver_mouth_lip' in v14_extras:
            cross_attn_driver_mouth.load_state_dict(
                v14_extras['cross_attn_driver_mouth_lip'], strict=False)
            print('[v30e] loaded prior cross_attn_driver_mouth_lip')
        elif 'cross_attn_driver_mouth' in v14_extras:
            cross_attn_driver_mouth.load_state_dict(
                v14_extras['cross_attn_driver_mouth'], strict=False)
            print('[v30e] loaded prior cross_attn_driver_mouth (single head)')
        if 'cross_attn_driver_mouth_cavity' in v14_extras:
            cross_attn_driver_mouth_cavity.load_state_dict(
                v14_extras['cross_attn_driver_mouth_cavity'], strict=False)
            print('[v30e] loaded prior cross_attn_driver_mouth_cavity')
        elif 'cross_attn_driver_mouth' in v14_extras:
            # Bootstrap cavity head from single-head ckpt as warm-start.
            cross_attn_driver_mouth_cavity.load_state_dict(
                v14_extras['cross_attn_driver_mouth'], strict=False)
            print('[v30e] cavity head bootstrapped from single-head ckpt')
        if 'aperture_aux_head' in v14_extras:
            aperture_aux_head.load_state_dict(v14_extras['aperture_aux_head'])
            print('[v18] loaded prior aperture_aux_head')

    head_optimizer = torch.optim.Adam(
        list(phoneme_head.parameters()) +
        list(albedo_head.parameters()) +
        list(albedo_head_mouth.parameters()) +
        list(cross_attn_driver.parameters()) +
        list(cross_attn_driver_mouth.parameters()) +
        list(cross_attn_driver_mouth_cavity.parameters()) +
        list(aperture_aux_head.parameters()),
        lr=albedo_lr, weight_decay=1e-6,
    )
    # V18: unfreeze motion_net_mouth.audio_net so aperture aux can flow back.
    for p in motion_net_mouth.audio_net.parameters():
        p.requires_grad = True
    mouth_audio_optimizer = torch.optim.Adam(
        list(motion_net_mouth.audio_net.parameters()), lr=5e-4, weight_decay=1e-6,
    )

    # V22: snapshot features_dc for anti-drift anchor reg.  We freeze the
    # snapshot detached so backward only affects current `_features_dc`.
    feat_anchor_init = gaussians._features_dc.detach().clone()
    print(f"[v22] features_dc anchor snapshot @ iter 0: shape={feat_anchor_init.shape}, mean={feat_anchor_init.mean().item():.4f}")
    print(f"[v22] detail_w={detail_w} (Sobel L1), feat_anchor_w={feat_anchor_w}")

    # V30E: compute cavity_idx for mouth Gaussians by projecting them to the
    # train cam with the largest cavity_mask; frozen for the fuse run.
    cavity_idx_mouth = None
    try:
        import glob as _glob
        _cav_files = _glob.glob(os.path.join(dataset.source_path, 'cavity_mask', '*.npy'))
        best_fid = -1; best_cav = -1
        for _f in _cav_files:
            _fid = int(os.path.splitext(os.path.basename(_f))[0])
            _s = int(np.load(_f).sum())
            if _s > best_cav:
                best_cav = _s; best_fid = _fid
        if best_fid >= 0 and best_cav > 32:
            best_cam = None
            for _cams in (scene.getTrainCameras(), scene.getTestCameras()):
                for _cam in _cams:
                    if int(_cam.talking_dict.get('img_id', -1)) == best_fid:
                        best_cam = _cam; break
                if best_cam is not None: break
            if best_cam is not None:
                if best_cam.original_image is None:
                    from utils.camera_utils import loadCamOnTheFly as _lcotf
                    import copy as _cp
                    best_cam = _lcotf(_cp.deepcopy(best_cam))
                cav_t = torch.as_tensor(best_cam.talking_dict['cavity_mask']).cuda().bool()
                xyz = gaussians_mouth.get_xyz.detach()
                ones = torch.ones(xyz.shape[0], 1, device=xyz.device)
                homog = torch.cat([xyz, ones], dim=1)
                ndc = homog @ best_cam.full_proj_transform.cuda()
                ndc = ndc[:, :3] / ndc[:, 3:4].clamp_min(1e-6)
                px = ((ndc[:, 0] + 1) * best_cam.image_width  * 0.5).long()
                py = ((ndc[:, 1] + 1) * best_cam.image_height * 0.5).long()
                H_, W_ = cav_t.shape
                valid = (px >= 0) & (px < W_) & (py >= 0) & (py < H_)
                px_c = px.clamp(0, W_ - 1); py_c = py.clamp(0, H_ - 1)
                hit = cav_t[py_c, px_c]
                cavity_idx_mouth = valid & hit
                n_cav = int(cavity_idx_mouth.sum().item())
                print(f"[v30e] cavity_idx_mouth: {n_cav}/{xyz.shape[0]} mouth Gaussians "
                      f"(cam fid={best_fid}, cav_pixels={best_cav})")
    except Exception as _e:
        print(f"[v30e] cavity_idx compute failed ({_e}); falling back to single-head mouth forward")
        cavity_idx_mouth = None

    # V20: freeze face if requested — locks in the 'good' face state from V17
    # while letting the mouth path continue to refine.
    if freeze_face:
        for p in motion_net.parameters():
            p.requires_grad = False
        for p in cross_attn_driver.parameters():
            p.requires_grad = False
        for name in ['_xyz', '_features_dc', '_features_rest', '_scaling',
                     '_rotation', '_opacity']:
            getattr(gaussians, name).requires_grad = False
        print('[v20] face frozen: gaussians + motion_net + cross_attn_driver')

    phoneme_path = os.path.join(dataset.source_path, "aud_phoneme.npy")
    phoneme_data = None
    if os.path.exists(phoneme_path):
        a = np.load(phoneme_path)
        p = a.mean(axis=1)
        p = p / (p.sum(axis=1, keepdims=True) + 1e-8)
        phoneme_data = torch.from_numpy(p).float().cuda()
        print(f"[v14] phoneme data loaded: T={phoneme_data.shape[0]}")

    # V17: AU sliding window. Pre-load the full au_exp17 array once so each
    # iter can gather a [T, 17] window around the current frame instead of
    # passing only the single-frame AU vector.
    au_full_path = os.path.join(dataset.source_path, "au.csv")
    au_window_full = None
    try:
        import pandas as _pd
        _df = _pd.read_csv(au_full_path)
        _ids = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        _cols = []
        for _i in _ids:
            _k = ' AU' + str(_i).zfill(2) + '_r'
            _vals = _df[_k].values if _k in _df.columns else _df[_k.strip()].values
            if _i == 45:
                _vals = _vals.clip(0, 2)
            _cols.append(_vals[:, None])
        _au17_arr = np.concatenate(_cols, axis=-1).astype(np.float32)
        au_window_full = torch.from_numpy(_au17_arr).cuda()
        print(f"[v17] AU sliding window loaded: T={au_window_full.shape[0]}, dim=17, window={au_window_T}")
    except Exception as _e:
        print(f"[v17] AU window pre-load failed ({_e}); falling back to single-frame AU")

    lpips_criterion = lpips.LPIPS(net=lpips_net).eval().cuda()
    # V25: ArcFace identity + DINOv2 perceptual losses (replace LPIPS for paper).
    arc_criterion = ArcFaceIdentityLoss() if arc_w > 0 else None
    dino_criterion = DinoV2PerceptualLoss() if dino_w > 0 else None
    if arc_criterion is not None: print(f'[v25] ArcFaceIdentityLoss enabled w={arc_w}')
    if dino_criterion is not None: print(f'[v25] DinoV2PerceptualLoss enabled w={dino_w}')

    bg_color = [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss = 0.0; ema_phon = 0.0; ema_alb = 0.0
    n_phon = 0; n_skip = 0
    progress_bar = tqdm(range(first_iter, total_iters), ascii=True, dynamic_ncols=True, desc="v15 train")
    first_iter += 1

    for iteration in range(first_iter, total_iters + 1):
        iter_start.record()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if viewpoint_cam.original_image is None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        gaussians.update_learning_rate(iteration)

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask = face_mask + hair_mask + mouth_mask

        # V14: encode audio. DETACH for the phoneme aux loss path so backward
        # never reaches motion_net (avoids the V12/V13 grad-pollution mystery).
        with torch.no_grad():
            audio_raw = viewpoint_cam.talking_dict['auds'].cuda()
        audio_emb = motion_net.encode_audio(audio_raw)            # used for albedo (grad to motion_net OK while motion_net trainable)
        audio_emb_det = audio_emb.detach()                        # for phoneme aux loss (no backward to motion_net)
        au17 = viewpoint_cam.talking_dict.get('au_exp17', None)
        if au17 is None:
            au17 = torch.zeros(17, device=audio_emb.device)
        else:
            au17 = au17.to(audio_emb.device).float()

        # V17: gather AU temporal window [T, 17] around current frame.  The
        # albedo head still consumes the single-frame AU; only cross-attn
        # gets the sliding window.
        au17_window = au17
        if au_window_full is not None:
            _img_id = int(viewpoint_cam.talking_dict.get('img_id', -1))
            if 0 <= _img_id < au_window_full.shape[0]:
                _half = au_window_T // 2
                _lo = _img_id - _half
                _hi = _img_id + (au_window_T - _half)
                _l_pad = max(0, -_lo)
                _r_pad = max(0, _hi - au_window_full.shape[0])
                _lo_c = max(0, _lo)
                _hi_c = min(au_window_full.shape[0], _hi)
                _seq = au_window_full[_lo_c:_hi_c]
                if _l_pad:
                    _seq = torch.cat([_seq[:1].expand(_l_pad, -1), _seq], dim=0)
                if _r_pad:
                    _seq = torch.cat([_seq, _seq[-1:].expand(_r_pad, -1)], dim=0)
                au17_window = _seq.float()

        albedo_residual = albedo_head(gaussians.get_xyz.detach(),
                                       audio_emb_det.squeeze(0), au17)
        # E2 (2026-05-20): mouth albedo residual. Gated.
        if _use_d2:
            with torch.no_grad():
                audio_emb_mouth_det = motion_net_mouth.encode_audio(audio_raw)
            albedo_residual_mouth = albedo_head_mouth(gaussians_mouth.get_xyz.detach(),
                                                       audio_emb_mouth_det.squeeze(0), au17)
        else:
            albedo_residual_mouth = None

        # V15: cross-attention residual on motion. We need a per-Gaussian xyz feat;
        # reuse motion_net.encode_x (tri-plane hashgrid) and DETACH so backward only
        # touches cross_attn_driver (motion_net is frozen anyway after iter 0).
        with torch.no_grad():
            xyz_feat = motion_net.encode_x(gaussians.get_xyz, bound=motion_net.bound)
        # audio sequence for cross-attn: encode the raw 8-frame audio per-frame.
        with torch.no_grad():
            # motion_net.audio_net produces [B, audio_dim] from raw [B, 1024, 2] / etc.
            audio_seq = motion_net.audio_net(audio_raw)            # [8, audio_dim] approximately
            if audio_seq.dim() == 1:
                audio_seq = audio_seq.unsqueeze(0)
        # V17: feed AU sliding window (au17_window: [T, 17]) instead of single
        # frame so the transformer can attend over AU temporal evolution,
        # symmetric to the audio 8-frame window.
        cross_attn_residual = cross_attn_driver(xyz_feat, audio_seq, au17_window)

        # V18: mouth cross-attn driver. xyz_feat from motion_net_mouth's hashgrid
        # (detached), audio_seq from motion_net_mouth.audio_net (NOT detached so
        # the aperture aux loss can flow back into the mouth audio encoder).
        with torch.no_grad():
            xyz_feat_mouth = motion_net_mouth.encode_x(
                gaussians_mouth.get_xyz, bound=motion_net_mouth.bound)
        audio_seq_mouth = motion_net_mouth.audio_net(audio_raw)
        if audio_seq_mouth.dim() == 1:
            audio_seq_mouth = audio_seq_mouth.unsqueeze(0)
        # V30E: dual-head mouth cross-attn split, mirroring train_mouth_v30.
        if cavity_idx_mouth is not None and int(cavity_idx_mouth.sum().item()) > 8 \
                and int((~cavity_idx_mouth).sum().item()) > 8 \
                and cavity_idx_mouth.shape[0] == xyz_feat_mouth.shape[0]:
            xf_lip = xyz_feat_mouth[~cavity_idx_mouth]
            xf_cav = xyz_feat_mouth[cavity_idx_mouth]
            au_lip_in = au17_window
            au_cav_in = au17_window * _CAVITY_AU_MASK if au17_window.dim() == 1 \
                else au17_window * _CAVITY_AU_MASK[None, :]
            res_lip = cross_attn_driver_mouth(xf_lip, audio_seq_mouth, au_lip_in)
            res_cav = cross_attn_driver_mouth_cavity(xf_cav, audio_seq_mouth, au_cav_in)
            N = xyz_feat_mouth.shape[0]
            cross_attn_residual_mouth = {}
            for k in res_lip:
                if isinstance(res_lip[k], torch.Tensor) and res_lip[k].dim() >= 1 \
                        and res_lip[k].shape[0] == xf_lip.shape[0]:
                    merged = torch.zeros((N,) + tuple(res_lip[k].shape[1:]),
                                          device=res_lip[k].device, dtype=res_lip[k].dtype)
                    merged[~cavity_idx_mouth] = res_lip[k]
                    merged[cavity_idx_mouth]  = res_cav[k]
                    cross_attn_residual_mouth[k] = merged
                else:
                    cross_attn_residual_mouth[k] = res_lip[k]
        else:
            cross_attn_residual_mouth = cross_attn_driver_mouth(
                xyz_feat_mouth, audio_seq_mouth, au17_window)

        image, pkg_face, pkg_mouth = render_fuse_v18(
            viewpoint_cam, gaussians, motion_net,
            gaussians_mouth, motion_net_mouth, pipe, background,
            albedo_residual=albedo_residual,
            cross_attn_residual=cross_attn_residual,
            cross_attn_residual_mouth=cross_attn_residual_mouth,
            albedo_residual_mouth=albedo_residual_mouth,
        )
        gt_image = viewpoint_cam.original_image.cuda() / 255.0

        # V4 protocol: freeze motion_net & rigid Gaussian state after warm-up bg_iter=0.
        if iteration > bg_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
            for param in motion_net_mouth.parameters():
                param.requires_grad = False
            gaussians._xyz.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False
            gaussians_mouth._xyz.requires_grad = False
            gaussians_mouth._opacity.requires_grad = False
            gaussians_mouth._scaling.requires_grad = False
            gaussians_mouth._rotation.requires_grad = False
            # E2 (2026-05-20): freeze mouth SH so albedo_head_mouth isn't undone.
            if _use_d2:
                gaussians_mouth._features_dc.requires_grad = False
                gaussians_mouth._features_rest.requires_grad = False

        # ---- V4-style base loss (full-frame L1+SSIM+LPIPS, no head-only mask) ----
        # V24 staged schedule: gate the smoothing-prone L1+SSIM until
        # pixel_start_iter so the model first develops high-freq detail under
        # LPIPS+Sobel pressure, then locks in pixel accuracy in the final phase.
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        if iteration >= pixel_start_iter:
            loss = Ll1 + opt.lambda_dssim * Lssim
        else:
            # zero-tensor placeholder so backward graph still works downstream
            loss = 0.0 * Ll1
        if iteration > lpips_start_iter:
            patch_size = random.randint(patch_min // 2, patch_max // 2) * 2
            Llpips = lpips_criterion(
                patchify(image[None, ...] * 2 - 1, patch_size),
                patchify(gt_image[None, ...] * 2 - 1, patch_size)
            ).mean()
            loss = loss + lpips_w * Llpips

        # V25: ArcFace identity + DINOv2 perceptual losses.  Active from iter 0
        # (cheap and beneficial early), independent of pixel_start_iter.
        loss_arc_val = 0.0
        loss_dino_val = 0.0
        if arc_criterion is not None:
            loss_arc = arc_criterion(image, gt_image)
            loss = loss + arc_w * loss_arc
            loss_arc_val = float(loss_arc.detach().item())
        if dino_criterion is not None:
            loss_dino = dino_criterion(image, gt_image)
            loss = loss + dino_w * loss_dino
            loss_dino_val = float(loss_dino.detach().item())

        # V22: Sobel gradient loss — penalises any high-frequency content the
        # model fails to reproduce.  Forces face Gaussians to keep skin pores,
        # hair edges, eyebrow strands etc. instead of smoothing them away to
        # game LPIPS.  Computed on the full image with a fixed-weight Sobel.
        if detail_w > 0:
            with torch.no_grad():
                sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                        device='cuda').view(1, 1, 3, 3)
                sobel_y = sobel_x.transpose(2, 3)
                kx = sobel_x.expand(3, 1, 3, 3).contiguous()
                ky = sobel_y.expand(3, 1, 3, 3).contiguous()
            pred_gx = F.conv2d(image[None], kx, padding=1, groups=3)
            pred_gy = F.conv2d(image[None], ky, padding=1, groups=3)
            gt_gx   = F.conv2d(gt_image[None], kx, padding=1, groups=3)
            gt_gy   = F.conv2d(gt_image[None], ky, padding=1, groups=3)
            loss_detail = (pred_gx - gt_gx).abs().mean() + (pred_gy - gt_gy).abs().mean()
            loss = loss + detail_w * loss_detail
            loss_detail_val = float(loss_detail.detach().item())
        else:
            loss_detail_val = 0.0

        # V22: features_dc anchor.  Snapshots the SH DC at iter 0, then on
        # every iter penalises drift from that snapshot.  Stops the slow
        # cumulative averaging of skin tones over multi-stage fuse training.
        if feat_anchor_w > 0 and feat_anchor_init is not None:
            loss = loss + feat_anchor_w * (gaussians._features_dc - feat_anchor_init).abs().mean()

        # ---- V14-D: PhonemeAuxHead (audio_emb detached) ----
        loss_phon_val = 0.0
        if phoneme_data is not None and phoneme_w > 0:
            cur_fid = int(viewpoint_cam.talking_dict.get('img_id', -1))
            if 0 <= cur_fid < phoneme_data.shape[0]:
                target = phoneme_data[cur_fid]
                log_probs = phoneme_head(audio_emb_det.squeeze(0))
                loss_phon = phoneme_aux_loss(log_probs, target)
                loss = loss + phoneme_w * loss_phon
                loss_phon_val = float(loss_phon.detach().item())
                n_phon += 1

        # V14-A: small L1 reg on albedo residual to prevent runaway.
        loss = loss + 0.05 * albedo_residual.abs().mean()
        loss_alb_val = float(albedo_residual.abs().mean().detach().item())

        # V18: aperture aux. mouth audio_emb (NOT detached) → AU25/AU26 prediction.
        # AU exp17 indices: [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        #                                                                14    15
        loss_aper_val = 0.0
        if aperture_w > 0 and au17 is not None and au17.numel() >= 17:
            mouth_audio_emb = audio_seq_mouth.mean(dim=0) if audio_seq_mouth.dim() == 2 else audio_seq_mouth
            au25_pred_pair = aperture_aux_head(mouth_audio_emb)              # [2]
            au25_gt_pair = au17[14:16].detach()                                # AU25, AU26
            loss_aper = (au25_pred_pair - au25_gt_pair).abs().mean()
            loss = loss + aperture_w * loss_aper
            loss_aper_val = float(loss_aper.detach().item())

        # V26: lip aperture GEOMETRIC loss — directly reward the face/mouth
        # Gaussians' projected vertical extent inside lips_rect to scale with
        # AU25.  Forces "open mouth audio → wider Gaussian spread" rather than
        # letting the L1+SSIM-favoured closed-mouth mean pose dominate.
        loss_lap_val = 0.0
        if aperture_geom_w > 0 and au17 is not None and au17.numel() >= 17:
            au25_now = float(au17[14].detach().item())
            try:
                lr = viewpoint_cam.talking_dict.get('lips_rect', None)
                if lr is not None and len(lr) == 4:
                    xmin, xmax, ymin, ymax = [int(v) for v in lr]
                    # Project face Gaussian centres (post-motion) to image plane.
                    means3D_face = gaussians.get_xyz + cross_attn_residual['d_xyz']
                    pad = torch.ones(means3D_face.shape[0], 1, device='cuda')
                    homog = torch.cat([means3D_face, pad], dim=1)
                    proj = homog @ viewpoint_cam.full_proj_transform
                    proj = proj[:, :2] / proj[:, 2:3].clamp_min(1e-6)
                    px = (proj[:, 0] + 1) * viewpoint_cam.image_width  * 0.5
                    py = (proj[:, 1] + 1) * viewpoint_cam.image_height * 0.5
                    inside_lr = (py > xmin) & (py < xmax) & (px > ymin) & (px < ymax)
                    if inside_lr.sum() > 8:
                        py_in = py[inside_lr]
                        # robust spread: 90th - 10th percentile
                        py_lo = torch.quantile(py_in, 0.10)
                        py_hi = torch.quantile(py_in, 0.90)
                        spread_pred = py_hi - py_lo
                        # target: scale AU25 to a reasonable lip-height (in px).
                        # lips_rect is square-padded with side ~ 60-100 px on macron.
                        lr_h = float(xmax - xmin)
                        spread_target = (au25_now / 3.0) * lr_h * 0.45  # ~45% of lr_h at full open
                        loss_lap = torch.abs(spread_pred - spread_target) / max(1.0, lr_h)
                        loss = loss + aperture_geom_w * loss_lap
                        loss_lap_val = float(loss_lap.detach().item())
            except Exception:
                pass

        if not torch.isfinite(loss):
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians_mouth.optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            mouth_audio_optimizer.zero_grad(set_to_none=True)
            n_skip += 1
            continue

        loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            ema_phon = 0.4 * loss_phon_val + 0.6 * ema_phon
            ema_alb  = 0.4 * loss_alb_val + 0.6 * ema_alb
            if 'ema_aper' not in dir():
                pass
            try:
                ema_aper = 0.4 * loss_aper_val + 0.6 * ema_aper  # type: ignore
            except NameError:
                ema_aper = loss_aper_val
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "L": f"{ema_loss:.3f}",
                    "Lphon": f"{ema_phon:.3f}",
                    "Laper": f"{ema_aper:.3f}",
                    "skip": n_skip,
                })
                progress_bar.update(10)
            if iteration == total_iters:
                progress_bar.close()
            if iteration in saving_iterations:
                scene.save(iteration)
            if iteration in checkpoint_iterations:
                ckpt = (gaussians.capture(), motion_net.state_dict(),
                        gaussians_mouth.capture(), motion_net_mouth.state_dict(),
                        {'phoneme_head': phoneme_head.state_dict(),
                         'albedo_head': albedo_head.state_dict(),
                         'albedo_head_mouth': albedo_head_mouth.state_dict(),
                         'cross_attn_driver': cross_attn_driver.state_dict(),
                         'cross_attn_driver_mouth': cross_attn_driver_mouth.state_dict(),
                         'cross_attn_driver_mouth_lip': cross_attn_driver_mouth.state_dict(),
                         'cross_attn_driver_mouth_cavity': cross_attn_driver_mouth_cavity.state_dict(),
                         'aperture_aux_head': aperture_aux_head.state_dict(),
                         'audio_dim': audio_dim,
                         'audio_dim_mouth': audio_dim_mouth,
                         'albedo_residual_scale': albedo_residual_scale,
                         'albedo_residual_scale_mouth': 0.35})
                # V30E: force tag, dual-head mouth fuse fork.
                _tag = "v30e"
                torch.save(ckpt, scene.model_path + f"/chkpnt_fuse_{_tag}_{iteration}.pth")
                torch.save(ckpt, scene.model_path + f"/chkpnt_fuse_{_tag}_latest.pth")
            if iteration < total_iters:
                gaussians.optimizer.step()
                gaussians_mouth.optimizer.step()
                head_optimizer.step()
                mouth_audio_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians_mouth.optimizer.zero_grad(set_to_none=True)
                head_optimizer.zero_grad(set_to_none=True)
                mouth_audio_optimizer.zero_grad(set_to_none=True)
    print(f"[v14] hits: phon={n_phon}, skip={n_skip}")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


if __name__ == "__main__":
    parser = ArgumentParser(description="v15 training (V14 + cross-attn driver)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--lpips_w", type=float, default=1.0)
    parser.add_argument("--lpips_net", type=str, default="vgg")
    parser.add_argument("--patch_min", type=int, default=64)
    parser.add_argument("--patch_max", type=int, default=80)
    parser.add_argument("--phoneme_w", type=float, default=0.3)
    parser.add_argument("--albedo_lr", type=float, default=1e-3)
    parser.add_argument("--albedo_residual_scale", type=float, default=0.05)
    parser.add_argument("--init_ckpt", type=str,
                        default="macrontest/macron_v28/chkpnt_fuse_v30e_init.pth")
    parser.add_argument("--total_iters", type=int, default=8000)  # V28: from face+mouth init, longer than V17 finetune
    parser.add_argument("--au_window_T", type=int, default=8)
    parser.add_argument("--aperture_w", type=float, default=0.2)
    parser.add_argument("--freeze_face", action="store_true",
                        help="V20: freeze face Gaussians + motion_net + face cross-attn")
    parser.add_argument("--detail_w", type=float, default=0.5,
                        help="V22: Sobel gradient L1 weight (anti-smoothing, default 0.5)")
    parser.add_argument("--feat_anchor_w", type=float, default=0.0,
                        help="V22: SH features_dc anchor reg weight (default off)")
    parser.add_argument("--pixel_start_iter", type=int, default=0,
                        help="V24: iter at which L1+SSIM are added (deferred to prevent over-smoothing)")
    parser.add_argument("--arc_w", type=float, default=0.0,
                        help="V25: ArcFace identity loss weight (cos-distance, ~0.1)")
    parser.add_argument("--dino_w", type=float, default=0.0,
                        help="V25: DINOv2 patch feature L1 weight (~0.5)")
    parser.add_argument("--aperture_geom_w", type=float, default=0.0,
                        help="V26: face Gaussian vertical-spread vs AU25 geometric loss weight")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.debug_from,
             lpips_w=float(args.lpips_w), lpips_net=str(args.lpips_net),
             patch_min=int(args.patch_min), patch_max=int(args.patch_max),
             phoneme_w=float(args.phoneme_w),
             albedo_lr=float(args.albedo_lr),
             albedo_residual_scale=float(args.albedo_residual_scale),
             init_ckpt=str(args.init_ckpt),
             total_iters=int(args.total_iters),
             au_window_T=int(args.au_window_T),
             aperture_w=float(args.aperture_w),
             freeze_face=bool(args.freeze_face),
             detail_w=float(args.detail_w),
             feat_anchor_w=float(args.feat_anchor_w),
             pixel_start_iter=int(args.pixel_start_iter),
             arc_w=float(args.arc_w),
             dino_w=float(args.dino_w),
             aperture_geom_w=float(args.aperture_geom_w))
    print("\nv18 training complete.")
