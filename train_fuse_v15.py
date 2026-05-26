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
    # V23 fix: face_v3 calls add_densification_stats(viewspace_point_tensor)
    # which needs means2D.grad after backward. Without retain_grad, .grad is
    # None for non-leaf tensors and subscripting raises TypeError.
    try:
        means2D.retain_grad()
    except Exception:
        pass
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
             init_ckpt="macrontest/macron/chkpnt_fuse_v14_latest.pth",
             total_iters=10000):
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
        motion_net.load_state_dict(mp)
        gaussians_mouth.restore(gpm, opt)
        motion_net_mouth.load_state_dict(mpm)
        print(f"[v15] init from V14 ckpt: {init_ckpt}")
    else:
        raise RuntimeError(f"init_ckpt not found: {init_ckpt}")

    audio_dim = motion_net.audio_dim
    phoneme_head = PhonemeAuxHead(audio_dim=audio_dim, hidden=128, n_phonemes=392).cuda()
    albedo_head  = PerGaussianAlbedoMLP(audio_dim=audio_dim, au_dim=17,
                                         hidden=128, residual_scale=albedo_residual_scale).cuda()
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
    print(f"[v15] PhonemeAuxHead audio_dim={audio_dim}, KL weight={phoneme_w}")
    print(f"[v15] PerGaussianAlbedoMLP residual_scale={albedo_residual_scale}, lr={albedo_lr}")
    print(f"[v15] CrossAttnDriver in_dim={motion_net.in_dim}, d_model=128, heads=4, tokens=16")

    # Load V14 head weights if present (continues from V14, not from scratch).
    if v14_extras is not None:
        if 'phoneme_head' in v14_extras:
            phoneme_head.load_state_dict(v14_extras['phoneme_head'])
            print('[v15] loaded V14 phoneme_head')
        if 'albedo_head' in v14_extras:
            albedo_head.load_state_dict(v14_extras['albedo_head'])
            print('[v15] loaded V14 albedo_head')

    head_optimizer = torch.optim.Adam(
        list(phoneme_head.parameters()) +
        list(albedo_head.parameters()) +
        list(cross_attn_driver.parameters()),
        lr=albedo_lr, weight_decay=1e-6,
    )

    phoneme_path = os.path.join(dataset.source_path, "aud_phoneme.npy")
    phoneme_data = None
    if os.path.exists(phoneme_path):
        a = np.load(phoneme_path)
        p = a.mean(axis=1)
        p = p / (p.sum(axis=1, keepdims=True) + 1e-8)
        phoneme_data = torch.from_numpy(p).float().cuda()
        print(f"[v14] phoneme data loaded: T={phoneme_data.shape[0]}")

    lpips_criterion = lpips.LPIPS(net=lpips_net).eval().cuda()

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

        albedo_residual = albedo_head(gaussians.get_xyz.detach(),
                                       audio_emb_det.squeeze(0), au17)

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
        cross_attn_residual = cross_attn_driver(xyz_feat, audio_seq, au17)

        image, pkg_face, pkg_mouth = render_fuse_v15(
            viewpoint_cam, gaussians, motion_net,
            gaussians_mouth, motion_net_mouth, pipe, background,
            albedo_residual=albedo_residual,
            cross_attn_residual=cross_attn_residual,
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

        # ---- V4-style base loss (full-frame L1+SSIM+LPIPS, no head-only mask) ----
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * Lssim
        if iteration > lpips_start_iter:
            patch_size = random.randint(patch_min // 2, patch_max // 2) * 2
            Llpips = lpips_criterion(
                patchify(image[None, ...] * 2 - 1, patch_size),
                patchify(gt_image[None, ...] * 2 - 1, patch_size)
            ).mean()
            loss = loss + lpips_w * Llpips

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

        if not torch.isfinite(loss):
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians_mouth.optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            n_skip += 1
            continue

        loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            ema_phon = 0.4 * loss_phon_val + 0.6 * ema_phon
            ema_alb  = 0.4 * loss_alb_val + 0.6 * ema_alb
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "L": f"{ema_loss:.3f}",
                    "Lphon": f"{ema_phon:.3f}",
                    "alb": f"{ema_alb:.4f}",
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
                         'cross_attn_driver': cross_attn_driver.state_dict(),
                         'audio_dim': audio_dim,
                         'albedo_residual_scale': albedo_residual_scale})
                torch.save(ckpt, scene.model_path + f"/chkpnt_fuse_v15_{iteration}.pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_fuse_v15_latest.pth")
            if iteration < total_iters:
                gaussians.optimizer.step()
                gaussians_mouth.optimizer.step()
                head_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians_mouth.optimizer.zero_grad(set_to_none=True)
                head_optimizer.zero_grad(set_to_none=True)
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
                        default="macrontest/macron/chkpnt_fuse_v14_latest.pth")
    parser.add_argument("--total_iters", type=int, default=5000)  # finetune from V14
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
             total_iters=int(args.total_iters))
    print("\nv15 training complete.")
