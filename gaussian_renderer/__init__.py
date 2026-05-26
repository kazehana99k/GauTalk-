#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from utils.sh_utils import eval_sh

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    override_xyz=None,
    override_scaling=None,
    override_rotation=None,
    override_opacity=None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    base_xyz = override_xyz if override_xyz is not None else pc.get_xyz
    screenspace_points = torch.zeros_like(base_xyz, dtype=base_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=getattr(pipe, "antialiasing", True),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = base_xyz
    means2D = screenspace_points
    opacity = override_opacity if override_opacity is not None else pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = override_scaling if override_scaling is not None else pc.get_scaling
        rotations = override_rotation if override_rotation is not None else pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    outputs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Backward-compatible unpack. INRIA's updated rasterizer returns
    # (color, radii, invdepths) with NO alpha. Older builds return 4 values
    # including alpha. When alpha is missing we MUST NOT default to ones_like
    # (that breaks every face/mouth alpha-blend downstream — e.g. fuse compositing
    # turns mouth_pre into a full-frame opaque layer that hides face). Instead, do
    # ONE extra rasterization with white precomputed colors and the same opacities
    # to obtain a real alpha channel (this is exactly the standard 3DGS alpha-only
    # pass).
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        rendered_image, radii, rendered_depth = outputs
        # R-ENG-1 (gated): set env TG_ALPHA_GRAD=1 to enable grad-enabled alpha
        # pass. Default OFF — preflight audit shows zero effect at init, but
        # the pre-R-ENG-1 baseline at 35.2 dB (macron) relied on alpha being
        # treated as a constant in the composite. Off until separately verified.
        _alpha_grad = os.environ.get('TG_ALPHA_GRAD', '0') == '1'
        _alpha_ctx = (lambda: torch.enable_grad()) if _alpha_grad else torch.no_grad
        with _alpha_ctx():
            n_pts = means3D.shape[0]
            white = torch.ones((n_pts, 3), dtype=means3D.dtype, device=means3D.device)
            black_settings = GaussianRasterizationSettings(
                image_height=raster_settings.image_height,
                image_width=raster_settings.image_width,
                tanfovx=raster_settings.tanfovx,
                tanfovy=raster_settings.tanfovy,
                bg=torch.zeros(3, device=bg_color.device, dtype=bg_color.dtype),
                scale_modifier=raster_settings.scale_modifier,
                viewmatrix=raster_settings.viewmatrix,
                projmatrix=raster_settings.projmatrix,
                sh_degree=raster_settings.sh_degree,
                campos=raster_settings.campos,
                prefiltered=raster_settings.prefiltered,
                debug=raster_settings.debug,
                antialiasing=raster_settings.antialiasing,
            )
            black_rasterizer = GaussianRasterizer(raster_settings=black_settings)
            alpha_outputs = black_rasterizer(
                means3D=means3D,
                means2D=means2D.detach(),
                shs=None,
                colors_precomp=white,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            rendered_alpha = alpha_outputs[0].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = outputs

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "radii": radii}


def render_motion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=getattr(pipe, "antialiasing", True),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()

    # ind_code = motion_net.individual_codes[frame_idx if frame_idx is not None else viewpoint_camera.talking_dict["img_id"]]
    ind_code = None
    motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code) #  
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa'])
    opacity = pc.get_opacity

    cov3D_precomp = None
    # scales = pc.get_scaling
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])

    colors_precomp = None
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    outputs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        rendered_image, radii, rendered_depth = outputs
        # R-ENG-1 (gated): set env TG_ALPHA_GRAD=1 to enable alpha-pass grad
        _alpha_grad = os.environ.get('TG_ALPHA_GRAD', '0') == '1'
        _alpha_ctx = (lambda: torch.enable_grad()) if _alpha_grad else torch.no_grad
        with _alpha_ctx():
            n_pts = means3D.shape[0]
            white = torch.ones((n_pts, 3), dtype=means3D.dtype, device=means3D.device)
            black_settings = GaussianRasterizationSettings(
                image_height=raster_settings.image_height,
                image_width=raster_settings.image_width,
                tanfovx=raster_settings.tanfovx,
                tanfovy=raster_settings.tanfovy,
                bg=torch.zeros(3, device=bg_color.device, dtype=bg_color.dtype),
                scale_modifier=raster_settings.scale_modifier,
                viewmatrix=raster_settings.viewmatrix,
                projmatrix=raster_settings.projmatrix,
                sh_degree=raster_settings.sh_degree,
                campos=raster_settings.campos,
                prefiltered=raster_settings.prefiltered,
                debug=raster_settings.debug,
                antialiasing=raster_settings.antialiasing,
            )
            black_rasterizer = GaussianRasterizer(raster_settings=black_settings)
            alpha_outputs = black_rasterizer(
                means3D=means3D, means2D=means2D.detach(), shs=None,
                colors_precomp=white, opacities=opacity,
                scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
            )
            rendered_alpha = alpha_outputs[0].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = outputs

    # Attn
    rendered_attn = None
    if return_attn:
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        attn_outputs = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = attn_precomp,
            opacities = opacity.detach(),
            scales = scales.detach(),
            rotations = rotations.detach(),
            cov3D_precomp = cov3D_precomp)
        if isinstance(attn_outputs, (tuple, list)) and len(attn_outputs) == 3:
            rendered_attn, _, _ = attn_outputs
        else:
            rendered_attn, _, _, _ = attn_outputs


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds,
            'attn': rendered_attn}





def render_motion_mouth(viewpoint_camera, pc : GaussianModel, motion_net : MouthMotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False, landmark = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=getattr(pipe, "antialiasing", True),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()

    motion_preds = motion_net(pc.get_xyz, audio_feat, landmark=landmark)
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    colors_precomp = None
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    outputs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        rendered_image, radii, rendered_depth = outputs
        # R-ENG-1 (gated): set env TG_ALPHA_GRAD=1 to enable alpha-pass grad
        _alpha_grad = os.environ.get('TG_ALPHA_GRAD', '0') == '1'
        _alpha_ctx = (lambda: torch.enable_grad()) if _alpha_grad else torch.no_grad
        with _alpha_ctx():
            n_pts = means3D.shape[0]
            white = torch.ones((n_pts, 3), dtype=means3D.dtype, device=means3D.device)
            black_settings = GaussianRasterizationSettings(
                image_height=raster_settings.image_height,
                image_width=raster_settings.image_width,
                tanfovx=raster_settings.tanfovx,
                tanfovy=raster_settings.tanfovy,
                bg=torch.zeros(3, device=bg_color.device, dtype=bg_color.dtype),
                scale_modifier=raster_settings.scale_modifier,
                viewmatrix=raster_settings.viewmatrix,
                projmatrix=raster_settings.projmatrix,
                sh_degree=raster_settings.sh_degree,
                campos=raster_settings.campos,
                prefiltered=raster_settings.prefiltered,
                debug=raster_settings.debug,
                antialiasing=raster_settings.antialiasing,
            )
            black_rasterizer = GaussianRasterizer(raster_settings=black_settings)
            alpha_outputs = black_rasterizer(
                means3D=means3D, means2D=means2D.detach(), shs=None,
                colors_precomp=white, opacities=opacity,
                scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
            )
            rendered_alpha = alpha_outputs[0].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = outputs


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds}

