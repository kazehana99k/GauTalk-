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

from scene.cameras import Camera
import torch
import numpy as np
import os
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    
    if cam_info.image is not None:
        image_rgb = PILtoTorch(cam_info.image).type("torch.ByteTensor")
        gt_image = image_rgb[:3, ...]
    else:
        gt_image = None
        
    if cam_info.background is not None:
        background = PILtoTorch(cam_info.background)[:3, ...].type("torch.ByteTensor")
    else:
        background = None

    loaded_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, background=background, talking_dict=cam_info.talking_dict,
                  image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def loadCamOnTheFly(camera):
    image_path = camera.image_path
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    bg_img = PILtoTorch(np.array(Image.open(os.path.join("/".join(image_path.split("/")[:-2]), 'bc.jpg')).convert("RGB"))).to(camera.data_device)
    torso_img_path = image_path.replace("gt_imgs", "torso_imgs").replace("jpg", "png")
    torso_img = PILtoTorch(np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0).to(camera.data_device)
    bg = torso_img[:3] * torso_img[3:] / 255 + bg_img * (1.0 - torso_img[3:] / 255)

    teeth_mask_path = image_path.replace("gt_imgs", "teeth_mask").replace("jpg", "npy")
    teeth_mask = torch.as_tensor(np.load(teeth_mask_path)).to(camera.data_device)

    mask_path = image_path.replace("gt_imgs", "parsing").replace("jpg", "png")
    mask = PILtoTorch(np.array(Image.open(mask_path).convert("RGB")) * 1.0).to(camera.data_device)
    camera.talking_dict['hair_mask'] = (mask[0] < 1) * (mask[1] < 1) * (mask[2] < 1)

    # V28: opt-in lip/cavity refinement -- if TG_LIP_CAVITY=1 and the per-frame
    # lip_mask + cavity_mask .npy files exist, use them. Otherwise fall back to
    # the legacy V26 path (face_parsing_fine class==11 + teeth_mask).
    _use_lc = os.environ.get('TG_LIP_CAVITY', '0') == '1'
    _lip_p = image_path.replace("gt_imgs", "lip_mask").replace("jpg", "npy")
    _cav_p = image_path.replace("gt_imgs", "cavity_mask").replace("jpg", "npy")
    if _use_lc and os.path.exists(_lip_p) and os.path.exists(_cav_p):
        _lip = torch.as_tensor(np.load(_lip_p)).to(camera.data_device)
        _cav = torch.as_tensor(np.load(_cav_p)).to(camera.data_device)
        camera.talking_dict['lip_mask']    = _lip
        camera.talking_dict['cavity_mask'] = _cav
        camera.talking_dict['mouth_mask']  = _lip | _cav
        camera.talking_dict['face_mask']   = (mask[2] > 254) * (mask[0] == 0) * (mask[1] == 0) ^ camera.talking_dict['mouth_mask']
    else:
        camera.talking_dict['face_mask'] = (mask[2] > 254) * (mask[0] == 0) * (mask[1] == 0) ^ teeth_mask
        # V26: prefer fp==11 (mouth_inner) from face_parsing_fine over BiSeNet RGB=100
        # to eliminate lip-jitter contamination of the mouth supervision mask.
        _fp_p = image_path.replace("gt_imgs", "face_parsing_fine").replace("jpg", "npy")
        if os.path.exists(_fp_p):
            _fp = np.load(_fp_p)
            if _fp.ndim == 3 and _fp.shape[0] == 1: _fp = _fp[0]
            if _fp.ndim == 3 and _fp.shape[-1] == 1: _fp = _fp[..., 0]
            _fp_t = torch.as_tensor(_fp).to(camera.data_device)
            # R-FIXMASK (gated, default OFF): expand mouth_mask to include upper
            # lip (class 12) + lower lip (class 13). Diagnostic showed baseline
            # mouth_mask = class 11 only = ~508 px on apex frames, missing both
            # lips → mouth Gaussians never supervised on lip pixels → teeth area
            # smearing. Set TG_MOUTH_MASK_FULL=1 to include classes 11+12+13.
            _mask_full = os.environ.get('TG_MOUTH_MASK_FULL', '0') == '1'
            if _mask_full:
                camera.talking_dict['mouth_mask'] = ((_fp_t == 11) | (_fp_t == 12) | (_fp_t == 13) | teeth_mask.bool())
            else:
                camera.talking_dict['mouth_mask'] = ((_fp_t == 11) | teeth_mask.bool())
            # R-SUP-1: also populate fp_eye_mask on lazy-load path. Active reader
            # writes it under preload=True; train_face_v30 also hits the lazy path
            # when cam.original_image is None, so we must add it here too.
            if 'fp_eye_mask' not in camera.talking_dict:
                camera.talking_dict['fp_eye_mask'] = ((_fp_t == 4) | (_fp_t == 5))
        else:
            camera.talking_dict['mouth_mask'] = (mask[0] == 100) * (mask[1] == 100) * (mask[2] == 100) + teeth_mask

    camera.original_image = PILtoTorch(image).type("torch.ByteTensor").clamp(0, 255).to(camera.data_device)
    camera.background = bg.type("torch.ByteTensor").clamp(0, 255).to(camera.data_device)
    camera.image_width = camera.original_image.shape[2]
    camera.image_height = camera.original_image.shape[1]
    
    return camera
