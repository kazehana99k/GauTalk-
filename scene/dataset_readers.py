

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
import sys
import torch
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd
from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features
from utils.lip_cavity_masks import load_face_mouth_masks  # V28
from scene.gaussian_model import BasicPointCloud
from PIL import Image, ImageDraw

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _poly_mask(h, w, pts_xy):
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    if len(pts_xy) >= 3:
        draw.polygon(pts_xy, outline=1, fill=1)
    return (np.array(mask_img) > 0)

def _roi_from_lms(h, w, lms):
    def xy(idx_list):
        return [(int(lms[i, 1]), int(lms[i, 0])) for i in idx_list]

    jaw = list(range(0, 17))
    rbrow = list(range(17, 22))
    lbrow = list(range(22, 27))
    nose = list(range(27, 36))
    reye = list(range(36, 42))
    leye = list(range(42, 48))
    mouth_outer = list(range(48, 60))

    m_brows = _poly_mask(h, w, xy(rbrow) + xy(lbrow)[::-1])
    m_eyes = _poly_mask(h, w, xy(reye)) | _poly_mask(h, w, xy(leye))
    m_nose = _poly_mask(h, w, xy(nose))
    m_mouth = _poly_mask(h, w, xy(mouth_outer))

    by = np.array([lms[i, 0] for i in rbrow + lbrow], dtype=np.float32)
    ey = np.array([lms[i, 0] for i in reye + leye], dtype=np.float32)
    # lift = int(max(2, np.median(by - ey) * 0.6))
    brow_mid = np.median(by)
    eye_mid = np.median(ey)
    lift = int(max(2, (brow_mid - eye_mid) * 0.6))

    def shift_up(pts):
        return [(x, max(0, y - lift)) for (x, y) in pts]

    brow_poly = xy(rbrow) + xy(lbrow)[::-1]
    m_forehead = _poly_mask(h, w, shift_up(brow_poly))
    m_face = _poly_mask(h, w, xy(jaw) + shift_up(xy(jaw))[::-1])
    m_cheeks = m_face & (~(m_eyes | m_nose | m_mouth | m_forehead))

    return {
        "roi_brows": m_brows,
        "roi_eyes": m_eyes,
        "roi_nose": m_nose,
        "roi_mouth": m_mouth,
        "roi_forehead": m_forehead,
        "roi_cheeks": m_cheeks,
    }

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        frames = contents["frames"]

        # Hard truncate BEFORE building CameraInfo objects
        try:
            if transformsfile.startswith("transforms_train"):
                _max_n = int(os.environ.get("TG_MAX_TRAIN_FRAMES", "0") or "0")
                if _max_n > 0 and len(frames) > _max_n:
                    frames = frames[:_max_n]
            if transformsfile.startswith("transforms_val"):
                _max_n = int(os.environ.get("TG_MAX_TEST_FRAMES", "0") or "0")
                if _max_n > 0 and len(frames) > _max_n:
                    frames = frames[:_max_n]
        except Exception:
            pass

        bg_img = None
        if preload:
            bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert('RGB'))

        # audio
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features).float().permute(0, 2, 1)
        auds = aud_features

        au_info = pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(' ', '') in df.columns: return df[key.replace(' ', '')].values
            return np.zeros(len(df), dtype=np.float32)

        au_blink = get_au_col(au_info, ' AU45_r')
        au25 = get_au_col(au_info, ' AU25_r')
        au25 = np.clip(au25, 0, np.percentile(au25, 95))
        au25_25, au25_50, au25_75, au25_100 = np.percentile(au25, 25), np.percentile(au25, 50), np.percentile(au25, 75), au25.max()

        au_exp = []
        for i in [1,4,5,6,7,45,25]:  # R-FACE-AU25: added AU25 (7-dim)
            _key = ' AU' + str(i).zfill(2) + '_r'
            au_exp_t = get_au_col(au_info, _key)
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
            au_exp.append(au_exp_t[:, None])
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

        aud_ids_17 = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        au_exp17 = []
        for i in aud_ids_17:
            _key = ' AU' + str(i).zfill(2) + '_r'
            vals = get_au_col(au_info, _key)
            if i == 45:
                vals = np.clip(vals, 0, 2)
            au_exp17.append(vals[:, None])
        au_exp17 = np.concatenate(au_exp17, axis=-1, dtype=np.float32)

        # pose meta (optional)
        pose_meta_dir = os.path.join(path, 'pose_meta')
        pose_meta_cache = {}
        def load_pose_meta(img_id):
            if not os.path.isdir(pose_meta_dir):
                return None
            candidates = []
            try:
                iid = int(img_id)
                candidates.append(f"{iid:05d}.json")
                candidates.append(f"{iid}.json")
            except Exception:
                candidates.append(f"{img_id}.json")
            for name in candidates:
                meta_path = os.path.join(pose_meta_dir, name)
                if os.path.exists(meta_path):
                    if name not in pose_meta_cache:
                        with open(meta_path, 'r') as f:
                            pose_meta_cache[name] = json.load(f)
                    return pose_meta_cache[name]
            return None

        # preload=False: get w,h once
        fixed_w = fixed_h = None
        if not preload and len(frames) > 0:
            first_id = frames[0]['img_id']
            first_path = os.path.join(path, 'gt_imgs', str(first_id) + extension)
            with Image.open(first_path) as im:
                fixed_w, fixed_h = im.size[0], im.size[1]

        # V17/landmarks: per-frame .lms files give 68 landmarks. The original
        # TG reader (kept around in earlier copies of this file) uses these to
        # build mouth_bound / lips_rect / lhalf_rect / brow_rect — fields that
        # train_face / train_mouth assume exist on every camera.  This active
        # reader was previously slimmed for the AU-editor path and dropped
        # them; we reinstate the pre-pass here so train_face_v2 / train_mouth_v2
        # can rely on talking_dict['mouth_bound'] etc.
        ldmks_lips_arr = ldmks_mouth_arr = ldmks_lhalf_arr = ldmks_brow_arr = None
        mouth_lb = mouth_ub = 0
        try:
            _ldmks_lips_l, _ldmks_mouth_l, _ldmks_lhalf_l, _ldmks_brow_l = [], [], [], []
            for _frm in frames:
                _lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(_frm['img_id']) + '.lms'))
                _lips = slice(48, 60); _mouth = slice(60, 68); _brow = slice(17, 27)
                _xmin, _xmax = int(_lms[_lips, 1].min()), int(_lms[_lips, 1].max())
                _ymin, _ymax = int(_lms[_lips, 0].min()), int(_lms[_lips, 0].max())
                _ldmks_lips_l.append([_xmin, _xmax, _ymin, _ymax])
                _ldmks_mouth_l.append([int(_lms[_mouth, 1].min()), int(_lms[_mouth, 1].max())])
                _lh_xmin = int(_lms[31:36, 1].min()); _lh_xmax = int(_lms[:, 1].max())
                _all_ymin, _all_ymax = int(_lms[:, 0].min()), int(_lms[:, 0].max())
                _ldmks_lhalf_l.append([_lh_xmin, _lh_xmax, _all_ymin, _all_ymax])
                _bxmin, _bxmax = int(_lms[_brow, 1].min()), int(_lms[_brow, 1].max())
                _bymin, _bymax = int(_lms[_brow, 0].min()), int(_lms[_brow, 0].max())
                _pad_y = max(2, (_bymax - _bymin) // 4); _pad_x = max(2, (_bxmax - _bxmin) // 6)
                _ldmks_brow_l.append([_bxmin - _pad_x, _bxmax + _pad_x, _bymin - _pad_y, _bymax + _pad_y])
            ldmks_lips_arr = np.array(_ldmks_lips_l)
            ldmks_mouth_arr = np.array(_ldmks_mouth_l)
            ldmks_lhalf_arr = np.array(_ldmks_lhalf_l)
            ldmks_brow_arr = np.array(_ldmks_brow_l)
            mouth_lb = (ldmks_mouth_arr[:, 1] - ldmks_mouth_arr[:, 0]).min()
            mouth_ub = (ldmks_mouth_arr[:, 1] - ldmks_mouth_arr[:, 0]).max()
        except Exception as _e:
            # If .lms files are missing, leave landmarks unset — the consumer
            # will surface a clearer error than reading an unloaded array.
            print(f"[reader] landmark pre-pass skipped: {_e}")

        for idx, frame in tqdm(enumerate(frames)):
            img_id = frame['img_id']
            cam_name = os.path.join('gt_imgs', str(img_id) + extension)
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            c2w = np.array(frame['transform_matrix'])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]

            if preload:
                image_pil = Image.open(image_path)
                w, h = image_pil.size[0], image_pil.size[1]
                image = np.array(image_pil.convert('RGB'))
            else:
                image = None
                w, h = fixed_w, fixed_h

            bg = None
            if preload:
                torso_img_path = os.path.join(path, 'torso_imgs', str(img_id) + '.png')
                torso_img = np.array(Image.open(torso_img_path).convert('RGBA')) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)

            talking_dict = {'img_id': img_id}
            if preload:
                # 保持 TalkingGaussian 原始行为：在 preload 模式下预先构造 face/hair/mouth mask
                teeth_mask_path = os.path.join(path, 'teeth_mask', str(img_id) + '.npy')
                teeth_mask = np.load(teeth_mask_path)

                mask_path = os.path.join(path, 'parsing', str(img_id) + '.png')
                mask = np.array(Image.open(mask_path).convert('RGB')) * 1.0
                talking_dict['face_mask'] = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                # V26: prefer face_parsing_fine label 11 (mouth_inner) over the
                # coarse BiSeNet RGB=100 region — the latter is jittery and on
                # open-mouth frames it sometimes swallows lip-inner pixels,
                # contaminating the mouth Gaussians' supervision.  fp==11 is
                # strictly inner-mouth in the fine parser; falls back to the
                # original BiSeNet rule when face_parsing_fine is missing.
                _fp_p = os.path.join(path, 'face_parsing_fine', str(img_id) + '.npy')
                if os.path.exists(_fp_p):
                    _fp = np.load(_fp_p)
                    if _fp.ndim == 3 and _fp.shape[0] == 1: _fp = _fp[0]
                    if _fp.ndim == 3 and _fp.shape[-1] == 1: _fp = _fp[..., 0]
                    if _fp.shape[:2] != mask.shape[:2]:
                        _fp = np.array(Image.fromarray(_fp.astype(np.uint8)).resize(
                            (mask.shape[1], mask.shape[0]), Image.NEAREST), dtype=np.uint8)
                    # R-FIXMASK gate: include upper/lower lip (classes 12+13) if env set
                    if os.environ.get('TG_MOUTH_MASK_FULL', '0') == '1':
                        talking_dict['mouth_mask'] = (_fp == 11) | (_fp == 12) | (_fp == 13) | teeth_mask.astype(bool)
                    else:
                        talking_dict['mouth_mask'] = (_fp == 11) | teeth_mask.astype(bool)
                else:
                    talking_dict['mouth_mask'] = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask

                # 兼容旧脚本：若存在 face_parsing_fine，则在 preload 下也可写入 fp_* mask
                fp_label_path = os.path.join(path, 'face_parsing_fine', str(img_id) + '.npy')
                if os.path.exists(fp_label_path):
                    fp = np.load(fp_label_path)
                    if fp.ndim == 3 and fp.shape[0] == 1:
                        fp = fp[0]
                    if fp.ndim == 3 and fp.shape[-1] == 1:
                        fp = fp[..., 0]
                    if fp.shape[:2] != mask.shape[:2]:
                        fp = np.array(Image.fromarray(fp.astype(np.uint8)).resize((mask.shape[1], mask.shape[0]), Image.NEAREST), dtype=np.uint8)
                    fp_brow = np.isin(fp, [2, 3])
                    fp_eye_core = np.isin(fp, [4, 5])
                    fp_eye_wide = np.isin(fp, [4, 5, 6])
                    fp_nose = (fp == 10)
                    fp_mouth_inner = (fp == 11)
                    fp_lip_upper = (fp == 12)
                    fp_lip_lower = (fp == 13)
                    fp_lips = fp_lip_upper | fp_lip_lower
                    fp_skin = (fp == 1)
                    fp_face = fp_skin | fp_brow | fp_eye_wide | fp_nose | fp_lips | fp_mouth_inner
                    fp_cheek = fp_face & (~(fp_eye_wide | fp_nose | fp_lips | fp_brow | fp_mouth_inner))
                    talking_dict['fp_face_mask'] = fp_face.astype(np.uint8)
                    talking_dict['fp_brow_mask'] = fp_brow.astype(np.uint8)
                    talking_dict['fp_eye_mask'] = fp_eye_core.astype(np.uint8)
                    talking_dict['fp_nose_mask'] = fp_nose.astype(np.uint8)
                    talking_dict['fp_lips_mask'] = fp_lips.astype(np.uint8)
                    talking_dict['fp_cheek_mask'] = fp_cheek.astype(np.uint8)
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, img_id)
                if img_id > auds.shape[0]:
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break

            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[img_id], 0, 2) / 2)
            talking_dict['au25'] = [au25[img_id], au25_25, au25_50, au25_75, au25_100]
            talking_dict['au_exp'] = torch.as_tensor(au_exp[img_id])
            talking_dict['au_exp17'] = torch.as_tensor(au_exp17[img_id])

            pose_meta = load_pose_meta(img_id)
            if pose_meta is not None:
                try:
                    talking_dict['R_head'] = torch.as_tensor(pose_meta['R_head']).float()
                    talking_dict['t_head'] = torch.as_tensor(pose_meta['t_head']).float()
                except Exception:
                    pass

            # V17/landmarks: fields required by train_face_v2 / train_mouth_v2.
            # Replicate the original TG reader's square-pad on lips_rect — a
            # thin closed-mouth bbox (e.g. 2 px tall) breaks LPIPS later via a
            # 0-sized maxpool output. l = max(half-width, half-height) ensures
            # a min crop side of 2*l.
            if ldmks_lips_arr is not None and idx < len(ldmks_lips_arr):
                _lr = ldmks_lips_arr[idx]
                _xmin, _xmax, _ymin, _ymax = int(_lr[0]), int(_lr[1]), int(_lr[2]), int(_lr[3])
                _cx = (_xmin + _xmax) // 2
                _cy = (_ymin + _ymax) // 2
                _l = max(_xmax - _xmin, _ymax - _ymin) // 2
                # guarantee a usable LPIPS crop (LPIPS alex needs >= ~32 px)
                _l = max(_l, 32)
                talking_dict['lips_rect'] = [_cx - _l, _cx + _l, _cy - _l, _cy + _l]
                _lh = ldmks_lhalf_arr[idx]
                talking_dict['lhalf_rect'] = [int(_lh[0]), int(_lh[1]), int(_lh[2]), int(_lh[3])]
                _br = ldmks_brow_arr[idx]
                talking_dict['brow_rect'] = [int(_br[0]), int(_br[1]), int(_br[2]), int(_br[3])]
                talking_dict['mouth_bound'] = [
                    int(mouth_lb), int(mouth_ub),
                    int(ldmks_mouth_arr[idx, 1] - ldmks_mouth_arr[idx, 0])
                ]

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    preload = not bool(getattr(args, 'au_editor_mode', False))

    if not eval:
        print('Reading Training Transforms')
        train_cam_infos = readCamerasFromTransforms(path, 'transforms_train.json', white_background, extension, audio_file, audio_extractor, preload=preload)
    print('Reading Test Transforms')
    test_cam_infos = readCamerasFromTransforms(path, 'transforms_val.json', white_background, extension, audio_file, audio_extractor, preload=preload)

    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, 'points3d.ply')
    if not os.path.exists(ply_path) or True:
        num_pts = args.init_num
        print(f'Generating random point cloud ({num_pts})...')
        xyz = np.random.random((num_pts, 3)) * 0.2 - 0.1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    'Colmap': None,
    'Blender': readNerfSyntheticInfo,
}
