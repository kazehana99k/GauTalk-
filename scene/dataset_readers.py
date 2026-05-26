

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        # ============================================================
        # AU Editor / Debug：在构建 CameraInfo 前截断 frames（硬限制内存）
        # ============================================================
        import os as _os
        try:
            if transformsfile.startswith("transforms_train"):
                _max_n = int(_os.environ.get("TG_MAX_TRAIN_FRAMES", "0") or "0")
                if _max_n > 0 and len(frames) > _max_n:
                    frames = frames[:_max_n]
            if transformsfile.startswith("transforms_val"):
                _max_n_val = int(_os.environ.get("TG_MAX_TEST_FRAMES", "0") or "0")
                if _max_n_val > 0 and len(frames) > _max_n_val:
                    frames = frames[:_max_n_val]
        except Exception:
            pass

        # ============================================================
        # AU Editor / Debug 模式：在构建 CameraInfo 之前就截断 frames
        # （否则 7938 帧会在此处全部重对象化，直接吃满系统内存）
        # ============================================================
        import os as _os
        try:
            if transformsfile.startswith("transforms_train") and "TG_MAX_TRAIN_FRAMES" in _os.environ:
                _max_n = int(_os.environ.get("TG_MAX_TRAIN_FRAMES", "0") or "0")
                if _max_n > 0 and len(frames) > _max_n:
                    frames = frames[:_max_n]
            if transformsfile.startswith("transforms_val") and "TG_MAX_TEST_FRAMES" in _os.environ:
                _max_n_val = int(_os.environ.get("TG_MAX_TEST_FRAMES", "0") or "0")
                if _max_n_val > 0 and len(frames) > _max_n_val:
                    frames = frames[:_max_n_val]
        except Exception:
            pass

        # ============================================================
        # AU Editor / Debug 模式：在构建 CameraInfo 之前就截断 frames
        # （否则 7938 帧会在此处全部重对象化，直接吃满系统内存）
        # ============================================================
        import os as _os
        try:
            if transformsfile.startswith("transforms_train") and "TG_MAX_TRAIN_FRAMES" in _os.environ:
                _max_n = int(_os.environ.get("TG_MAX_TRAIN_FRAMES", "0") or "0")
                if _max_n > 0 and len(frames) > _max_n:
                    frames = frames[:_max_n]
            if transformsfile.startswith("transforms_val") and "TG_MAX_TEST_FRAMES" in _os.environ:
                _max_n_val = int(_os.environ.get("TG_MAX_TEST_FRAMES", "0") or "0")
                if _max_n_val > 0 and len(frames) > _max_n_val:
                    frames = frames[:_max_n_val]
        except Exception:
            pass
        # 若存在环境变量 TG_MAX_TRAIN_FRAMES / TG_MAX_TEST_FRAMES，则可以在
        # 读取 transforms 阶段就截断帧数量，避免在长序列上生成过多 CameraInfo。
        # - 仅当对应变量 > 0 时生效；
        # - 对 train / val 分别使用不同的变量。
        import os as _os
        if transformsfile.startswith("transforms_train") and "TG_MAX_TRAIN_FRAMES" in _os.environ:
            try:
                _max_n = int(_os.environ.get("TG_MAX_TRAIN_FRAMES", "0"))
            except ValueError:
                _max_n = 0
            if _max_n > 0 and len(frames) > _max_n:
                frames = frames[:_max_n]
        if transformsfile.startswith("transforms_val") and "TG_MAX_TEST_FRAMES" in _os.environ:
            try:
                _max_n_val = int(_os.environ.get("TG_MAX_TEST_FRAMES", "0"))
            except ValueError:
                _max_n_val = 0
            if _max_n_val > 0 and len(frames) > _max_n_val:
                frames = frames[:_max_n_val]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        ldmks_brow = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            brow = slice(17, 27)  # 17-21: left brow, 22-26: right brow
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            # 眉毛区域矩形（适当扩一点）
            bxmin, bxmax = int(lms[brow, 1].min()), int(lms[brow, 1].max())
            bymin, bymax = int(lms[brow, 0].min()), int(lms[brow, 0].max())
            pad_y = max(2, (bymax - bymin) // 4)
            pad_x = max(2, (bxmax - bxmin) // 6)
            bxmin = bxmin - pad_x
            bxmax = bxmax + pad_x
            bymin = bymin - pad_y
            bymax = bymax + pad_y
            ldmks_brow.append([int(bxmin), int(bxmax), int(bymin), int(bymax)])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        ldmks_brow = np.array(ldmks_brow)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m

                # ------------------------------------------------------------
                # BiSeNet 精细 face parsing：
                # 之前这里会直接 np.load 全部 face_parsing_fine/{id}.npy 并派生
                # fp_*_mask 存进 talking_dict，导致长序列时常驻内存巨大。
                # 为了节省内存，这里只记录该帧是否存在 parsing 文件，真正的
                # fp_*_mask 在训练脚本（如 train_au_editor.py）中按需加载。
                # ------------------------------------------------------------
                fp_label_path = os.path.join(path, 'face_parsing_fine', str(frame['img_id']) + '.npy')
                talking_dict['has_fp_parsing'] = os.path.exists(fp_label_path)



            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['brow_rect'] = ldmks_brow[idx].tolist()
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor

    # 对于 AU 编辑器等仅需按需加载图像 / mask 的场景，可以在 args 上设置
    # au_editor_mode=True，从而关闭预加载（preload=False），避免一次性把所有
    # 帧的 gt_imgs / torso_imgs / parsing 等大数组塞进内存。
    preload = not bool(getattr(args, "au_editor_mode", False))

    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_train.json",
            white_background,
            extension,
            audio_file,
            audio_extractor,
            preload=preload,
        )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_val.json",
        white_background,
        extension,
        audio_file,
        audio_extractor,
        preload=preload,
    )
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

# ======================================================================================
# AU-Editor-safe overrides (FINAL definitions)
# --------------------------------------------------------------------------------------
# This file historically contained multiple duplicated definitions of:
#   - readCamerasFromTransforms
#   - readNerfSyntheticInfo
# causing later duplicates to silently override the optimized versions.
#
# To guarantee correctness (frame truncation + preload control) we place a FINAL override
# here at the end of the file, and rebind sceneLoadTypeCallbacks accordingly.
#
# Key properties:
#   - Respect env vars TG_MAX_TRAIN_FRAMES / TG_MAX_TEST_FRAMES by slicing frames BEFORE
#     constructing CameraInfo objects.
#   - Respect args.au_editor_mode by using preload=False, so dataset init will NOT load
#     all images/masks into RAM.
# ======================================================================================

def readCamerasFromTransforms(
    path,
    transformsfile,
    white_background,
    extension=".jpg",
    audio_file="",
    audio_extractor="deepspeech",
    preload=True,
):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]

        frames = contents["frames"]
        # Hard truncate BEFORE building Python objects
        try:
            if transformsfile.startswith("transforms_train"):
                _max_n = int(os.environ.get("TG_MAX_TRAIN_FRAMES", "0") or "0")
                if _max_n > 0 and len(frames) > _max_n:
                    frames = frames[:_max_n]
            if transformsfile.startswith("transforms_val"):
                _max_n_val = int(os.environ.get("TG_MAX_TEST_FRAMES", "0") or "0")
                if _max_n_val > 0 and len(frames) > _max_n_val:
                    frames = frames[:_max_n_val]
        except Exception:
            pass

        # Background image is only needed if preload=True (compositing torso)
        bg_img = None
        if preload:
            bg_img = np.array(Image.open(os.path.join(path, "bc.jpg")).convert("RGB"))

        # Audio features (loaded once)
        if audio_file == "":
            aud_features = np.load(os.path.join(path, f"aud_{postfix_dict[audio_extractor]}.npy"))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features).float().permute(0, 2, 1)
        auds = aud_features

        au_info = pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns:
                return df[key].values
            if key.strip() in df.columns:
                return df[key.strip()].values
            if key.replace(" ", "") in df.columns:
                return df[key.replace(" ", "")].values
            return np.zeros(len(df), dtype=np.float32)

        au_blink = get_au_col(au_info, " AU45_r")
        au25 = get_au_col(au_info, " AU25_r")
        au25 = np.clip(au25, 0, np.percentile(au25, 95))
        au25_25, au25_50, au25_75, au25_100 = (
            np.percentile(au25, 25),
            np.percentile(au25, 50),
            np.percentile(au25, 75),
            au25.max(),
        )

        au_exp = []
        for i in [1, 4, 5, 6, 7, 45, 25]:  # R-FACE-AU25: added AU25 (7-dim)
            _key = " AU" + str(i).zfill(2) + "_r"
            au_exp_t = get_au_col(au_info, _key)
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
            au_exp.append(au_exp_t[:, None])
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

        aud_ids_17 = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        au_exp17 = []
        for i in aud_ids_17:
            _key = " AU" + str(i).zfill(2) + "_r"
            vals = get_au_col(au_info, _key)
            if i == 45:
                vals = np.clip(vals, 0, 2)
            au_exp17.append(vals[:, None])
        au_exp17 = np.concatenate(au_exp17, axis=-1, dtype=np.float32)

        # Pose meta loader (small JSONs)
        pose_meta_dir = os.path.join(path, "pose_meta")
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
                        with open(meta_path, "r") as f:
                            pose_meta_cache[name] = json.load(f)
                    return pose_meta_cache[name]
            return None

        # For preload=False, avoid opening every image just to get size; read once.
        fixed_w = fixed_h = None
        if not preload and len(frames) > 0:
            first_id = frames[0]["img_id"]
            first_path = os.path.join(path, "gt_imgs", str(first_id) + extension)
            with Image.open(first_path) as _im:
                fixed_w, fixed_h = _im.size[0], _im.size[1]

        for idx, frame in tqdm(enumerate(frames)):
            img_id = frame["img_id"]
            cam_name = os.path.join("gt_imgs", str(img_id) + extension)
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            if preload:
                image = Image.open(image_path)
                w, h = image.size[0], image.size[1]
                image = np.array(image.convert("RGB"))
            else:
                image = None
                w, h = fixed_w, fixed_h

            bg = None
            if preload:
                torso_img_path = os.path.join(path, "torso_imgs", str(img_id) + ".png")
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)

            talking_dict = {"img_id": img_id}

            if audio_file == "":
                talking_dict["auds"] = get_audio_features(auds, 2, img_id)
                if img_id > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict["auds"] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break

            talking_dict["blink"] = torch.as_tensor(np.clip(au_blink[img_id], 0, 2) / 2)
            talking_dict["au25"] = [au25[img_id], au25_25, au25_50, au25_75, au25_100]
            talking_dict["au_exp"] = torch.as_tensor(au_exp[img_id])
            talking_dict["au_exp17"] = torch.as_tensor(au_exp17[img_id])

            pose_meta = load_pose_meta(img_id)
            if pose_meta is not None:
                try:
                    talking_dict["R_head"] = torch.as_tensor(pose_meta["R_head"]).float()
                    talking_dict["t_head"] = torch.as_tensor(pose_meta["t_head"]).float()
                    if "flame_exp" in pose_meta:
                        talking_dict["flame_exp"] = torch.as_tensor(pose_meta["flame_exp"]).float()
                    if "jaw_pose" in pose_meta:
                        talking_dict["jaw_pose"] = torch.as_tensor(pose_meta["jaw_pose"]).float()
                except Exception:
                    pass

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=w,
                    height=h,
                    background=bg,
                    talking_dict=talking_dict,
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    preload = not bool(getattr(args, "au_editor_mode", False))

    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_train.json",
            white_background,
            extension,
            audio_file,
            audio_extractor,
            preload=preload,
        )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_val.json",
        white_background,
        extension,
        audio_file,
        audio_extractor,
        preload=preload,
    )

    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 0.2 - 0.1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": None,
    "Blender": readNerfSyntheticInfo,
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        # 支持通过环境变量在 transforms 层面截断帧数量，避免在长序列上生成过多 CameraInfo：
        #   - TG_MAX_TRAIN_FRAMES：限制 transforms_train.json
        #   - TG_MAX_TEST_FRAMES ：限制 transforms_val.json
        import os as _os2
        if transformsfile.startswith("transforms_train") and "TG_MAX_TRAIN_FRAMES" in _os2.environ:
            try:
                _max_n = int(_os2.environ.get("TG_MAX_TRAIN_FRAMES", "0") or "0")
            except ValueError:
                _max_n = 0
            if _max_n > 0 and len(frames) > _max_n:
                frames = frames[:_max_n]
        if transformsfile.startswith("transforms_val") and "TG_MAX_TEST_FRAMES" in _os2.environ:
            try:
                _max_n_val = int(_os2.environ.get("TG_MAX_TEST_FRAMES", "0") or "0")
            except ValueError:
                _max_n_val = 0
            if _max_n_val > 0 and len(frames) > _max_n_val:
                frames = frames[:_max_n_val]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor

    # AU 编辑模式下关闭大规模 preload，只在训练循环中用 loadCamOnTheFly
    # / load_fp_masks_on_the_fly 逐帧读取图像和 mask，避免一次性占满内存。
    preload = not bool(getattr(args, "au_editor_mode", False))

    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_train.json",
            white_background,
            extension,
            audio_file,
            audio_extractor,
            preload=preload,
        )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_val.json",
        white_background,
        extension,
        audio_file,
        audio_extractor,
        preload=preload,
    )
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    # AU Editor 模式：关闭 preload，避免在 dataset 初始化阶段加载全部图像/分割/roi
    preload = not bool(getattr(args, "au_editor_mode", False))
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_train.json",
            white_background,
            extension,
            audio_file,
            audio_extractor,
            preload=preload,
        )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_val.json",
        white_background,
        extension,
        audio_file,
        audio_extractor,
        preload=preload,
    )
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    # AU Editor 模式：关闭 preload，避免在 dataset 初始化阶段加载全部图像/分割/roi
    preload = not bool(getattr(args, "au_editor_mode", False))
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_train.json",
            white_background,
            extension,
            audio_file,
            audio_extractor,
            preload=preload,
        )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_val.json",
        white_background,
        extension,
        audio_file,
        audio_extractor,
        preload=preload,
    )
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    preload = not bool(getattr(args, "au_editor_mode", False))
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(
            path, "transforms_train.json", white_background, extension, audio_file, audio_extractor, preload=preload
        )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_val.json", white_background, extension, audio_file, audio_extractor, preload=preload
    )
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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
from PIL import Image, ImageDraw
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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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
from PIL import Image, ImageDraw
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
    mask_img = Image.new(L, (w, h), 0)
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
        roi_brows: m_brows,
        roi_eyes: m_eyes,
        roi_nose: m_nose,
        roi_mouth: m_mouth,
        roi_forehead: m_forehead,
        roi_cheeks: m_cheeks,
    }


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

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
from PIL import Image, ImageDraw
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
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info=pd.read_csv(os.path.join(path, os.environ.get('TG_AU_CSV', 'au.csv')))

        def get_au_col(df, key):
            if key in df.columns: return df[key].values
            if key.strip() in df.columns: return df[key.strip()].values
            if key.replace(" ", "") in df.columns: return df[key.replace(" ", "")].values
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



        # 旧逻辑：这里预先为所有帧加载 landmarks 并生成 ROI（lips_rect / lhalf_rect 等），
        # 在 AU editor 中我们已经主要依赖 BiSeNet face_parsing_fine 的 fp_*_mask 做精细 ROI，
        # 因此关闭这一步的大规模 landmarks ROI 预计算，以减少初始化阶段的 IO 和内存占用。

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





        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                _face_m, _mouth_m, _lip_m, _cav_m, _used_v28 = load_face_mouth_masks(
                    path, str(frame['img_id']), mask)
                talking_dict['face_mask'] = _face_m
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = _mouth_m
                if _used_v28:
                    talking_dict['lip_mask'] = _lip_m
                    talking_dict['cavity_mask'] = _cav_m

                # ------------------------------------------------------------
                # 额外：加载 BiSeNet face parsing 的原始 label（face_parsing_fine/{id}.npy），
                # 生成精细的 fp_* mask，供 AU 训练 / ROI 调试使用。
                # ------------------------------------------------------------
                fp_label_path = os.path.join(path, 'face_parsing_fine', str(frame['img_id']) + '.npy')
                if os.path.exists(fp_label_path):
                    fp = np.load(fp_label_path)
                    # 统一形状到 [H,W]
                    if fp.ndim == 3 and fp.shape[0] == 1:
                        fp = fp[0]
                    if fp.ndim == 3 and fp.shape[2] == 1:
                        fp = fp[..., 0]
                    if fp.shape[:2] != mask.shape[:2]:
                        fp = np.array(
                            Image.fromarray(fp.astype(np.uint8)).resize(
                                (mask.shape[1], mask.shape[0]), Image.NEAREST
                            ),
                            dtype=np.uint8,
                        )

                    # CelebAMask-HQ / face-parsing.PyTorch 类别定义：
                    # 1: skin; 2,3: brows; 4,5: eyes; 6: eye-glass; 10: nose;
                    # 11: mouth(inner); 12: upper-lip; 13: lower-lip; 17: hair
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
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])
            talking_dict['au_exp17'] = torch.as_tensor(au_exp17[frame['img_id']])
            pose_meta = load_pose_meta(frame['img_id'])
            if pose_meta is not None:
                try:
                    talking_dict['R_head'] = torch.as_tensor(pose_meta['R_head']).float()
                    talking_dict['t_head'] = torch.as_tensor(pose_meta['t_head']).float()
                    if 'flame_exp' in pose_meta:
                        talking_dict['flame_exp'] = torch.as_tensor(pose_meta['flame_exp']).float()
                    if 'jaw_pose' in pose_meta:
                        talking_dict['jaw_pose'] = torch.as_tensor(pose_meta['jaw_pose']).float()
                except Exception:
                    pass
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
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
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}

# ======================================================================================
# AU-Editor-safe overrides (APPENDED FINAL)
# --------------------------------------------------------------------------------------
# Guarantee that the *last* definitions in this file respect:
#   1) TG_MAX_TRAIN_FRAMES / TG_MAX_TEST_FRAMES: hard truncate before CameraInfo creation
#   2) args.au_editor_mode: preload=False so dataset init won't load all images/masks
# --------------------------------------------------------------------------------------
# NOTE: This file contains many historical duplicate definitions; appending here ensures
# Python uses these final ones.
# ======================================================================================

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
