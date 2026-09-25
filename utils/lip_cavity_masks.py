"""V28 lip/cavity mask helper. Toggled via env TG_LIP_CAVITY=1.

When enabled and per-frame lip_mask/cavity_mask .npy exist, mouth_mask is
the union of the two and face_mask excludes them. Otherwise this falls back
exactly to the legacy teeth_mask + parsing(100,100,100) gray pipeline.
"""

import os
import numpy as np


_USE_LIP_CAVITY = os.environ.get('TG_LIP_CAVITY', '0') == '1'


def use_lip_cavity():
    return _USE_LIP_CAVITY


def load_face_mouth_masks(path, img_id, mask_rgb):
    """Returns (face_mask, mouth_mask, lip_mask_or_none, cavity_mask_or_none, used_v28)."""
    if _USE_LIP_CAVITY:
        lip_p = os.path.join(path, 'lip_mask', f'{img_id}.npy')
        cav_p = os.path.join(path, 'cavity_mask', f'{img_id}.npy')
        if os.path.exists(lip_p) and os.path.exists(cav_p):
            lip = np.load(lip_p).astype(np.bool_)
            cav = np.load(cav_p).astype(np.bool_)
            mouth_mask = lip | cav
            bg_blue = (mask_rgb[:, :, 2] > 254) * (mask_rgb[:, :, 0] == 0) * (mask_rgb[:, :, 1] == 0)
            face_mask = bg_blue ^ mouth_mask
            return face_mask, mouth_mask, lip, cav, True
    teeth_p = os.path.join(path, 'teeth_mask', f'{img_id}.npy')
    teeth = np.load(teeth_p).astype(np.bool_)
    bg_blue = (mask_rgb[:, :, 2] > 254) * (mask_rgb[:, :, 0] == 0) * (mask_rgb[:, :, 1] == 0)
    face_mask = bg_blue ^ teeth
    gray = (mask_rgb[:, :, 0] == 100) * (mask_rgb[:, :, 1] == 100) * (mask_rgb[:, :, 2] == 100)
    mouth_mask = gray + teeth
    return face_mask, mouth_mask, None, None, False
