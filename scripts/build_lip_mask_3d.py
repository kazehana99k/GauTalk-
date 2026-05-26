"""Build a per-Gaussian lip mask via FP-class votes (classes 11+12+13).
Saves a [N,1] float tensor to OUT_PATH (defaults to <source>/lip_mask_3d.pt).
"""
import sys, os, argparse, copy, torch, numpy as np
sys.path.insert(0, '.')
from PIL import Image
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import loadCamOnTheFly
from utils.graphics_utils import fov2focal
from arguments import ModelParams, PipelineParams, get_combined_args
from train_au_editor import canonical_to_world

parser = argparse.ArgumentParser()
model = ModelParams(parser); pipeline = PipelineParams(parser)
parser.add_argument('--ckpt', required=True)
parser.add_argument('--out', default=None)
parser.add_argument('--sample_views', type=int, default=120)
parser.add_argument('--vote_ratio', type=float, default=0.25)
parser.add_argument('--depth_margin', type=float, default=0.02)
parser.add_argument('--lip_labels', type=str, default='11,12,13', help='comma-sep FP classes')
parser.add_argument('--disable_pose', action='store_true')
parser.add_argument('--quiet', action='store_true')
args = get_combined_args(parser); safe_state(args.quiet)
dataset = model.extract(args); setattr(dataset, 'au_editor_mode', True)

device = 'cuda'
g = GaussianModel(dataset.sh_degree); scene = Scene(dataset, g, shuffle=False)
raw = torch.load(args.ckpt)
g.restore(raw[0], None)
base_xyz = g.get_xyz.detach()
N = base_xyz.shape[0]
print(f'[build_lip_mask] N={N} Gaussians')

train_cams = scene.getTrainCameras()
n_cams = len(train_cams)
print(f'[build_lip_mask] {n_cams} train cams; sampling {args.sample_views}')
sample_ids = np.linspace(0, n_cams-1, min(args.sample_views, n_cams), dtype=int).tolist()
fp_dir = os.path.join(args.source_path, 'face_parsing_fine')
assert os.path.isdir(fp_dir), f'no face_parsing_fine at {fp_dir}'
lip_classes = [int(x) for x in args.lip_labels.split(',')]
print(f'[build_lip_mask] using FP classes {lip_classes}')

votes = torch.zeros(N, device=device, dtype=torch.float32)
valid_views = 0
for sid in sample_ids:
    cam = train_cams[int(sid)]
    if cam.original_image is None:
        cam = loadCamOnTheFly(copy.deepcopy(cam))
    img_id = cam.talking_dict.get('img_id', None)
    if img_id is None: continue
    fp_path = os.path.join(fp_dir, f'{int(img_id)}.npy')
    if not os.path.exists(fp_path): continue
    fp = np.load(fp_path)
    if fp.ndim == 3 and fp.shape[0] == 1: fp = fp[0]
    if fp.ndim == 3 and fp.shape[-1] == 1: fp = fp[..., 0]
    if fp.ndim != 2: continue
    h = int(cam.image_height); w = int(cam.image_width)
    if fp.shape[:2] != (h, w):
        fp = np.array(Image.fromarray(fp.astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8)
    lip2d = np.isin(fp, lip_classes)
    if lip2d.mean() <= 1e-6: continue
    lip_mask_2d = torch.as_tensor(lip2d, dtype=torch.float32, device=device)

    xyz_world = canonical_to_world(base_xyz, cam, args.disable_pose)
    R = torch.as_tensor(cam.R, dtype=torch.float32, device=device)
    T = torch.as_tensor(cam.T, dtype=torch.float32, device=device)
    xyz_cam = xyz_world @ R.T + T.unsqueeze(0)
    fovx = cam.FoVx if hasattr(cam, 'FoVx') else cam.FovX
    fovy = cam.FoVy if hasattr(cam, 'FoVy') else cam.FovY
    fx = float(fov2focal(float(fovx), w)); fy = float(fov2focal(float(fovy), h))
    cx = w*0.5; cy = h*0.5
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    xp = fx * (xyz_cam[:, 0] / (z+1e-8)) + cx
    yp = fy * (xyz_cam[:, 1] / (z+1e-8)) + cy
    valid = valid & (xp>=0) & (xp<w) & (yp>=0) & (yp<h)
    if valid.sum().item() == 0: continue
    xi = torch.clamp(xp.long(), 0, w-1)
    yi = torch.clamp(yp.long(), 0, h-1)
    # near-front filter
    flat = yi*w + xi
    flat_v = flat[valid]
    z_v = z[valid]
    zmin = torch.full((h*w,), float('inf'), device=device, dtype=z.dtype)
    if flat_v.numel() > 0:
        order = torch.argsort(flat_v)
        flat_s = flat_v[order]; z_s = z_v[order]
        uniq, counts = torch.unique_consecutive(flat_s, return_counts=True)
        st = 0
        for k, c in enumerate(counts.tolist()):
            zmin[uniq[k].long()] = z_s[st:st+c].min()
            st += c
    z_front = zmin[flat]
    front = z <= (z_front + args.depth_margin)
    fv = valid & front
    votes += lip_mask_2d[yi, xi] * fv.float()
    valid_views += 1
print(f'[build_lip_mask] {valid_views} valid views; vote thr = {valid_views*args.vote_ratio:.1f}')

thr = max(1.0, valid_views * args.vote_ratio)
mask = (votes >= thr).float().unsqueeze(1)
cov = float(mask.mean().item())
print(f'[build_lip_mask] coverage: {cov:.4f} ({int(mask.sum().item())}/{N})')
if cov < 0.001:
    print('WARN: tiny coverage, mask may be unreliable')

out = getattr(args, 'out', '') or os.path.join(args.source_path, 'lip_mask_3d.pt')
torch.save({'mask': mask.cpu(), 'votes': votes.cpu(), 'valid_views': valid_views,
            'lip_labels': lip_classes, 'vote_ratio': args.vote_ratio}, out)
print(f'[build_lip_mask] saved to {out}')
