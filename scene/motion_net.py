import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64

        # R-FACE-AU25 (2026-05-25): exp_feat extended 6->7 by appending AU25.
        # First 5 AUs (AU01,04,05,06,07) are encoded; AU45 and AU25 are kept
        # raw at the end of enc_e so the model can learn distinct responses
        # to blink (AU45) and jaw drop (AU25).
        self.exp_in_dim = 7 - 2  # encode first 5 (AU01..AU07)
        self.eye_dim = 7 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 2, 16, 2)  # eye_dim-2 raw slots

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a


    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att

        eye_att = torch.relu(self.eye_att_net(enc_x))
        # R-FACE-AU25: encode first 5 AUs, keep AU45 + AU25 raw at end.
        enc_e = self.exp_encode_net(e[:-2])
        enc_e = torch.cat([enc_e, e[-2:]], dim=-1)  # [enc(AU01..07), AU45, AU25]
        enc_e = enc_e * eye_att
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        h = self.sigma_net(h)

        # R-AU45-AMP (gated, inference-time only): post-hoc amplifier for AU45→eye
        # motion. Diagnostic showed baseline model maps AU45=2 → |d_y|=0.001 in eye
        # region (~1 pixel) but actual eyelid motion needs ~5 pixels. Run a second
        # forward with AU45=0 and amplify the diff in eye_att-masked region.
        # Set env TG_AU45_EYE_GAIN > 1 to enable (e.g., 5 means 5× amplification).
        import os as _os
        _au45_gain = float(_os.environ.get('TG_AU45_EYE_GAIN', '1.0'))
        if _au45_gain > 1.0 and e is not None:
            with torch.no_grad():
                # R-FACE-AU25: AU45 is now at index -2 (AU25 at -1); zero AU45 only.
                _e_zero = e.clone(); _e_zero[-2] = 0
                _enc_e_zero = self.exp_encode_net(_e_zero[:-2])
                _enc_e_zero = torch.cat([_enc_e_zero, _e_zero[-2:]], dim=-1)
                _enc_e_zero = _enc_e_zero * eye_att
                if c is not None:
                    _h_zero = torch.cat([enc_x, enc_w, _enc_e_zero, c], dim=-1)
                else:
                    _h_zero = torch.cat([enc_x, enc_w, _enc_e_zero], dim=-1)
                _h_zero = self.sigma_net(_h_zero)
                # diff in d_xyz raw (before tanh cap)
                _diff_dy_raw = (h[..., 1] - _h_zero[..., 1]) * 1e-2  # match d_xyz scale
                _eye_mask = eye_att[:, -2]  # AU45 channel is now second-to-last
                _amp = (_au45_gain - 1.0) * _diff_dy_raw * _eye_mask
                # apply to h[..., 1] before cap so tanh still bounds final output
                h = h.clone()
                h[..., 1] = h[..., 1] + _amp / 1e-2  # invert the scale

        # R-GEO-1: tanh cap on d_xyz / d_scale (parallels MouthMotionNetwork capE).
        # Without a cap, face MotionNetwork can output extreme d_xyz on
        # out-of-distribution audio (val frames with rare open-mouth pose),
        # pushing Gaussians off-screen. This is the geometric mechanism behind
        # macron lower-lip "vanishing" on test frames.
        # Caps are ~60% of mouth's (face motion is smaller overall):
        #   mouth: [0.025, 0.15, 0.04];  face: [0.015, 0.06, 0.025]
        d_xyz_raw = h[..., :3] * 1e-2
        _cap_x, _cap_y, _cap_z = 0.015, 0.06, 0.025
        d_xyz = torch.stack([
            torch.tanh(d_xyz_raw[..., 0] / _cap_x) * _cap_x,
            torch.tanh(d_xyz_raw[..., 1] / _cap_y) * _cap_y,
            torch.tanh(d_xyz_raw[..., 2] / _cap_z) * _cap_z,
        ], dim=-1)

        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        # d_scale: cap exp(d_scale) ∈ [0.6, 1.7] roughly (tanh*0.5)
        d_scale_raw = h[..., 8:11]
        d_scale = torch.tanh(d_scale_raw / 0.5) * 0.5
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params




class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

        # yBypass (2026-05-18): direct y-position injection for anti-correlated
        # lip motion. HashGrid spatial smoothness prevents nearby upper/lower
        # lip Gaussians from learning opposite motion. By feeding y_rel
        # (normalized y-pos) and audio jointly to a small MLP that outputs a
        # y-displacement, we bypass HashGrid's smoothing for the y axis.
        # Result d_xyz_y_extra is ADDED to sigma_net's d_xyz_y before tanh cap.
        self.y_bypass_proj = MLP(self.audio_dim + 1, 1, 16, 2)

        # Q1 (2026-05-19): Audio→lip-landmark joint conditioning.
        # External predictor produces 20 lip landmarks (relative). We embed them
        # into a 32-d vector and use it as ADDITIVE residual to d_xyz (not
        # replacing sigma_net). This grounds motion in explicit geometric target.
        self.lmk_proj = MLP(20 * 2, 32, 32, 2)
        self.lmk_to_dxyz = MLP(self.in_dim + 32, 3, 32, 3)

        # R-MOUTH-AU (2026-05-24): mouth motion currently only takes audio.
        # Diagnostic showed mouth open amplitude is conservative (model only
        # listens to audio, ignores explicit AU signal). Add a small AU-driven
        # branch that produces an ADDITIVE d_xyz residual. Zero-init last layer
        # so it starts as a no-op (preserves baseline ckpt compatibility under
        # strict=False load). Input: per-Gaussian xyz feat + AU25 scalar (jaw
        # drop). Output: 3-d d_xyz residual.
        self.au_mouth_branch = MLP(self.in_dim + 1, 3, 16, 2)
        # zero-init last layer for safe no-op start
        with torch.no_grad():
            for _name, _m in self.au_mouth_branch.named_modules():
                if isinstance(_m, nn.Linear):
                    pass  # find last linear below
            _linears = [_m for _m in self.au_mouth_branch.modules() if isinstance(_m, nn.Linear)]
            if _linears:
                nn.init.zeros_(_linears[-1].weight)
                if _linears[-1].bias is not None:
                    nn.init.zeros_(_linears[-1].bias)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
    

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


    def forward(self, x, a, landmark=None, au25=None):
        # x: [N, 3], in [-bound, bound]
        # landmark (optional): [20, 2] predicted lip landmark relative coordinates
        # au25 (optional): scalar tensor of current frame's AU25 (jaw drop).
        #   When provided, the zero-init au_mouth_branch produces an additive
        #   d_xyz residual conditioned on (xyz_feat, AU25).
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        # aud_ch_att = self.aud_ch_att_net(enc_x)
        # enc_w = enc_a * aud_ch_att

        h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5

        # R-MOUTH-AU: additive d_xyz residual conditioned on AU25 (jaw drop).
        # Zero-init at construction → no-op until training shapes it. Operates
        # only on Gaussians where xyz_feat is informative (sigma_net's own
        # encoder shares the same hashgrid).
        if au25 is not None and hasattr(self, 'au_mouth_branch'):
            _au25_v = au25.view(-1)[:1].to(enc_x.dtype)  # scalar
            _au25_g = _au25_v.expand(enc_x.shape[0], 1)
            _au_in = torch.cat([enc_x, _au25_g], dim=-1)
            _dxyz_au = self.au_mouth_branch(_au_in) * 1e-2  # small scale
            d_xyz = d_xyz + _dxyz_au

        # Q1 (2026-05-19): landmark-driven additive residual to d_xyz
        if landmark is not None and hasattr(self, 'lmk_proj'):
            lmk_flat = landmark.reshape(-1)  # [40]
            lmk_emb = self.lmk_proj(lmk_flat.unsqueeze(0))    # [1, 32]
            lmk_emb_g = lmk_emb.expand(enc_x.shape[0], -1)    # [N, 32]
            h_lmk = torch.cat([enc_x, lmk_emb_g], dim=-1)     # [N, in_dim + 32]
            d_xyz_lmk = self.lmk_to_dxyz(h_lmk) * 1e-2        # [N, 3], small scale
            d_xyz = d_xyz + d_xyz_lmk

        # yBypass (2026-05-18): direct y-injection bypassing HashGrid smoothness
        if hasattr(self, 'y_bypass_proj'):
            y_med = x[..., 1].median().detach()
            y_rel = ((x[..., 1:2] - y_med) / 0.05)   # normalized: lip extent ~ 0.1 → [-1, 1]
            _bypass_in = torch.cat([enc_w, y_rel], dim=-1)   # [N, audio_dim + 1]
            _dy_extra = self.y_bypass_proj(_bypass_in).squeeze(-1) * 1e-2  # [N], small init scale
            d_xyz = d_xyz.clone()
            d_xyz[..., 1] = d_xyz[..., 1] + _dy_extra

        # capE (2026-05-17): hard tanh cap on per-axis d_xyz. With corrected
        # AU on obama, audit found motion_net learned raw output norm ≈ 350
        # (output d_xyz_y = -3.5), bypassing the soft 1e-2 scale above and
        # producing the "audio → mouth Gaussians fly off-screen" sink optimum.
        # Tanh is industry standard for blendshape coefficient decoders.
        # Per-axis caps (y largest because mouth opening is mostly vertical):
        # capE-relaxed (2026-05-18 A+C): widened y cap to 0.15 to allow actual
        # lip separation amplitude (~half of 0.10-0.12 lip vertical extent).
        # Previous 0.06 cap was found to limit mouth opening to ~half GT.
        # Sink protection now comes from noPrune+softHinge, not the cap.
        # R-MOUTH-MOTION-GAIN (gated, inference-time): pre-cap scaling of
        # mouth d_xyz to amp mouth-open amplitude. tanh cap below still bounds
        # the output so over-scaling is safe. Set env TG_MOUTH_D_XYZ_GAIN > 1.
        import os as _os
        _gain_xyz = float(_os.environ.get('TG_MOUTH_D_XYZ_GAIN', '1.0'))
        if _gain_xyz != 1.0:
            d_xyz = d_xyz * _gain_xyz

        _cap_x, _cap_y, _cap_z = 0.025, 0.15, 0.040
        d_xyz = torch.stack([
            torch.tanh(d_xyz[..., 0] / _cap_x) * _cap_x,
            torch.tanh(d_xyz[..., 1] / _cap_y) * _cap_y,
            torch.tanh(d_xyz[..., 2] / _cap_z) * _cap_z,
        ], dim=-1)
        return {
            'd_xyz': d_xyz,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        # yBypass small head (2026-05-18) — higher lr to accelerate
        if hasattr(self, 'y_bypass_proj'):
            params.append({'params': self.y_bypass_proj.parameters(),
                           'lr': lr_net * 3, 'weight_decay': wd})

        # Q1 landmark heads
        if hasattr(self, 'lmk_proj'):
            params.append({'params': self.lmk_proj.parameters(), 'lr': lr_net * 3, 'weight_decay': wd})
            params.append({'params': self.lmk_to_dxyz.parameters(), 'lr': lr_net * 3, 'weight_decay': wd})

        return params
