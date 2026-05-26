# PROJECT_HISTORY.md

> 本文件严格基于 `.session-notes/*.md` 重建，跨越 2026-04-16 → 2026-05-12 共 8 份有实质内容的 session。
> 编写规则：不同笔记之间细节冲突时以时间更晚的为准并注明；不确定处写"待确认"；不引用任何空 dump 中"声称"的会话内容。

---

## 1. 项目背景与目标

- 基础项目：**TalkingGaussian**（3D Gaussian 头像 + audio-driven talking face）。仓库根目录 `/home/labliu/wangshiyuan/TalkingGaussian/`，主分支 `main`，存在副本 worktree `/home/labliu/.cursor/worktrees/TalkingGaussian__SSH__labliu_100.102.101.127_/qpz`（两份代码并存，编辑路径会混用）。
- 数据集：**macron**（法国总统讲话视频），用 `face_parsing_fine` 做语义分割。**该数据集没有 label 6 (eyelid)**——这是 AU45 主线问题的根因之一（见 §4 / §5）。
- 改造主线：在 frozen / 联合微调的 canonical 3DGS 之上，外挂 **AU editor**，让模型可由 Action Unit（AU）驱动局部表情/口型；目标是支持**任意新音频**驱动渲染（不靠 GT-fallback 抄像素）。
- 路线演进（粗粒度）：
  1. **单 AU45 闭眼编辑器**（`train_au_editor.py`）→ 调五轮放弃，根因是 frozen canonical + 单 xyz 通道做 14 mm 位移注定糊。
  2. **MultiAUDeformer / 17-AU 多通道**（`train_multi_au.py` + `models/multi_au_deformer.py` / `models/au_transformer_deformer.py` / `models/tg_style_deformer.py`）→ 解决了"全脸糊"，但在 2026-04-26 被定性 17-AU **独立可控性事实层面不成立**。
  3. **Fuse pipeline**（`train_fuse_v*.py` + `train_mouth.py` + `synthesize_fuse_clean.py` + `scripts/render_multi_au.py --use_tg_fuse`）→ 在 2026-05-12 揭穿 v29.7 渲染协议 GT-fallback 系统性虚高 2.1 dB，TG-native clean 协议下嘴部仍黑洞。
- 当前已确认的核心 baseline：**v4 fuse ckpt + v29.7 渲染合成**，PSNR 35.83 / SSIM 0.9665 / LPIPS-vgg 0.0434 / LMD 0.0369 / AOC 0.600 / aperture 0.840（来自 eab0bb52）。但 v29.7 不能写论文（详 §6）。

---

## 2. 演进时间线

> 仅列入有实质内容的会话。空 dump、被打断、OFF-TOPIC 的会话见 §2 末尾"非主线会话"小节。

### 2.1 起源：单 AU45 闭眼编辑器（2026-04-16，a1022478）

- **目标**：用户报告"delta=0.2 即把整个眼区糊死"，从 `monofix_v3` ckpt 起手 debug。
- **关键修改**：
  - `train_au_editor.py` v1（"anti_blanket"）：损失从 `.abs()` 改为 signed close-y `-y * sign(a45)`；新增 `--au45_cover_cap` / `--au45_mono_pp_w` / `--auto_high_abs_target_max`。
  - `scripts/train_fullclose_curriculum_20k.sh` 大幅收紧：`au45_xyz_clip 0.045→0.012`、`residual_xyz_w 1.8→0.55`、`cover_target 0.022→0.004`。
  - v2（"anti_smudge"）：引入 `eye_core_static_w` 只锁 AU45 通路（不冻 `d_xyz`），加 `blink_active_mask_3d = eye_mask * (1 - core)`，让 magnitude loss 在 eyelid 平均、不让 frozen 眼球稀释分母。启用 `--eye_core_from_fp 1`、`--au45_eye_core_static_w 1.0`。
  - v3：加 `loss_raw_l2`、上眼皮 band mask（数据集无 FP label 6，靠 y-band 几何替代），加 `--au45_eye_core_opacity_drop 0.85`。
  - v4 three_branch：去 `--disable_scale_branch`、`--disable_dynamic_opacity`；加 `--au45_residual_scale_w 0.40`、`--au45_residual_opacity_w 1.50`；加 `loss_var` / `loss_scale_aniso` / `loss_eye_lp`（eye-bbox LPIPS）。
  - T3-H sh_finetune：放开 canonical `_features_dc` 在 eye_core 区域可训，加 `--canonical_eye_sh_reg_w 0.10`（SH drift L2 reg）；canonical SH 存的是虹膜色，不放开永远不会变皮肤色。
- **切换决策**：离线推理 step25000 ckpt 发现 611 个 active Gaussian 全部塞到 clip 上限，**输出已饱和为常数闭眼场，零空间结构**。`scale_abs=0.585` 跑满 0.60 上限仍是模糊粉色。结论：**"frozen canonical + 外挂单 AU MLP" 触及理论上限**。
- **架构切换**：新建 `models/multi_au_deformer.py`（hashgrid + 5 区域 attention + 独立 d_xyz/d_scale/d_rot/d_opa head）、`train_multi_au.py`、`scripts/render_multi_au.py`、`scripts/train_multi_au.sh`。
  - v1：l1 0.56→0.040，SSIM 0.90，LPIPS 0.26（指标飞跃）。
  - v2 face_locked：`models/multi_au_deformer.py:155-170` forward 末乘 `face_mask_3d` 硬空间门控；canonical DC/opa backward hook 限脸区。
  - v3 geom_loose：FP `vote_ratio 0.20→0.50` + y/x 几何切，face_mask 80%→34%；cheek prior 改纯几何 bbox 57%→6.7%；正则压缩 5×（`reg_xyz_w 0.10→0.02`）；`au_xyz_scale 1e-2→2e-2`。
  - v4 transformer：新建 `models/au_transformer_deformer.py`。xyz hashgrid → spatial feat[N,36] → Linear(d_model) → 17 AU tokens × au_val + mag_mlp 残差 → cross-attention ×2；head 零初始化；`FP_FACE_LABELS=[1,2,3,4,5,10]`；构造 `head_mask_3d`；渲染前 `opa *= head_mask_3d`，bbox 裁剪 + `--bg_rgb` 纯色背景。
- **卡点**：v4 渲染时 `FileNotFoundError: macron_v4_transformer/multi_au_step20000.pth` + 两次 `API Error: 500`，会话中断。Smoke test 30 步被作者标注"deformer 基本没学到东西"。

### 2.2 v4 transformer 修救 + proportional response（2026-04-19，268ed005 / 6cc6a4a7）

- **目标**：v4 step 18400 attention 塌成 one-hot，step 19008 起永久 NaN（log 千行 `[warn] non-finite loss step ..., skip`）。
- **关键修改**：
  - v5 fresh / v5 resume：`au_transformer_deformer.py:218-248` entropy 改双向 hinge `relu(entropy − 1.1)`，target=log(3)；AU 输入 `clamp(−3,3)`、`mag_residual.clamp(−8,8)`；`train_multi_au.py` 加 last-good snapshot + auto-restore on NaN；canonical DC/opacity 也做 NaN clamp。
  - v6 live_head：诊断"渲染图无变化"。head 零初始化导致死梯度，改为小幅非零初始化。
  - v8 prop_response：加 **proportional response loss** `L_prop = |deformer(xyz, α·Δau) − α·deformer(xyz, Δau).detach()|`，α∈[1.5,4.0]；AU 幅度加权 photo loss `au_weight = 1 + 2·|Δau|_sum`；位移下界 `relu(disp_floor·|Δau| − |d_xyz|_mean)`；`reg_xyz_w 0.02→0.005`。
  - v8b_gentle：v8 直接炸图、"生成的都不是脸了"，所有"鼓励大位移"旋钮 ×1/5：`prop_w 0.10→0.02`、`prop_alpha_max 4.0→2.0`、`au_reweight_scale 2.0→0.5`、`disp_floor 5e-4→1e-4`、`floor_w 0.5→0.05`、`reg_xyz_w` 回 0.02、`grad_clip 0.25→0.15`、`au_xyz_scale 2e-2→1e-2`、`au_scale_scale 1.5e-1→8e-2`、`au_opa_scale 2e-1→1e-1`。
- **卡点**：会话最后两条用户消息"还是不像人啊" / "还是不像人 很虚"，附图触发 2000px 维度限制，**v8b 视觉效果未被作者评估**。

### 2.3 scoped PSNR loss 反制"假闭眼"（2026-04-21，b7fffc65）

- **目标**：模型用低频肉色"假闭眼"骗 L1（PSNR 25 dB 卡住）。
- **关键修改**：
  - `train_multi_au.py`：在 L1/SSIM/LPIPS 之后插入 scoped PSNR loss：`face_mask_2d = fp ∈ [1,2,3,4,5,10]`（天然排掉背景/嘴/头发/耳/上额），`psnr = -10*log10(mse+1e-6)`，`loss += psnr_w * (-psnr)`，仅当 `mask.sum() > 16` 触发。`scripts/train_multi_au.sh` 加 `--psnr_w 0.05`。预期 25 dB → 32–35 dB。
  - `models/tg_style_deformer.py` v17 / v17.1：transformer 输入改 `val_proj(tanh(au_v))` 替代 `val_proj(au_v)`；`coef` clamp `[0,3] → [0.8,1.25]`。理由：AU45 推理到 3.0 时 transformer 落到未见区域，coef 跌到 0.3–0.7 把响应砍半，1.5 之后就闭不上。tanh + 收紧 clamp 保证最差也是 v16 的 80% 响应。
- **结论/卡点**：v18 PSNR loss 是否真把闭眼推到清晰边缘未在 session 内验证；`eyegrad_bridge` 日志显示 `cover_raw=0.0192 / cover_target=0.0158 / pair=0 / highabs_rate=99.97% / au45_abs_avg=1.97`，模型仍在饱和均匀场附近。

### 2.4 v22 / v23 motion_net 集成尝试（2026-04-23，d96e7fd6）

- **目标**：在 v17–v21 基础上继续推到 v23，重点解决 dxyz 位移上限不足、blink 帧 frame_mul 误放大 drift reg、把 TG `motion_net` 拉回来冻结使用。
- **关键修改**：
  - v18–v20.3：低秩分解；TG magnitude 对齐；修 sweep `+delta` 语义；AU45 幅值驱动眼区权重 + 2D mask 膨胀；帧级 `frame_mul`；眼权 3→7。
  - v21：TG 式 canonical 联合微调（开 `--train_canonical_xyz/scaling/rotation`，独立 lr `lr_canonical_xyz=2e-6`，drift L2 reg xyz=50 / scaling=1.0 / rotation=1.0，per-param grad clip xyz=0.1）。
  - v22（两个致命 bug 修复）：
    - `locality_w 0.3→0`（`dxyz_ceil=3e-3 ≈ 5 mm`，闭眼实际需要 ~9e-3 ≈ 15 mm，停在半闭被迫用 d_scale 拉糊）。
    - `frame_mul` 改为**只乘光度 loss**（L1/SSIM/LPIPS/PSNR/region），不再放大 drift reg——之前在 blink 帧用 5× 力把 canonical 拉回原位，抵消了 v21 的联合微调。
  - v23：把 TG `motion_net` 请回来冻结使用；render 时把 swept `au_vec` 注入 `exp6 = stack([au_vec[0,2,3,4,5,16]])`，对应 motion_net 的 [AU01,AU04,AU05,AU06,AU07,AU45]；修了 train fallback 同样的索引 bug。
- **卡点（决定性）**：motion_net 是和 TG 原始 canonical 联合训练的，v21 微调过的 canonical 让它输出不对位 → **反而学不会闭眼**。回退到 v22 + K=3 步强制选一帧 AU45 > 1.0 的 blink 帧做确定性过采样；但该回退方案尚未给出落地 diff 或新 OUT_DIR，会话在此截断。

### 2.5 嘴部黑洞 + lip-sync 体系 + 17-AU 独立可控性失败（2026-04-26，f2ea2eb0 / c0fc744d）

- **目标（用户主诉）**：嘴内部黑洞、口型与音频不同步、嘴张得不够 / 卷唇不足；同时给 lip-sync 引入定量指标，对标 EmoTaG / IP-LAP / TG 论文。
- **关键修改**：
  - `scripts/render_multi_au.py:942-956` **修了黑洞根因**：seq_test 路径原本漏传 `fuse_*` / `tg_mouth_*` 参数，走 legacy 分支导致嘴中心 alpha≈0。
  - `scripts/render_multi_au.py:697-714`：从 fuse ckpt 的 `audio_net.encoder_conv.0.weight` shape 自动检测 hubert(1024) vs deepspeech(29)，覆盖 `dataset.audio_extractor`，并给 `--audio_extractor` 加 deepspeech 兜底（避免 `'esperanto' in None` 崩）。
  - `--composite_mode`：**默认改回 `face_over_mouth`**。`mouth_over_face + mouth_punch=0.7` 是 ratio 暴涨到 1.6 的根因。
  - v29.7 fringe 修复：bg 由 `torso_imgs` 改为 `cam.original_image`；alpha 3-pass erosion + 阈值 0.8（外）/ 0.5（mouth），消除 silhouette 紫边和嘴内侧彩斑。
  - 推理 flag：`--mouth_gain`（默认 1.2）、`--mouth_deformer_gain`、`--canonical_from_fuse`。
  - `train_multi_au.py` 新增训练 flag：`--lip_lpips_w`（lips_rect crop 上 LPIPS）、`--mouth_temporal_w`、`--mouth_aperture_w`（用 `.lms` 计 GT aperture 直接监督）、`--mouth_mag_asym_w`（非对称 under-motion 罚 3×）、`--mouth_photo_boost`、`--lips_rect_w`、`--mouth_dscale_reg_w`、`--mouth_dxyz_reg_w`、`--canonical_from_fuse`、`--tg_fuse_ckpt`；`lips_rect` 在 macron 缺失，加 FP mouth label(7,8) 动态 bbox fallback。
  - `models/mouth_deformer.py`：`xy_compress 0.2→0.3`、`z_compress 0.2→0.6`、`scale_scale/rot_scale 3e-2→6e-2`（放开形变自由度）。
  - 新建 `scripts/eval_lipsync.py`（LMD / AOC / Lip-LPIPS / aperture ratio）、`scripts/eval_full.py`（PSNR/SSIM/LPIPS full+mouth ROI、LMD/AOC、可选 AUE 调本地 OpenFace）、`scripts/eval_au_controllability.py` + `scripts/eval_v5_inline.py`。
  - 新建 `train_fuse_v2/v3/v4/v5/v6.py`：v4 是 LPIPS alex→vgg、weight 0.5→1.0、patch 32→64-80 的"ABC"三件套；v5/v6 加 LightingHead（view-conditional 颜色残差头，失败）。
  - `train_mouth_v4.py` / `train_face_v4.py`：试图把 v4 LPIPS 下沉到 face/mouth 阶段并加 5 个 while 死循环 break 保护，最终因卡死放弃（详 §5）。
- **结论**：
  - **最终选型基线 = v4 fuse ckpt + v29.7 渲染**：PSNR 35.83 / SSIM 0.9665 / LPIPS-vgg 0.0434 / LMD 0.0369 / AOC 0.600 / aperture 0.840。SSIM 超过论文，PSNR 超 IP-LAP/EmoTaG，LPIPS 仍逊于 EmoTaG 0.026。
  - AOC 0.20 是采样噪声（N=16），N=128 后 baseline 0.58；`fuse mouth_gain=1.2` → AOC 0.64（甜点）。
  - **决定性结论**：Path A AU 可控性评估（17 AU sweep + OpenFace 反测），C[i,i] ≈ 0；放大 `au_gain` 后差异主要是色彩破碎/纹理噪声，不是语义 AU motion → **17-AU 独立可控性事实层面不成立**。
  - 创新清单盘点：真能写论文的只有 17-AU TGStyleDeformer + cross-AU transformer + MouthDeformer Base/Residual/Gate，但**都没参与到指标提升里**，fuse bypass 路径完全绕过它们。
- **c0fc744d**（同日，紧邻 f2ea2eb0）："你看整张脸就糊了 压根没学会变化"，用户在助手回复前即 `[Request interrupted by user]`，未进入诊断。

### 2.6 2026-04-29 一日：上下文探测 + 杂活，无实质代码进展

- **cff4b6f2**（00:30）：用户问"你可以阅读之前窗口的聊天记录吗"，助手确认无法跨会话读取、memory 目录为空，给出三种恢复上下文的路径（粘贴 / 读现有 md / 存 memory），用户未回。
- **a9b2bfc7**（00:30）：OFF-TOPIC，RustDesk + FRP 内网穿透配置，与项目无关。
- **c1aa321b**（01:20）：空 dump，用户开头一句"我已经改了三处关键点"但正文未捕获。
- **a0c70eb0**（02:17）：用户让助手读 `.session-dump-2.txt` + 61 个 md，助手提了"并行两个 Explore subagent 分别摘要"方案立即被打断，未执行。
- **6af9c38f**（06:25）：空 dump。

### 2.7 Fuse v7–v11 + TG-native clean 协议 + GT-fallback 揭穿（2026-05-12，eab0bb52）

- **目标**：在 V4 + V29.7 baseline 上继续推 fuse 训练改进，验证 temporal / phoneme / lip-region loss 是否真带来提升；解决 apex 帧"嘴部错位 / 上唇渲染错 / 接缝"；评估当前指标可否作为论文数字；最终被用户点醒：项目必须支持任意新音频 → 回到 TG 原版 self-contained 渲染协议。
- **关键修改**：
  - `train_fuse_v7.py` + `scripts/train_fuse_v7.sh`：v4 + short-scale temporal patch loss（Δpred(t,t+1) ≈ ΔGT），`frame_id_to_cam` 用 `cam.talking_dict['img_id']`，`TG_MAX_TRAIN_FRAMES=4000` 确保 (t,t+1) 配对。
  - `train_fuse_v8.py`：v7 + Hierarchical Phoneme-Aware Velocity Matching（JS-div 计算 phoneme boundary，中程 k∈{3,5} 在边界处 ×3 加权；长程 k∈{8,13} 因 `talking_dict` 缺 `lips_rect` 未触发）。
  - `train_fuse_v9.py` / `v9.1`：alpha hinge + lip-region L1+LPIPS recon（修了 squeeze bug + NaN guard + ckpt 路径）。
  - `train_fuse_v10.py` / `v10.1`：V4 基础 + head-only L1×2 + L2×1 + SSIM×0.4 + 全帧 LPIPS 1.0 + lip-bbox L1×3（不带 LPIPS，避开 v9.1 颜色错位陷阱）。原 V10 全帧 L2 把 face Gaussians 拉去画 torso → PSNR 跌到 7；V10.1 限制在 head_mask 内修复。
  - `train_fuse_v11.py`：V10 减 lip_L1，把 L1/L2/SSIM 限制在 non-mouth head_mask 内，验证"嘴部 pixel loss 推 face 投机加厚填充"假说。
  - `scripts/render_multi_au.py`：加 `--seq_alpha_mode {v29.7, soft, raw, smart}` + `--seam_feather`。smart 三段分割：`hard_hole`（GT 填）/ `edge_zone`（pred 自洽）/ `dense`（pred）；smart3 修 mouth ROI bbox fallback；smart4 调阈值 0.55/0.7 找平衡。
  - **新建 `synthesize_fuse_clean.py`**：完全 TG-native 渲染协议（[0,0,0] 黑底减法 + torso bg，无 GT-fallback），用于验证模型在"无 GT 抄袭"协议下的真实质量。
- **结论（关键）**：
  - **v29.7 模式 GT-fallback 系统性虚高 PSNR ~2.1 dB**：V7 v29.7=35.89 → V7 soft=33.75；V10 v29.7=36.70 → V10 soft=34.74。**嘴中心 100% 是 `cam.original_image`（测试帧 GT）复制粘贴的像素**，模型自身在嘴部 alpha≈0。
  - V7（HVM short）在 soft 公平评估下是最稳赢家（LMD/AOC/mouth-LPIPS 全胜 V4）；V8 phoneme prior 的 +1.8% AOC 优势在 soft 模式下反向（0.586→0.549），**说明是 GT-fallback 假象**。
  - V10 v29.7 数字漂亮但同样借自 GT-fallback；V11 ≈ V10（差异 0.01 量级），**lip_L1 不是 face 投机加厚的元凶**。
  - 用户最终点出致命问题：v29.7 / smart 等 alpha 后处理路径**只对复现测试集有效，对新音频生成 100% 失效**。这条路线（V4-V11 + render hack）**不能写论文**。
  - TG-native clean 协议下 5 ckpt 真实指标：PSNR full=7.27（torso bg vs 真背景错位导致），但 mouth-PSNR/SSIM 显示 V10 最佳（21.36/0.7219），V11 mouth-LPIPS 最低 0.2454，aperture V10 达 0.937（face 自己撑起来的真实开合）。
  - **根因诊断（决定性）**：`train_mouth.py` 用 `gt*mouth_mask + bg*~mouth_mask` 训练，而 face_parsing 给的 `mouth_mask` 在 apex 帧只覆盖 **52–563 像素（0.02–0.2%）**，mouth Gaussians 因此只学画牙齿小区域，嘴中心 alpha→0 → 黑洞 → v29.7 才需要 GT 抄袭兜底。

### 2.8 非主线会话（2026-05-12 13:43–14:06）

22 份会话笔记集中爆发，全部是流水线自身（"把 dump 转写为笔记"）的空 dump 输出，无项目信息。UUID 列表见阶段 1 清单的 §19–40。**不引用。**

### 2.9 跨主体 head 移植 + mouth 暴力 prune 根因定位 + Fix A/B/C 落地（2026-05-12）

2026-05-12 — Fix A+B+C 落地：切断 macron→obama/may head_prior 移植 + 软化 mouth_v30 绿背景 prune + may_v30 重建
目标：
- 修 #1：obama/may fuse 阶段一直把 macron-V17 的 phoneme/albedo/aperture_aux head 当 warm-start，跨主体分布错位导致 LMD 飘 / aperture 偏低。
- 修 #2：`train_mouth_v30.py` 在 densify 后用纯绿背景色直接把命中 bg_color_mask 的 Gaussian `_opacity:=0.1` + `_scaling/=10`，把高开嘴瞬间穿越绿背景的合法 lip Gaussian 当场清零，网络学到 "别动" → 推理嘴不张。
- 修 #3：`macrontest/may_v30/cfg_args` 含 `au_editor_mode=True`（已废弃路径残留），必须从零重建。
关键修改：
- `scripts/rebuild_fuse_obama_may.sh:13,55` — `fuse_iters=5000→15000`、`--head_prior ""`（让 `build_fuse_v30_init.py:73` 的 truthy 检查跳过 macron warm-start，phoneme/albedo/aperture_aux head 走 fresh init）。
- `scripts/train_subject_anticheat_ablation.sh:19,54` — 同步默认 `fuse_iters=15000` + `--head_prior ""`（这是 `cross_subject_chain.sh` 重建 may 时实际跑的脚本）。
- `train_mouth_v30.py:531-534` — 保留 `xyz_gradient_accum[bg_color_mask] /= 2`（让 densify_and_prune 软淘汰），注释掉 `_opacity` 与 `_scaling` 的直接 mutation。
- `macrontest/may_v30/` → `macrontest/may_v30_DELETED_20260512_171602/`（2.7G，重命名而非 rm，可回滚；脚本 `[ -f ... ]` 守卫看到空路径会触发完整 face→mouth→fuse 重建）。
- `D`（subject-specific aperture/arc/dino 微调）与 `E`（au.csv schema 防御）按用户标记"可选/可不做"未落地。
命令 / 输出目录：
- 验证：`python -m py_compile train_mouth_v30.py` ✓、`bash -n` 三个脚本 ✓。
- 后续待跑（用户决定）：`bash scripts/rebuild_fuse_obama_may.sh`（obama 仅 fuse 重建，~1.5h）+ `bash scripts/cross_subject_chain.sh`（may 全链 ~3.5h；obama 段会被 `[ -f ]` 守卫跳过 face/mouth）；之后 `synthesize_fuse_clean.py` + `eval_full.py` + `eval_lipsync.py` 看 obama LMD 是否回落到 macron LMD × 1.3 内、may aperture > 0.7。
结论 / 数字：
- 训练未启动，无新指标。代码落地 + 语法检查通过。
卡点或下一步：
1. 用户决定何时启动训练（rebuild_fuse_obama_may.sh 对 obama 即可见效；may 需要 cross_subject_chain.sh 全链）。
2. 训练完跑论文路径 `synthesize_fuse_clean.py` 看 obama/may 嘴中心是否还黑（v30 mouth 黑洞与 v30 本身 mouth_mask 范围未改，根因 #1（macron-v4 mouth_mask 太小）依然存在，但 Fix B 的软 prune 减少了二次伤害；如仍黑则需 `train_mouth_v2.py` 真扩 mask 方案）。
3. 跨主体表格若数字回不来，再评估 Fix D（per-subject aperture/arc/dino ×1.2）。
4. 旧 `macrontest/may_v30_DELETED_20260512_171602/` 在确认重建数字 OK 之后再 `rm -rf` 释放 2.7G。

2026-05-12 evening — obama 段完成：Fix A 单独对 noDino variant 解锁 +2.17 dB / -6.09 LMD，其它 variant 被 DinoV2 掩盖污染基本不动
目标：拿 obama 4 fuse variants 公平对比，验证 Fix A 是否真有效。
关键修改：
- 启动 `bash scripts/cross_subject_chain.sh`（PID 1959408，log `logs/cross_subject_chain_FixABC_20260512_172221.log`），obama 段 17:22 → 20:52（3.5h，比估的 6h 快）；进 may 段在跑。
- 旁路并行跑 `synthesize_fuse_clean.py --max_frames 128` 给 obama V30g_full 拿 quick 数字（论文路径 clean 协议，无 GT-fallback）。
- 全集 727/728 帧公平对比读取 `output_viz/obama_v30{f,g,g_noArc,g_noDino}_eval.txt` vs `_preFixA_20260512_171800.txt`。
命令 / 输出目录：
- 关键 ckpt：`macrontest/obama_v30/chkpnt_fuse_v30{f,g,g_noArc,g_noDino}_latest.pth`（全部 17:22 之后的新版，preFixA 备份在 `_preFixA_20260512_171800.pth`）。
- Quick eval 输出：`output/obama_v30g_clean_FixABC_eval/seq_test/`（128 帧）。
- Chain 自带全集 render+eval：`macrontest/obama_v30/render_v30{f,g,g_noArc,g_noDino}_full/seq_test/` 728 帧。
结论 / 数字（全集 727 vs 728 帧公平对比，Fix A before → after）：
- v30f: PSNR 33.14→33.16 (+0.02)，LMD 8.36→8.11 (-0.25)。
- v30g_full: PSNR 33.13→33.14 (+0.01)，LMD 7.53→7.62 (持平)。**LMD 门槛 4.17 未达成（仍 7.62）**。
- v30g_noArc: PSNR 33.12→33.13 (+0.01)，LMD 8.07→7.91 (-0.16)。
- **v30g_noDino: PSNR 32.44→34.62 (+2.17 dB)，LMD 10.96→4.87 (-6.09)，LPIPS-vgg 0.0794→0.0613 (-0.018)。** 是唯一显著跳变的 variant，**LMD 接近 macron 基线 3.21（差 1.66）**。
- 解读：带 DinoV2 时，深层语义 loss 掩盖了 macron→obama head_prior 污染；不带 DinoV2 时 Fix A 让 phoneme/albedo/aperture_aux 在 obama 自己分布上 fresh-init 重学 → 解锁巨大潜力。**这是 anti-cheat ablation 表里有意思的新发现**：DinoV2 既是稳健器也是天花板。
卡点或下一步：
1. v30g_full LMD 7.62 还远没回到 4.17 门槛 — Fix B 没对 obama 的 mouth_v30 生效（obama mouth ckpt 是 Fix B 之前训的）。要真修 obama mouth，需要：搬走 obama 的 `chkpnt_mouth_v30_latest.pth` + 重训 mouth_v30 50k + 重建 fuse 4 variants ≈ ~5h。建议等 may 跑完再决定。
2. may 段 20:52 开始（face_v30 25k → mouth_v30 50k → 4 fuse → render+eval），Fix B 第一次真正生效，预计 ~3-4h 后出数字。
3. 论文 ablation 表新发现可入正文："Fix A（cross-subject head_prior 切除）单独对带 DinoV2 variant 不显著，但对 noDino variant 解锁 +2.17 dB / -6.09 LMD，证明跨主体 head_prior 是被 DinoV2 掩盖的隐性污染源"。
4. obama V30g_full quick 128 帧 eval 数字（PSNR 33.65 / SSIM 0.9586 / LPIPS-vgg 0.0715 / LMD 7.90）与全集 728 帧（PSNR 33.14 / LMD 7.62）有 ~0.5 dB / -0.3 LMD 差异，说明前 128 帧偏难一些，**未来 quick eval 数字要打折看**。

---

## 3. 关键文件清单

> 仅列入笔记中明确提到过的文件。文件实际内容请用 git/Read 现场核查（部分文件名是 session 内新建/重写，但当前仓库是否就绪未验证 → "待确认"）。

### 3.1 训练入口

| 文件 | 作用 | 出处 |
|---|---|---|
| `train_au_editor.py` | 单 AU45 闭眼编辑器（已弃用） | a1022478 §2.1 |
| `train_multi_au.py` | 17-AU 多通道 deformer 训练（主路径） | a1022478 / 268ed005 / d96e7fd6 / f2ea2eb0 |
| `train_fuse_v2/v3/v4/v5/v6.py` | fuse pipeline 三件套迭代；**v4 是当前 fuse 基线** | f2ea2eb0 |
| `train_fuse_v7.py` ~ `v11.py` | v7 temporal patch / v8 phoneme HVM / v9 lip recon / v10 head-only / v11 减 lip_L1 | eab0bb52 |
| `train_mouth.py` | mouth 阶段训练；**mouth_mask 太小是黑洞根因（待修）** | eab0bb52 |
| `train_mouth_v4.py` / `train_face_v4.py` | 把 v4 LPIPS 改造下沉，因 6 次卡死被放弃 | f2ea2eb0 |

### 3.2 模型

| 文件 | 作用 | 出处 |
|---|---|---|
| `models/multi_au_deformer.py` | hashgrid + 5 区域 attention + 独立 d_xyz/d_scale/d_rot/d_opa head；forward 末乘 `face_mask_3d` 硬门控 | a1022478 / 268ed005 |
| `models/au_transformer_deformer.py` | cross-attention 自动发现 AU↔区域；entropy 双向 hinge；head 小幅非零初始化 | a1022478 / 268ed005 |
| `models/tg_style_deformer.py` | v16 起照抄 TG 结构；v17.1 `val_proj(tanh(au_v))` + `coef.clamp(0.8,1.25)` | b7fffc65 / d96e7fd6 |
| `models/mouth_deformer.py` | `xy_compress 0.3 / z_compress 0.6 / scale_scale 6e-2 / rot_scale 6e-2`；Base/Residual/Gate（论文级创新但未参与指标） | f2ea2eb0 |

### 3.3 渲染 / 评估脚本

| 文件 | 作用 | 出处 |
|---|---|---|
| `scripts/render_multi_au.py` | 主渲染入口；`--seq_alpha_mode {v29.7,soft,raw,smart}`；`--seam_feather`；`--composite_mode` 默认 `face_over_mouth`；`--mouth_gain` / `--mouth_deformer_gain` / `--canonical_from_fuse`；自动检测 hubert vs ds | f2ea2eb0 / eab0bb52 |
| `scripts/render_au45_sweep.py` | 单 AU45 sweep（au_editor 路径，已弃用） | a1022478 / d96e7fd6 |
| `synthesize_fuse_clean.py` | **TG-native clean 渲染协议**（无 GT-fallback，论文路线必经路径） | eab0bb52 |
| `scripts/eval_lipsync.py` | LMD / AOC / Lip-LPIPS / aperture ratio | f2ea2eb0 |
| `scripts/eval_full.py` | PSNR/SSIM/LPIPS full+mouth ROI、LMD/AOC、可选 AUE 调本地 OpenFace | f2ea2eb0 |
| `scripts/eval_au_controllability.py` | 17 AU sweep + OpenFace 反测；C[i,i]≈0 证伪了独立可控性 | f2ea2eb0 |

### 3.4 训练脚本

| 文件 | 作用 |
|---|---|
| `scripts/train_fullclose_curriculum_20k.sh` / `_cursorfix.sh` | au_editor 课程训练（已弃用） |
| `scripts/train_multi_au.sh` / `_cursorfix.sh` | 多 AU 训练主脚本，每次需新 OUT_DIR、不可 resume |
| `scripts/train_fuse_v7.sh` | fuse v7 入口 |
| `scripts/train_xx.sh` | 当前 git status 显示 M，但具体改动未在笔记中体现，**待确认** |

### 3.5 副本 / 残留

- 仓库根目录散落 ~60 份 `AU*_*.md` / `CANONICAL_*.md` / `CORRECT_AU_STRATEGY.md` 等历史方案文档，全部 untracked。多份**结论已被后续 session 推翻**（例如 `AU_TO_EXPRESSION_DEFORMER_SOLUTION.md` 早于 17-AU 独立可控性证伪）。**不要把这些 md 作为权威**。
- 仓库根目录有数十个以 `--` 开头的伪文件/目录（`--au45_cap` / `--model_path` / `--source_path` / `-s` 等），是某次 shell 命令把 argparse flag 当成 redirect 目标创建出来的，需用户授权后清理。
- 副本 worktree：`/home/labliu/.cursor/worktrees/TalkingGaussian__SSH__labliu_100.102.101.127_/qpz`。多个 session 在两份代码间编辑（特别是 multi_au 路径），存在 merge 风险。
- `CLAUDE.md.bak` 与 `CLAUDE.md` 并存。

---

## 4. 关键技术决策（按时间顺序）

| 时间 / Session | 决策 | 理由 |
|---|---|---|
| 2026-04-16 a1022478 | **abs → signed close-y loss** | abs 不罚反向，半数上飞半数下沉也能骗平均 |
| 2026-04-16 a1022478 | **`eye_core_static_w` 只锁 AU45 通路，不动 d_xyz** | 不能把 motion_net 的 lip-sync 一起冻住 |
| 2026-04-16 a1022478 | **`blink_active_mask = eye_mask * (1 - core)`，magnitude loss 在 eyelid 平均** | frozen 眼球稀释分母，导致 eyelid 必须过冲才能命中目标 |
| 2026-04-16 a1022478 | **放弃单 AU45 editor，启动 multi_au_deformer** | 离线诊断证实 611 个 active Gaussian 全部塞到 clip 上限，frozen canonical 没有上眼睑 Gaussian 可滑 |
| 2026-04-16 a1022478 | `face_mask_3d` 硬门控 forward 输出 | mouth 让给 motion_net，避免衣服 / 嘴部污染 |
| 2026-04-19 268ed005 | **attention entropy 双向 hinge**（target=log(3)） | step 18400 attention 塌成 one-hot 后永久 NaN |
| 2026-04-19 268ed005 | **head 改小幅非零初始化** | 零初始化死梯度，渲染图根本没变化 |
| 2026-04-19 268ed005 | **proportional response + AU 幅度加权 + disp_floor** | AU 训练分布只到 [−0.15,+0.50]，sweep 到 1.5 是 3–10× 外推，模型按训练幅度学幅度过小 |
| 2026-04-19 268ed005 | v8 直接炸图后，所有大位移旋钮 ×1/5 (`v8b_gentle`) | "鼓励大位移"参数 vs "维持稳定" 必须按 1/5 步长调，不能一次放开 |
| 2026-04-21 b7fffc65 | **加 scoped PSNR loss**（face_mask_2d ∈ FP[1,2,3,4,5,10]） | 模型用低频肉色"假闭眼"骗 L1/SSIM；PSNR 在 mse→0 处梯度大 |
| 2026-04-21 b7fffc65 | tg_style_deformer v17 `tanh(au_v)` + `coef.clamp(0.8,1.25)` | AU45 推理到 3.0 时 transformer 在未见区域 coef 跌到 0.3-0.7 |
| 2026-04-23 d96e7fd6 | **`locality_w 0.3→0`**（取消 dxyz 5mm 上限） | 闭眼需要 ~15mm，5mm 卡住被迫用 d_scale 拉糊 |
| 2026-04-23 d96e7fd6 | **`frame_mul` 只乘光度 loss，不乘 drift reg** | 之前在 blink 帧用 5× 力把 canonical 拉回原位，抵消 v21 联合微调 |
| 2026-04-23 d96e7fd6 | v23 motion_net 冻结使用后回退（用 v22 + blink 过采样） | motion_net 与 TG 原始 canonical 联合训练，v21 微调过的 canonical 让 motion_net 输出不对位 |
| 2026-04-26 f2ea2eb0 | **`--composite_mode` 默认 `face_over_mouth`** | `mouth_over_face + mouth_punch=0.7` 是 ratio 暴涨到 1.6 的根因 |
| 2026-04-26 f2ea2eb0 | **seq_test 路径补传 fuse_* / tg_mouth_*** | 黑洞 = seq_test 走 legacy 分支导致嘴中心 alpha≈0 |
| 2026-04-26 f2ea2eb0 | bg `torso_imgs → cam.original_image` + alpha 3-pass erosion + 阈值 0.8/0.5 | 修 silhouette 紫边 + 嘴内侧彩斑 |
| 2026-04-26 f2ea2eb0 | **17-AU 独立可控性被证伪**（C[i,i]≈0） | 17 AU sweep + OpenFace 反测，差异主要是色彩噪声不是语义 AU motion |
| 2026-05-12 eab0bb52 | **v29.7 alpha 后处理路径不能写论文** | 测试帧能复现是因为嘴中心是 cam.original_image 复制；对新音频生成 100% 失效 |
| 2026-05-12 eab0bb52 | **TG-native clean 协议 = 论文唯一可接受路径** | 必须 `[0,0,0]` 黑底减法 + torso bg，无 GT-fallback |
| 2026-05-12 eab0bb52 | **根因定位：mouth_mask 在 apex 帧 0.02–0.2%** | mouth Gaussians 只学画牙齿小区域，整张嘴中心 alpha→0；修复路径 = train_mouth_v2.py 扩 mask 到 lips_rect + alpha hinge≥0.95 + init 350→1000 + 重训 fuse |

---

## 5. 走过的弯路与已排除方案

| 弯路 | 排除时间 | 排除理由 |
|---|---|---|
| **`.abs()` 形式的对称 loss**（response/cover/pair/high_abs/mono） | 2026-04-16 | 不罚反向，模型用"半数上飞半数下沉"骗均值 |
| **单 AU45 外挂 editor + frozen canonical**（train_au_editor.py 全家） | 2026-04-16 | 611 active Gaussian 全部塞到 clip 上限；canonical SH 存的是虹膜色；理论天花板 |
| **`disable_scale_branch` / `disable_dynamic_opacity`** | 2026-04-16 v4 | 单 xyz 通道做 14 mm 位移必然糊，必须三路打开 |
| **手工 region attention（5 区硬切）** | 2026-04-16 v4 transformer | 衣服 / 嘴部污染、几何 prior 太脆 |
| **head 零初始化** | 2026-04-19 v6 | 死梯度，渲染图全无变化 |
| **v8 一次性放开 prop_w 0.10 / floor_w 0.5 / au_*_scale 大幅** | 2026-04-19 v8 | 直接炸图"生成的都不是脸了"，必须 ×1/5 步长 (v8b_gentle) |
| **v17 transformer 输入不过 tanh** | 2026-04-21 | AU45 → 3.0 时落到未见区域 coef 跌到 0.3-0.7 |
| **v21 canonical 联合微调 + 仍用 motion_net** | 2026-04-23 v23 | motion_net 与 TG 原始 canonical 绑定，微调过的 canonical 让 motion_net 输出不对位 |
| **`locality_w=0.3 / dxyz_ceil=3e-3`** | 2026-04-23 v22 | 闭眼实测需要 ~9e-3，5 mm 上限不够 |
| **`frame_mul` 同时放大 drift reg** | 2026-04-23 v22 | 在 blink 帧 5× 力把 canonical 拉回原位 |
| **`composite_mode=mouth_over_face` + `mouth_punch=0.7`** | 2026-04-26 | ratio 暴涨到 1.6（嘴张过大变形） |
| **v27 non-fuse 训练 / v28 canonical_from_fuse 残差训练** | 2026-04-26 | 前者退化成常数张嘴；后者脸烂 LPIPS+67% |
| **v29 / v29.1 / v29.2 新 loss + 30k 延长 fuse 训练** | 2026-04-26 | 嘴透明 / 嘴过张 1.59；30k 指标全劣化 |
| **L2 全图 PSNR loss + LightingHead v5/v6** | 2026-04-26 | LightingHead 失败；全图 L2 把 face Gaussians 拉去画 torso |
| **v4full 完整重训 face+mouth（train_mouth_v4.py / train_face_v4.py）** | 2026-04-26 | 6 次重启全卡死；归因 limit train frames + AU25 严格筛选 while + 早期 densify_and_prune 把 init Gaussians 全 prune |
| **依赖 v29.7 / smart / soft alpha 后处理出论文数字** | 2026-05-12 | 嘴中心 100% 是 `cam.original_image` 复制；对新音频生成 100% 失效；GT-fallback 系统性虚高 PSNR ~2.1 dB |
| **靠扩 17-AU deformer 做独立 AU 编辑** | 2026-04-26 | OpenFace 反测 C[i,i]≈0，事实层面不成立；放大 au_gain 后差异是色彩噪声不是语义 motion |

---

## 6. 当前状态

### 6.1 最近一次实质工作的产出（2026-05-12 eab0bb52）

- 已有 ckpt：`output/{orig_tg, v4, v7, v10, v11}_clean_eval/`（5 个 fuse ckpt × `synthesize_fuse_clean.py` 输出），以及 `output/v{ver}_eval*/`、`output/orig_tg_eval*/`。
- 已有评估数据：
  - v29.7 协议：V4 PSNR 35.83 / V10 36.70（**虚高 ~2.1 dB**）。
  - soft 协议：V4 33.72 / V7 33.75 / V10 34.74。
  - TG-native clean 协议：PSNR full=7.27（torso bg 错位导致），mouth-PSNR V10=21.36 / mouth-SSIM V10=0.7219 / mouth-LPIPS V11=0.2454 / aperture V10=0.937。
- 主要可视化：`output/v8_compare/v4_v7_v8_compare_labeled.mp4`、`v10_apex_zoom_idx85.png`、`diagnostic_lip_seam_6col.png`、`v11_apex_no_fallback.png`、`v11_smart{3,4}_apex{18,33,58}.png`、`clean_6col_full_slow.mp4`、`clean_6col_mouth_slow.mp4`。

### 6.2 未解决问题（按优先级）

1. **mouth_mask 太小是黑洞根因**（eab0bb52）。修复路径计划：`train_mouth_v2.py` = `mouth_mask` 扩到 `lips_rect` + alpha hinge ≥0.95 + mouth init Gaussians 350→1000 + 重训 fuse；估计 ~30 min 工程。**尚未实施**。
2. `synthesize_fuse_clean.py` 中 motion_net_mouth 把 mouth opacity 推到 0（`opacity_raw=0.9756` 但渲染输出近黑），需进一步 instrument。
3. TG-native clean 协议下非头部背景 PSNR=7.27，需换 `cam.background` 或换 head-only metric 才能公平报数。
4. 论文 fairness 策略未拍板：报 soft 模式 ~34 dB（诚实但低于 IP-LAP 35.34）还是报 v29.7 + 声明 GT-fallback。**默认应选诚实路径（TG-native）**。
5. **17-AU 独立可控性事实不成立**：用户的 A1（坦白）/ A2（OpenFace 反向监督重训）/ B（工程改进）三选一未选定。论文 3 个卖点（per-AU spatial attention / cross-AU correlation transformer / linearity-preserving multiplicative fusion）写出来了但**未参与到指标提升**。
6. v23 motion_net 集成失败后的回退方案（v22 + K=3 blink 过采样）尚未给出落地 diff 或新 OUT_DIR。
7. v22 之后渲染表现是否真的解决 smudge 在 d96e7fd6 中没有最终结论。
8. canonical 缺 FP label 6 (eyelid) 这一**数据层面**根本问题未解决，目前都靠软约束 + 程序化窄带 mask + 颜色 SH 微调凑。
9. Sync-C / Sync-E 未集成（需 Wav2Lip SyncNet 权重）。
10. 是否 `cp chkpnt_fuse_v4_latest.pth → chkpnt_fuse_latest.pth` 把 v4 设永久 fuse 未拍板。
11. 仓库脏：~60 个 `--xxx` 伪文件、20+ 份 untracked 历史方案 md、`CLAUDE.md.bak`、`submodules/simple-knn` untracked。`gaussian_renderer/__init__.py` / `scene/dataset_readers.py` / `scripts/train_xx.sh` / `utils/camera_utils.py` 处于 M 状态但**改动归属与意图未在任何 session 中说明**——下次会话第一步应跑 `git diff` 定性。

### 6.3 最新假设

- **嘴部黑洞根因 = `train_mouth.py` 训练 mask 太小**（apex 帧 0.02–0.2%），不是渲染 bug。这是 eab0bb52 末尾的定性诊断，但**未通过 train_mouth_v2 重训验证**。
- **v29.7 协议永远不能用于新音频生成**，论文路径必须走 `synthesize_fuse_clean.py`。
- **17-AU 独立可控性**这条创新点在数据/方法层面就走不通，重训 OpenFace 反向监督是唯一抢救可能。

### 6.4 下一步计划（基于 eab0bb52 末态）

1. 实施 `train_mouth_v2.py` 扩 mask + alpha hinge + 大 init → 重训 fuse → 用 `synthesize_fuse_clean.py` 验证嘴中心是否还黑。
2. 把 V4 ckpt 路径明确（是否 `cp` 成永久 default），并清理 V5–V11 中除 V7/V10 之外的不必要 ckpt。
3. 决定论文报告协议（soft vs v29.7），写入 paper draft。
4. 决定 17-AU 创新点的处置（A1/A2/B 三选一）。

---

## 7. 命令 / 路径 / 输出目录命名规则

### 7.1 训练命令模板

```bash
# 必须先激活环境（多个 session 都会先 cd 进项目根）
conda activate talking_gaussian
cd /home/labliu/wangshiyuan/TalkingGaussian

# au_editor（已弃用）
OUT_DIR=output/au_editor/<name>      bash scripts/train_fullclose_curriculum_20k_cursorfix.sh

# multi_au（主路径）
OUT_DIR=output/multi_au/<name>       bash scripts/train_multi_au.sh
OUT_DIR=output/multi_au/<name>       bash scripts/train_multi_au_cursorfix.sh

# fuse 系列
OUT_DIR=...                          bash scripts/train_fuse_v7.sh

# 只渲染、不训练
RUN_RENDER=1 OUT_DIR=./output/multi_au/macron_v19 bash scripts/train_multi_au.sh
```

强制约定（多 session 重复出现）：**每轮必须用新 OUT_DIR，不可 resume**；每次脚本改动后跑一遍 `python -m py_compile <file>` 和 `bash -n scripts/<file>` 做语法体检。

### 7.2 渲染命令模板

```bash
python scripts/render_multi_au.py \
  --use_tg_fuse --use_tg_mouth \
  --seq_test_frames 128 --seq_stride 1 \
  --seq_alpha_mode {v29.7|soft|raw|smart} \
  --composite_mode face_over_mouth \
  --mouth_gain 1.2 \
  --bg_rgb 0 0 0 \
  ...

# TG-native clean（论文路径）
python synthesize_fuse_clean.py --ckpt <fuse ckpt> --out output/<name>_clean_eval/

# AU45 sweep（au_editor 路径，已弃用）
python scripts/render_au45_sweep.py ...

# 评估
python scripts/eval_lipsync.py --dir output/multi_au/<name>/render_eval128/seq_test
python scripts/eval_full.py    --dir <seq_test>
python scripts/eval_au_controllability.py ...
```

### 7.3 输出目录命名规则

| 路径前缀 | 内容 |
|---|---|
| `output/au_editor/<name>` | au_editor 路径（已弃用）。常见后缀 `macron_fullclose_curriculum_20k_<v>` |
| `output/multi_au/<name>` | multi_au 路径。常见后缀 `macron_v<n>[_face_locked\|_geom_loose\|_transformer\|_resume\|_fresh\|_live_head\|_prop_response\|_gentle\|_three_branch\|_sh_finetune\|_clean_eval]` |
| `output/v{ver}_eval*/` / `output/orig_tg_eval*/` | fuse 渲染评估输出 |
| `output/{orig_tg,v4,v7,v10,v11}_clean_eval/` | TG-native clean 协议渲染输出 |
| `output/v8_compare/`、`output/v4_compare/`、`output/v6_compare/` 等 | 视觉对比图、mp4 比较 |
| `output/au_sweep_visual/`、`output/v4_no_fringe/`、`output/v4_strong_fix/` | 单次实验视觉产物 |
| `macrontest/macron/`、`macrontest_v4/macron/` | full pipeline 重训产物（v4full 6 次卡死后保留） |

### 7.4 路径混用警告

- 项目存在主仓库 `/home/labliu/wangshiyuan/TalkingGaussian/` 与 worktree `/home/labliu/.cursor/worktrees/.../qpz/` 两份代码。multi_au 早期在 worktree 训练（其 `scripts/train_multi_au.sh` 的 `SOURCE_PATH`/`CANONICAL_MODEL` 改指主仓库以拿数据）。
- **下次会话开工前先 `git -C` 确认当前在哪一份代码上**，避免编辑落空。

### 7.5 文件命名后缀含义速查

- `_cursorfix` 结尾的脚本：Cursor IDE 兼容/桥接，功能与去 `_cursorfix` 版本等价。
- `_v29.7` / `_v29.1` 等点版本号在渲染参数命名里指 `--seq_alpha_mode` 不同模式。
- 训练 ckpt 命名：`chkpnt_fuse_v{N}_latest.pth`、`multi_au_step{N}.pth`、`chkpnt_fuse_latest.pth`（"当前生效"指针，目前事实指向 v4，是否永久化未定）。

---

## 附：本文件维护规则

- 本文件由 `.session-notes/*.md` 综合生成，不要直接当成单一权威；任何与代码现状冲突的地方以代码为准。
- 新增 session 时：按时间插入 §2，并在 §4 / §5 / §6 同步更新一行。
- 永远不要把空 dump（§2.8 那 22 份）当成事实来源。
- 不引用根目录散落的 `AU45_*.md` / `CANONICAL_*.md` / `CORRECT_AU_STRATEGY.md` 等历史方案 md——它们的结论多已被后续 session 推翻。
