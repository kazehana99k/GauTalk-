# CLAUDE.md

完整项目历史见 @PROJECT_HISTORY.md（仅在需要回溯具体决策、找历史 ckpt 路径、确认走过的弯路时再读）。

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project Context（TalkingGaussian / AU editor / fuse）

> 演进细节、决策依据、走过的弯路见 `PROJECT_HISTORY.md`。本节只放新会话开工最必要的信息。本节自 2026-05-12 基于 `.session-notes/*.md` 综合生成。

## 项目一句话

3D Gaussian Splatting talking head（基于 TalkingGaussian），目标：用 AU + audio 驱动 macron 数据集的头像生成，**支持任意新音频**（不靠 GT-fallback 抄像素）。

## 当前 baseline

- **v4 fuse ckpt + v29.7 渲染**：PSNR 35.83 / SSIM 0.9665 / LPIPS-vgg 0.0434 / LMD 0.0369 / AOC 0.600 / aperture 0.840。
- **重要**：上述数字依赖 v29.7 alpha 后处理把 `cam.original_image` 作为嘴中心 fallback，**对新音频生成 100% 失效，不能写论文**。论文路径必须走 `synthesize_fuse_clean.py`（TG-native clean 协议）。
- soft 公平协议下 V4=33.72 / V7=33.75 / V10=34.74（虚高 ~2.1 dB 被剥掉后的真值）。
- TG-native clean 协议下 mouth-PSNR V10=21.36 / mouth-LPIPS V11=0.2454 / aperture V10=0.937。

## 关键根因（已定位、未修复）

`train_mouth.py` 用 `gt*mouth_mask + bg*~mouth_mask` 训练，而 `face_parsing` 给的 `mouth_mask` 在 apex 帧只覆盖 **52–563 像素（0.02–0.2%）**。后果：mouth Gaussians 只学画牙齿，嘴中心 alpha→0 → 黑洞 → v29.7 才需要 GT 抄袭兜底。

**修复方案（未实施）**：新建 `train_mouth_v2.py`，把 `mouth_mask` 扩到 `lips_rect` + alpha hinge ≥0.95 + mouth init Gaussians 350→1000 + 重训 fuse。

## 已确认排除的方案（不要重试）

- 单 AU45 外挂 editor + frozen canonical（`train_au_editor.py` 全家）：611 active Gaussian 全部塞到 clip 上限，理论天花板。
- 依赖 v29.7 / smart / soft alpha 后处理出论文数字：测试帧能复现是因为嘴中心是 `cam.original_image` 复制。
- 17-AU 独立可控性（per-AU spatial attention / cross-AU transformer 单独可控）：OpenFace 反测 C[i,i]≈0，事实层面不成立。
- v21 canonical 联合微调 + 仍用 motion_net（v23）：motion_net 与原始 canonical 绑定，微调过的 canonical 让它输出不对位。
- v4full 完整重训 face+mouth（`train_mouth_v4.py` / `train_face_v4.py`）：6 次重启全卡死。

## 关键文件

- `train_multi_au.py` — 17-AU 多通道 deformer 训练主入口。
- `train_fuse_v7.py` ~ `v11.py` — fuse pipeline 迭代，**v4 是当前 fuse 基线**（在 `train_fuse_v4.py`）。
- `train_mouth.py` — mouth 阶段；**mouth_mask 太小是黑洞根因，待修**。
- `models/multi_au_deformer.py` — hashgrid + 5 区域 attention + face_mask_3d 硬门控。
- `models/au_transformer_deformer.py` — cross-attention；entropy 双向 hinge `target=log(3)`；head 小幅非零初始化（**不要零初始化**）。
- `models/tg_style_deformer.py` — v17.1 `val_proj(tanh(au_v))` + `coef.clamp(0.8, 1.25)`（**不要去掉 tanh**）。
- `models/mouth_deformer.py` — `xy_compress=0.3 / z_compress=0.6 / scale_scale=6e-2 / rot_scale=6e-2`。
- `scripts/render_multi_au.py` — `--seq_alpha_mode {v29.7,soft,raw,smart}`、`--composite_mode face_over_mouth`（**默认值，不要改回 mouth_over_face**）、`--mouth_gain 1.2`、auto-detect hubert/ds。
- `synthesize_fuse_clean.py` — **TG-native clean 渲染（论文唯一可接受路径）**。
- `scripts/eval_lipsync.py` / `eval_full.py` / `eval_au_controllability.py` — 指标。

## 命令模板

```bash
conda activate talking_gaussian
cd /home/labliu/wangshiyuan/TalkingGaussian

# multi_au 训练（每轮必须新 OUT_DIR，不可 resume）
OUT_DIR=output/multi_au/<name> bash scripts/train_multi_au.sh

# fuse v7
bash scripts/train_fuse_v7.sh

# 渲染（公平）
python scripts/render_multi_au.py --use_tg_fuse --use_tg_mouth \
  --seq_test_frames 128 --seq_stride 1 \
  --seq_alpha_mode soft --composite_mode face_over_mouth \
  --mouth_gain 1.2

# 论文路径渲染
python synthesize_fuse_clean.py --ckpt <fuse ckpt> --out output/<name>_clean_eval/

# 评估
python scripts/eval_full.py    --dir output/multi_au/<name>/render_eval128/seq_test
python scripts/eval_lipsync.py --dir output/multi_au/<name>/render_eval128/seq_test
```

体检（每次脚本改动）：`python -m py_compile <file>` 与 `bash -n scripts/<file>`。

## OUT_DIR 命名

主路径 `output/multi_au/<name>`；TG-native clean 输出 `output/{...}_clean_eval/`；其它命名规则与历史后缀速查见 PROJECT_HISTORY.md §7.3。

## 下一步（按优先级）

1. 实施 `train_mouth_v2.py`（扩 mask + alpha hinge + init 350→1000）→ 重训 fuse → 用 `synthesize_fuse_clean.py` 验证嘴中心是否还黑。
2. 决定 17-AU 创新点的处置（A1 论文坦白 / A2 OpenFace 反向监督重训 / B 工程改进）——用户三选一未拍板。
3. 决定论文报告协议（soft ~34 dB 诚实但低于 IP-LAP 35.34 vs v29.7 + 声明 GT-fallback）——默认建议诚实路径。
4. 把 V4 fuse ckpt 路径明确（是否 `cp chkpnt_fuse_v4_latest.pth chkpnt_fuse_latest.pth` 永久化）。

## 仓库脏状态（开工前先确认）

- `gaussian_renderer/__init__.py` / `scene/dataset_readers.py` / `scripts/train_xx.sh` / `utils/camera_utils.py` 处于 M，**任何 session 笔记都未说明改动归属与意图**。**开工第一件事跑 `git diff` 定性**。
- 仓库根目录有 ~60 个以 `--` 开头的伪文件/目录（`--au45_cap` / `--model_path` / `-s` 等），是 shell 把 argparse flag 当 redirect 目标创建的。**需用户授权后才能 `rm`**。
- ~20 份 untracked `AU*_*.md` / `CANONICAL_*.md` / `CORRECT_AU_STRATEGY.md` 历史方案文档：**不要把这些 md 当权威**，结论多已被后续 session 推翻；权威只看 `PROJECT_HISTORY.md` + 代码。
- `CLAUDE.md.bak` 与本文件并存。`submodules/simple-knn` untracked。
- 主仓库 `/home/labliu/wangshiyuan/TalkingGaussian/` 与 worktree `/home/labliu/.cursor/worktrees/.../qpz/` 两份代码并存。**编辑前 `pwd` 确认在哪份**。

## 硬约定（多 session 重复确认过的）

- 每次 multi_au 训练用新 OUT_DIR，不要 resume。
- 不要把 head 模块零初始化（会死梯度）。
- transformer 输入必须过 `tanh`（保证外推区间响应不塌）。
- `--composite_mode` 默认 `face_over_mouth`（`mouth_over_face` 会让 ratio 暴涨到 1.6）。
- "鼓励大位移"的旋钮（`prop_w`/`floor_w`/`au_*_scale`）一次最多调到原值 ×1/5，不要直接放开。

## 文档维护规则

完成下列任一情况后，**追加一段记录到 PROJECT_HISTORY.md 的时间线末尾**（不要写到 CLAUDE.md）：
- 跑通一组训练/评估、得到关键新数字（PSNR / LPIPS / AOC / LMD 等）
- 完成某个未解决问题的修复（例如 train_mouth_v2 落地）
- 引入新的关键文件、脚本、ckpt 路径
- 改变下一步技术决策或推翻之前的假设
- 单次会话累计 ≥5 处重要修改

追加方式（重要：append 不是 overwrite）：

1. Read PROJECT_HISTORY.md，定位"演进时间线"那一节的末尾
2. 用 Edit 工具，old_str 取时间线末尾几行（确保唯一），new_str = 旧内容 + 新条目
3. 新条目格式：

```
YYYY-MM-DD — <一句话主题>
目标：
关键修改：（文件路径 + 改了什么）
命令 / 输出目录：
结论 / 数字：
卡点或下一步：
```

4. 不要修改时间线之前的小节（项目背景、关键文件清单等）
5. 不要删改已有的历史条目
6. 只有当 CLAUDE.md 的"当前状态 / 未解决问题 / 下一步"小节真的过时了，才同步更新 CLAUDE.md 对应小节；其它情况只动 PROJECT_HISTORY.md

主动触发时机：
- 检测到上述节点之一 → **主动询问** "本次进展该入 PROJECT_HISTORY.md 时间线了，建议追加：<草稿>。要写入吗？"
- 用户打 `/update-state` 或说"记录一下" → 直接走流程，不再问

CLAUDE.md 始终 ≤200 行；超过的内容下沉到 PROJECT_HISTORY.md。
