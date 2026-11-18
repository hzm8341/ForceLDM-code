# Decisions

- 2026-11-16 — Reproduce on the local single RTX 5090 (32GB) by adding a new LoRA config (`pi0_forceldm_lora`), sharing the GPU with the user's running Isaac Sim RL jobs; the existing `pi0_forceldm` configs stay unchanged. Reason: the full fine-tune needs ~55GiB (recorded on an A800), which cannot fit. Decided by: user.
- 2026-11-16 — Evaluation: hold out ~5% of episodes per task as a validation split (excluded from training); report the student policy's sampled action-chunk MSE/L1 on it, compared against zero-shot pi0_base. Rejected: training-loss-only evaluation. Decided by: user.
- 2026-11-16 — First run: 5k steps to check the trend; extend toward 30k only if the results look reasonable. Decided by: user.
- 2026-11-16 — 建议（未确认）：评估对照中去掉“pi0_base 零样本”（pi0_base 不认识该机器人动作空间，ForceLDM 新增头为随机初始化，数字无意义），改为“保持当前状态”基线 + 各 checkpoint（1k…5k 步）曲线。Proposed by: Claude.
- 2026-11-16 — 用户确认上一条：评估对照改为“保持当前状态”基线 + 各 checkpoint 曲线，不做 pi0_base 零样本对比（覆盖第 2 条中的对照方式）。下载完成后直接开始训练，无需再确认。Decided by: user.
- 2026-11-16 — 与 rsi_rl forge-p8 RL 流水线共享 GPU 失败：ForceLDM 3 步测试使空闲显存跌到 242MiB，触发 P8 三个任务在 ep≈310 被停并重试；正式训练（batch 16）也因 OOM 退出。改为：等 P8 流水线结束（p8.log 出现 P8_ALL_DONE）后自动启动 ForceLDM 5k 步训练，独占 GPU、batch 16；推翻第 1 条中的“与 RL 任务共享 GPU”。Decided by: user.
- 2026-11-17 — 5k 步 LoRA 结果：训练集上好于“保持当前状态”基线，验证集上未超过（夹爪维泛化差，主要在削黄瓜任务）。用户决定：代码提交并推送到 https://github.com/hzm8341/ForceLDM-code；先排查削黄瓜夹爪问题；GPU 空闲时再继续训练，可在晚上 9 点（21:00）之后开始使用 GPU。Decided by: user.
- 2026-11-17 — 削黄瓜夹爪问题定位：验证集 ep473 为坏录制（3579 帧、机械臂全程不动、夹爪 6.7s 后张开静止），训练集 ep450 为 21 帧空录制；其余 episode 正常。不改划分（避免重算归一化），评估时用 `--exclude-episodes 473` 另报一份。续训：21:00 之后且 GPU 空闲时，从 4999 步续训到 30k（LR 调度本就是 30k 余弦，等价于直接训 30k），每 5000 步存 checkpoint。依据用户“晚上 9 点开始使用 GPU”的指示；排除 ep473 的做法为建议（未确认）。
- 2026-11-17 — 用户确认上一条：评估时除完整验证集外，另报一份排除 ep473 的结果（`--exclude-episodes 473`），划分不变；eval_offline.py 的 episode 过滤/逐样本误差改动与决策记录提交并推送到 hzm8341/ForceLDM-code。排除 ep473 后 4999 步整体 L1 0.0033 vs 基线 0.0030，削黄瓜夹爪与基线持平（0.0045）。Decided by: user.
- 2026-11-17 — 续训目标维持 30k 步（否掉提前到 20k 步结束的方案：20k 时余弦 LR 未降到底）。若 21:00 开始，预计次日约 06:30 训练+评估结束、释放 GPU。Decided by: user.
- 2026-11-18 — 30k 步结果（记录，非新决定）：排除 ep473 时 29999 步验证 L1 0.0021 vs 基线 0.0030、t17-32 0.0030 vs 0.0043，6 个任务均优于基线；含 ep473 时 0.0035 vs 0.0031。keep_period=5000 使 1000–4999 步 checkpoint 被清理。用户决定推送到 hzm8341/ForceLDM-code（results/ 按仓库 .gitignore 不提交）。Decided by: user.
