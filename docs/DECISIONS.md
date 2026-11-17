# Decisions

- 2026-11-16 — Reproduce on the local single RTX 5090 (32GB) by adding a new LoRA config (`pi0_forceldm_lora`), sharing the GPU with the user's running Isaac Sim RL jobs; the existing `pi0_forceldm` configs stay unchanged. Reason: the full fine-tune needs ~55GiB (recorded on an A800), which cannot fit. Decided by: user.
- 2026-11-16 — Evaluation: hold out ~5% of episodes per task as a validation split (excluded from training); report the student policy's sampled action-chunk MSE/L1 on it, compared against zero-shot pi0_base. Rejected: training-loss-only evaluation. Decided by: user.
- 2026-11-16 — First run: 5k steps to check the trend; extend toward 30k only if the results look reasonable. Decided by: user.
- 2026-11-16 — 建议（未确认）：评估对照中去掉“pi0_base 零样本”（pi0_base 不认识该机器人动作空间，ForceLDM 新增头为随机初始化，数字无意义），改为“保持当前状态”基线 + 各 checkpoint（1k…5k 步）曲线。Proposed by: Claude.
- 2026-11-16 — 用户确认上一条：评估对照改为“保持当前状态”基线 + 各 checkpoint 曲线，不做 pi0_base 零样本对比（覆盖第 2 条中的对照方式）。下载完成后直接开始训练，无需再确认。Decided by: user.
- 2026-11-16 — 与 rsi_rl forge-p8 RL 流水线共享 GPU 失败：ForceLDM 3 步测试使空闲显存跌到 242MiB，触发 P8 三个任务在 ep≈310 被停并重试；正式训练（batch 16）也因 OOM 退出。改为：等 P8 流水线结束（p8.log 出现 P8_ALL_DONE）后自动启动 ForceLDM 5k 步训练，独占 GPU、batch 16；推翻第 1 条中的“与 RL 任务共享 GPU”。Decided by: user.
- 2026-11-17 — 5k 步 LoRA 结果：训练集上好于“保持当前状态”基线，验证集上未超过（夹爪维泛化差，主要在削黄瓜任务）。用户决定：代码提交并推送到 https://github.com/hzm8341/ForceLDM-code；先排查削黄瓜夹爪问题；GPU 空闲时再继续训练，可在晚上 9 点（21:00）之后开始使用 GPU。Decided by: user.
