# RL 强化学习

在 AutoDL 从仓库根目录执行 `bash experiments/rl/commands.sh`。本阶段使用 GRPO，从 `out/expand_sft_384.pth` 开始，每题采样 6 条回答，并依照 HTTP 规则计算奖励。输出 `out/expand_rl_384.pth`。

- [训练实现](train_grpo.py) · [HTTP 奖励规则](../../trainer/http_demo_reward.py) · [完整指令](commands.sh)
- 扩大实验用了 320 条奖励题。 [训练数据](data/expanded/rl.jsonl) · [采样与奖励日志](logs/expanded/rl.log) · [截图](screenshots/expanded/04-jupyter-rl.png)
- 小样本实验。 [训练数据](data/small/rl.jsonl) · [日志](logs/small/rl.log) · [截图](screenshots/small/04-jupyter-rl.png)

下一阶段用同一份 RL 学生权重训练蒸馏组和纯 CE 对照组。[蒸馏](../distillation/README.md)
