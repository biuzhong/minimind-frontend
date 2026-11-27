# 四阶段演示

每个阶段目录都有可直接阅读的训练代码、运行指令、实际输入、日志和 AutoDL JupyterLab 截图。阶段内用 `expanded` 和 `small` 目录区分两轮从零训练。

| 阶段 | 训练入口 | 扩大实验材料 |
| --- | --- | --- |
| [预训练](pretrain/README.md) | [代码](pretrain/train_pretrain.py) · [命令](pretrain/commands.sh) | [文本](pretrain/data/expanded/pretrain.jsonl) · [日志](pretrain/logs/expanded/pretrain.log) · [截图](pretrain/screenshots/expanded/02-jupyter-pretrain.png) |
| [SFT](sft/README.md) | [代码](sft/train_full_sft.py) · [命令](sft/commands.sh) | [对话](sft/data/expanded/sft.jsonl) · [日志](sft/logs/expanded/sft.log) · [截图](sft/screenshots/expanded/03-jupyter-sft.png) |
| [RL](rl/README.md) | [GRPO 代码](rl/train_grpo.py) · [命令](rl/commands.sh) | [奖励题](rl/data/expanded/rl.jsonl) · [日志](rl/logs/expanded/rl.log) · [截图](rl/screenshots/expanded/04-jupyter-rl.png) |
| [蒸馏](distillation/README.md) | [同词表蒸馏](distillation/train_distillation.py) · [命令](distillation/commands.sh) · [1B 教师实验](distillation/distill_1b.py) | [蒸馏数据](distillation/data/expanded/teacher_sft.jsonl) · [日志](distillation/logs/expanded/distillation.log) · [截图](distillation/screenshots/expanded/05-jupyter-distillation.png) |

在 AutoDL 完成[依赖安装](../README.md#安装)后，从仓库根目录按顺序运行。学生权重保存在 `out/`，新日志默认写入 `runs/stages/`。使用独立 checkout 可以同时保留两轮权重。

```bash
bash experiments/pretrain/commands.sh
bash experiments/sft/commands.sh
bash experiments/rl/commands.sh
bash experiments/sft/commands.sh teacher
bash experiments/distillation/commands.sh
bash experiments/distillation/commands.sh ce
```

[统一复现入口](run.py)仍可一键运行整条链路。[完整分享稿](../docs/share.md)包含每阶段原理解读、实际回答和参数说明。

## 跨阶段对照

| 实验 | 共同记录 |
| --- | --- |
| 首轮官方权重 | [逐题输出](comparison/official/records/stage_outputs.json) · [多阶段截图](comparison/official/screenshots/03-jupyter-stage-outputs.png) · [当轮蒸馏日志](distillation/logs/official/cloud_kd.log) |
| 小样本从零 | [训练血缘](comparison/small/records/lineage.json) · [逐题输出](comparison/small/records/results.json) · [多阶段截图](comparison/small/screenshots/06-jupyter-effects.png) |
| 扩大实验 | [训练血缘](comparison/expanded/records/lineage.json) · [逐题输出](comparison/expanded/records/results.json) · [同题四阶段截图](comparison/expanded/screenshots/07-jupyter-one-question-four-stages.png) |
| 1B 教师对照 | [训练日志](distillation/logs/teacher-1b/training.log) · [实验清单](distillation/records/teacher-1b/manifest.json) · [新题输出](distillation/records/teacher-1b/challenge-results.json) · [结果截图](distillation/screenshots/teacher-1b/03-jupyter-scenario-results.png) |
