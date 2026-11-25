# 实跑记录

| 实验 | 主要过程 | 原始材料 |
| --- | --- | --- |
| 首轮官方权重 | 官方预训练、SFT、GRPO 快照推理，另实训了纯 CE 和蒸馏。 | [逐题输出](official/records/stage_outputs.json) · [截图](official/screenshots/03-jupyter-stage-outputs.png) |
| 小样本从零 | 学生随机初始化后实际运行预训练、SFT、GRPO 和蒸馏。 | [训练血缘](small/records/lineage.json) · [逐题输出](small/records/results.json) · [截图](small/screenshots/06-jupyter-effects.png) |
| 扩大实验 | 学生完整训练、64M 教师微调、纯 CE、逐题生成和教师 KL。 | [训练血缘](expanded/records/lineage.json) · [逐题输出](expanded/records/results.json) · [KL](expanded/records/teacher_kl.json) · [截图](expanded/screenshots/08-jupyter-distillation-effect.png) |
| 1B 教师对照 | MiniCPM5-1B 微调、两位教师的候选答案概率蒸馏、纯 CE 和 48 道额外成对评测。 | [训练日志](teacher-1b/records/training.log) · [实验清单](teacher-1b/records/manifest.json) · [原题输出](teacher-1b/records/original-results.json) · [额外题输出](teacher-1b/records/challenge-results.json) · [JupyterLab 截图](teacher-1b/screenshots/03-jupyter-scenario-results.png) |

各轮的 `records/` 保存本轮命令、数据记录与评测结果，`screenshots/` 保存 AutoDL JupyterLab 浏览器原图。实跑日志里的绝对路径属于当时的云端环境。复跑使用[统一入口](../run.py)。[完整分享稿](../../docs/share.md)
