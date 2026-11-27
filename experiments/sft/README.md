# SFT 监督微调

在 AutoDL 从仓库根目录执行 `bash experiments/sft/commands.sh`。学生从 `out/expand_pre_384.pth` 开始，输出 `out/expand_sft_384.pth`。要准备蒸馏用的 64M 教师，另运行 `bash experiments/sft/commands.sh teacher`。教师脚本下载并校验官方起点，写出 `out/expand_teacher_768.pth`。

- [训练实现](train_full_sft.py) · [学生和教师命令](commands.sh)
- 扩大实验用了 1,952 条学生对话。 [训练数据](data/expanded/sft.jsonl) · [日志](logs/expanded/sft.log) · [截图](screenshots/expanded/03-jupyter-sft.png)
- 教师训练和蒸馏共用同一份对话数据。 [扩大实验数据](../distillation/data/expanded/teacher_sft.jsonl) · [教师日志](logs/expanded/teacher.log)
- 小样本实验。 [学生对话](data/small/sft.jsonl) · [学生日志](logs/small/sft.log) · [教师对话](../distillation/data/small/teacher_sft.jsonl) · [教师日志](logs/small/teacher_sft.log) · [截图](screenshots/small/03-jupyter-sft.png)

下一阶段读取学生的 SFT 权重。[RL](../rl/README.md)
