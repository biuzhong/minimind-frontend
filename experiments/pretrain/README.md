# 预训练

在 AutoDL 从仓库根目录执行 `bash experiments/pretrain/commands.sh`。脚本保存随机学生起点，读取接口排障文本，训练并写出 `out/expand_pre_384.pth`。新日志保存在 `runs/stages/pretrain.log`。

- [训练实现](train_pretrain.py) · [完整指令](commands.sh)
- 扩大实验用了 5,760 段训练文本，另用留出文本测续写。 [训练数据](data/expanded/pretrain.jsonl) · [续写留出数据](data/expanded/prose_holdout.jsonl) · [日志](logs/expanded/pretrain.log) · [截图](screenshots/expanded/02-jupyter-pretrain.png) · [续写结果](evaluation/expanded/continuation.json)
- 小样本实验保留了学生和教师的预训练日志。 [训练数据](data/small/pretrain.jsonl) · [学生日志](logs/small/pretrain.log) · [教师日志](logs/small/teacher_pretrain.log) · [截图](screenshots/small/02-jupyter-pretrain.png)

下一阶段读取这份预训练权重。[SFT](../sft/README.md)
