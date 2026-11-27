# 蒸馏

在 AutoDL 从仓库根目录执行 `bash experiments/distillation/commands.sh`，用 `out/expand_rl_384.pth` 和 [SFT 阶段准备的 64M 教师](../sft/README.md)训练学生，输出 `out/expand_kd_384.pth`。执行 `bash experiments/distillation/commands.sh ce` 得到同一 RL 起点和同一批数据的纯 CE 对照。

- [同词表训练实现](train_distillation.py) · [完整命令](commands.sh) · [教师概率的 KL 评测代码](measure_teacher_kl.py)
- 扩大实验用了 3,200 条对话。 [训练数据](data/expanded/teacher_sft.jsonl) · [蒸馏日志](logs/expanded/distillation.log) · [纯 CE 日志](logs/expanded/ce_control.log) · [训练截图](screenshots/expanded/05-jupyter-distillation.png) · [分布对照](evaluation/expanded/teacher_kl.json)
- 小样本实验。 [训练数据](data/small/teacher_sft.jsonl) · [蒸馏日志](logs/small/distillation.log) · [纯 CE 日志](logs/small/ce_control.log) · [截图](screenshots/small/05-jupyter-distillation-utf8.png)
- 首轮官方权重实验使用上游 SFT 数据的前 512 条。 [固定版本的数据和下载指令](data/official/README.md) · [训练日志](logs/official/cloud_kd.log) · [截图](screenshots/official/02-jupyter-training.png)

## 1B 教师对照

1B 实验只从扩大实验的 RL 权重继续训练。教师与学生采用各自的词表，比较候选 JSON 答案的概率。用 `python experiments/run.py expanded` 生成扩大实验权重后，再从对应运行目录的 `source/` 调用训练脚本。

```bash
BASE="$HOME/minimind-runs"
hf download openbmb/MiniCPM5-1B \
  --revision ac83f813ba7da263c51a9f3ec37c2072597e54af \
  --include '*.json' '*.jinja' '*.safetensors' \
  --local-dir "$BASE/teacher-1b-base"
cd "$BASE/expanded/source"
python experiments/distillation/distill_1b.py \
  --run-dir ../records --teacher-dir "$BASE/teacher-1b-base" --output "$BASE/1b"
```

- [1B 训练实现](distill_1b.py) · [复用的 3,200 条训练数据](data/expanded/teacher_sft.jsonl) · [48 道额外评测题](data/teacher-1b/challenge.jsonl)
- [本轮实际运行命令](logs/teacher-1b/run-cloud.sh) · [训练与评测日志](logs/teacher-1b/training.log) · [实验清单](records/teacher-1b/manifest.json)
- [教师 SFT 截图](screenshots/teacher-1b/01-jupyter-teacher-sft.png) · [蒸馏截图](screenshots/teacher-1b/02-jupyter-answer-distillation.png) · [评测截图](screenshots/teacher-1b/03-jupyter-scenario-results.png)
