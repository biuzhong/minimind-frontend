# 首轮官方权重实验的数据

这轮蒸馏和纯 CE 共用 MiniMind 上游 `sft_t2t_mini.jsonl` 的前 512 行。[固定版本的数据来源和校验值](../../../inputs.json)

从仓库根目录运行下列命令，即可把精确的训练样本缓存到本目录。

```bash
python - <<'PYTHON'
from pathlib import Path
from experiments.run import fetch

fetch("sharing_kd_512.jsonl", Path("experiments/distillation/data/official"))
PYTHON
```

一键复跑使用 `python experiments/run.py official`，默认把同一份训练样本下载到 `.assets/`。
