# minimind-frontend

A MiniMind training walkthrough for frontend developers. Compare pretraining, supervised fine-tuning, GRPO, and white-box distillation on the same HTTP task.

[中文 README](README.md) · [Chinese walkthrough](docs/share.md) · [Recorded experiments](experiments/recorded/README.md)

## Setup

Use Linux and a BF16-capable NVIDIA GPU for training. The recorded environment used Python 3.12.3, PyTorch 2.8.0 and CUDA 12.8. Install a matching CUDA build of PyTorch and then install the minimal dependencies.

```bash
git clone https://github.com/biuzhong/minimind-frontend.git
cd minimind-frontend
python -m pip install -r experiments/requirements.txt
python experiments/verify.py
```

## Reproduce

| Experiment | Command |
| --- | --- |
| Small GPU smoke run | `python experiments/run.py smoke` |
| Official snapshots with fresh CE and distillation | `python experiments/run.py official` |
| Complete small training chain | `python experiments/run.py from-scratch` |
| Expanded run with held-out teacher KL | `python experiments/run.py expanded` |

Use `--run-dir PATH` to choose storage and `--resume --run-dir PATH` to resume. `--prepare-only` snapshots code without downloads or GPU use. Run directories keep weights under `source/out/` and logs and evaluations under `records/`.

The upstream project is [MiniMind](https://github.com/jingyaogong/minimind). Code uses [Apache-2.0](LICENSE). Downloaded model and data assets retain their upstream terms.
