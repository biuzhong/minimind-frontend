"""Run the MiniMind training stages on a small HTTP-status teaching dataset."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
SEQ_LEN = 64

CASES = {
    "200": ("成功", ["请求成功", "接口响应正常", "资源读取成功", "服务器正常返回"]),
    "401": ("未认证", ["未登录", "缺少凭证", "令牌已失效", "认证信息为空"]),
    "404": ("未找到", ["路径不存在", "资源没有找到", "路由未匹配", "页面不存在"]),
    "503": ("不可用", ["服务维护", "服务暂时不可用", "服务短时过载", "服务暂停提供"]),
}
TRAIN_TEMPLATES = ["{}，HTTP 状态码是？", "遇到「{}」时返回什么状态码？",
                   "只回复状态码。{}", "HTTP 速查，{}。"]
TEST_TEMPLATES = ["监控提示【{}】，应使用哪个 HTTP 状态码？",
                  "联调遇到{}，请给出三位数字状态码。"]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def conversation(question, answer):
    return {"conversations": [
        {"role": "system", "content": "你是 HTTP 状态码助手，只回复数字。"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def make_data(directory):
    text, train, test = [], [], []
    for answer, (label, situations) in CASES.items():
        for situation in situations[:3]:
            text.append({"text": f"HTTP {answer} 表示{label}。例如，{situation}。"})
            for template in TRAIN_TEMPLATES:
                train.append(conversation(template.format(situation), answer))
        for template in TEST_TEMPLATES:
            test.append(conversation(template.format(situations[3]), answer))
    write_jsonl(directory / "pretrain.jsonl", text * 4)
    write_jsonl(directory / "sft.jsonl", train)
    write_jsonl(directory / "holdout.jsonl", test)


def run_stage(script, name, data, logs, device, epochs, rate, **options):
    flags = {
        "device": device, "epochs": epochs, "learning_rate": rate, "batch_size": 4,
        "num_workers": 0, "accumulation_steps": 1, "max_seq_len": SEQ_LEN,
        "data_path": data, "save_weight": name, "log_interval": 4,
        **options,
    }
    command = [sys.executable, "-u", script]
    for key, value in flags.items():
        command.extend([f"--{key}", str(value)])
    print(f"Running {name}", flush=True)
    with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(
            command, cwd=TRAINER, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2",
                 "MKL_NUM_THREADS": "2"}, check=True,
        )


def evaluate(stages, data, device):
    import datasets  # Load pyarrow first on Windows, matching the trainers.
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from dataset.lm_dataset import SFTDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    held_out = SFTDataset(str(data), tokenizer, max_length=SEQ_LEN)
    random.seed(42)
    batches = list(held_out[index] for index in range(len(held_out)))
    results = []
    with torch.inference_mode():
        for name, width, layers, parent in stages:
            config = MiniMindConfig(hidden_size=width, num_hidden_layers=layers)
            torch.manual_seed(42)
            model = MiniMindForCausalLM(config).eval()
            weight = ROOT / "out" / f"{name}_{width}.pth"
            fingerprint = None
            if name != "random":
                model.load_state_dict(
                    torch.load(weight, map_location="cpu", weights_only=True), strict=True
                )
                fingerprint = hashlib.sha256(weight.read_bytes()).hexdigest()
            model.to(device)
            total_loss, total_tokens, exact = 0.0, 0, 0
            outputs = []
            for index, (tokens, labels) in enumerate(batches):
                ids = tokens[None].to(device)
                targets = labels[1:].to(device)
                logits = model(ids).logits[0, :-1].float()
                total_loss += F.cross_entropy(
                    logits, targets, ignore_index=-100, reduction="sum"
                ).item()
                total_tokens += (targets != -100).sum().item()

                messages = held_out.samples[index]["conversations"]
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True,
                    open_thinking=False,
                )
                encoded = tokenizer(prompt, return_tensors="pt").to(device)
                generated = model.generate(
                    **encoded, max_new_tokens=12, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
                answer = tokenizer.decode(
                    generated[0, encoded.input_ids.size(1):], skip_special_tokens=True
                ).strip()
                expected = messages[-1]["content"]
                exact += answer == expected
                outputs.append({"prompt": messages[-2]["content"],
                                "expected": expected, "generated": answer})
            if not total_tokens:
                raise RuntimeError("No supervised tokens in the held-out set.")
            results.append({
                "stage": name, "parent": parent, "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "sha256": fingerprint, "holdout_ce": round(total_loss / total_tokens, 4),
                "target_tokens": total_tokens, "exact": exact, "count": len(outputs),
                "examples": outputs,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "out" / "sharing")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(directory / "datasets-cache"))
    make_data(directory)
    pretrain, sft = directory / "pretrain.jsonl", directory / "sft.jsonl"
    for width, layers in [(128, 2), (192, 3)]:
        run_stage("train_pretrain.py", f"share_pre_{width}", pretrain, directory,
                  args.device, args.epochs, 8e-4, hidden_size=width,
                  num_hidden_layers=layers)
        run_stage("train_full_sft.py", f"share_sft_{width}", sft, directory,
                  args.device, args.epochs, 4e-4, hidden_size=width,
                  num_hidden_layers=layers, from_weight=f"share_pre_{width}")

    for name, alpha in [("share_ce", 1.0), ("share_kd", 0.5)]:
        run_stage("train_distillation.py", name, sft, directory, args.device,
                  args.epochs, 2e-4, alpha=alpha, temperature=2,
                  student_hidden_size=128, student_num_layers=2,
                  teacher_hidden_size=192, teacher_num_layers=3, teacher_use_moe=0,
                  from_student_weight="share_sft_128",
                  from_teacher_weight="share_sft_192")
    stages = [
        ("random", 128, 2, None),
        ("share_pre_128", 128, 2, "random"),
        ("share_sft_128", 128, 2, "share_pre_128"),
        ("share_sft_192", 192, 3, "share_pre_192"),
        ("share_ce", 128, 2, "share_sft_128"),
        ("share_kd", 128, 2, "share_sft_128"),
    ]
    results = evaluate(stages, directory / "holdout.jsonl", args.device)
    summary = directory / "results.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        print(f"{row['stage']:>14}  CE {row['holdout_ce']:.4f}  "
              f"exact {row['exact']}/{row['count']}  params {row['parameters']:,}")
    print(summary)


if __name__ == "__main__":
    main()
