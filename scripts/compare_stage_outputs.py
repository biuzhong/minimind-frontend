"""Export the same prompts from each 768-wide MiniMind checkpoint on a cloud GPU."""

import datasets  # Load pyarrow before torch on Windows, as in the training scripts.
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

PROMPTS = [
    '前端请求收到 HTTP 401。只输出一行 JSON，包含 code（数字）和 action（字符串）。',
    '用户说网页打不开，Network 面板显示 HTTP 503。用两句话回答：'
    '第一句解释状态码，第二句描述前端该怎么处理。',
    '解释浏览器缓存和 CDN 缓存的区别。请用两条列表，每条只有一句话。',
]

STAGES = [
    ("pretrain", "pretrain"),
    ("sft", "full_sft"),
    ("rl", "grpo"),
    ("distillation", "cloud_kd"),
]


@torch.inference_mode()
def compare(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    stages = STAGES + ([("ce_control", "cloud_ce")] if args.include_ce_control else [])
    paths = [(stage, args.weights / f"{prefix}_768.pth") for stage, prefix in stages]
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    results = [{"prompt": prompt, "outputs": []} for prompt in PROMPTS]
    for stage, path in paths:
        model = MiniMindForCausalLM(config).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True),
                              strict=True)
        model.to(device)
        for row in results:
            if stage == "pretrain" and args.template == "native":
                prompt = tokenizer.bos_token + row["prompt"]
                template = "continuation"
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False,
                    add_generation_prompt=True, open_thinking=False,
                )
                template = "chat"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            sequence = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )[0, inputs.input_ids.shape[1]:]
            answer = tokenizer.decode(sequence, skip_special_tokens=True).strip()
            row["outputs"].append({
                "stage": stage, "checkpoint": path.name, "template": template,
                "answer": answer, "generated_tokens": sequence.numel(),
            })
        del model
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", type=Path, default=ROOT / "out")
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "stage_outputs.json")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--template", choices=["native", "chat"], default="native")
    parser.add_argument("--include-ce-control", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    rows = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["prompt"])
        for output in row["outputs"]:
            print(f"  [{output['stage']}] {output['answer']}")
    print(args.output)


if __name__ == "__main__":
    main()
