"""Score held-out reference token likelihoods using the saved chain checkpoints."""

import argparse
import json
from pathlib import Path
import sys

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


@torch.inference_mode()
def measure(repo, records, device):
    sys.path.insert(0, str(repo))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from scripts.sharing_from_scratch import SYSTEM

    lineage = json.loads((records / "lineage.json").read_text(encoding="utf8"))["stages"]
    heldout = [json.loads(line) for line in (records / "holdout.jsonl").read_text(encoding="utf8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(repo / "model")
    rows = []
    for name, checkpoint in lineage.items():
        if name == "teacher_pretrain":
            continue
        width, depth = checkpoint["shape"]
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=width, num_hidden_layers=depth))
        state = torch.load(repo / "out" / checkpoint["weight"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        count, loss_sum = 0, 0.0
        for example in heldout:
            context = tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": example["prompt"]},
            ], tokenize=True, add_generation_prompt=True, open_thinking=False)
            answer = tokenizer.encode(example["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
            ids = torch.tensor([context + answer], device=device)
            logits = model(ids).logits[0, len(context)-1:len(context)+len(answer)-1].float()
            loss_sum += F.cross_entropy(logits, ids[0, len(context):], reduction="sum").item()
            count += len(answer)
        value = {"stage": name, "mean_reference_ce": round(loss_sum / count, 6), "tokens": count}
        rows.append(value)
        print(json.dumps(value), flush=True)
        del model, state
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = measure(args.repo.resolve(), args.records.resolve(), args.device)
    (args.records / "reference_ce.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
