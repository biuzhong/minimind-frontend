"""Measure teacher/student distribution alignment on the same held-out answer positions."""

import argparse
import json
from pathlib import Path
import sys

import datasets  # Match MiniMind's import ordering for pyarrow and torch.
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

STUDENTS = ("rl", "ce_control", "distillation")
GROUPS = ("all", "basic", "policy")


def pack(samples, tokenizer, system, device):
    sequences, prefix_sizes, sequence_sizes = [], [], []
    for row in samples:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": row["prompt"]}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                                                 add_generation_prompt=True,
                                                 open_thinking=False, return_dict=False)
        expected = tokenizer.encode(row["answer"], add_special_tokens=False)
        sequence = prefix + expected + [tokenizer.eos_token_id]
        sequences.append(sequence)
        prefix_sizes.append(len(prefix))
        sequence_sizes.append(len(sequence))
    width = max(sequence_sizes)
    ids = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in sequences],
                       device=device)
    lengths = torch.tensor(sequence_sizes, device=device)
    prefixes = torch.tensor(prefix_sizes, device=device)
    token_mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    positions = torch.arange(width - 1, device=device)[None, :]
    answer_mask = (positions >= prefixes[:, None] - 1) & (positions < lengths[:, None] - 1)
    return ids, token_mask, answer_mask


@torch.inference_mode()
def measure(repo, records, device, temperature):
    sys.path.insert(0, str(repo))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from scripts.sharing_from_scratch import SYSTEM

    manifest = json.loads((records / "lineage.json").read_text(encoding="utf8"))
    samples = [json.loads(line) for line in (records / "holdout.jsonl").read_text(encoding="utf8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(repo / "model")
    models = {}
    for stage in ("teacher", *STUDENTS):
        item = manifest["stages"][stage]
        width, layers = item["shape"]
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=width, num_hidden_layers=layers))
        state = torch.load(repo / "out" / item["weight"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        models[stage] = model.eval().to(device)
        del state

    counts = dict.fromkeys(GROUPS, 0)
    sums = {name: dict.fromkeys(GROUPS, 0.0) for name in STUDENTS}
    batch_size = 8
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        ids, attention, answer_mask = pack(batch, tokenizer, SYSTEM, device)
        # Positions t predict t+1. Select only positions whose next token is
        # part of the same held-out reference answer (including EOS).
        teacher_logits = models["teacher"](ids, attention_mask=attention).logits[:, :-1, :]
        teacher_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
        group_mask = {kind: torch.tensor([kind == "all" or row["kind"] == kind for row in batch],
                                         dtype=torch.bool, device=device) for kind in GROUPS}
        for kind, selector in group_mask.items():
            counts[kind] += answer_mask[selector].sum().item()
        for stage in STUDENTS:
            logits = models[stage](ids, attention_mask=attention).logits[:, :-1, :]
            student_log = F.log_softmax(logits.float() / temperature, dim=-1)
            tokens = F.kl_div(student_log, teacher_log, reduction="none", log_target=True).sum(dim=-1)
            totals = ((temperature**2) * tokens.masked_fill(~answer_mask, 0)).sum(dim=1)
            for kind, selector in group_mask.items():
                sums[stage][kind] += totals[selector].sum(dtype=torch.float64).item()
    assert counts["all"] == counts["basic"] + counts["policy"] and all(counts.values())
    return {"temperature": temperature, "questions": len(samples), "reference_tokens": counts,
            "mean_scaled_kl": {stage: {key: round(value / counts[key], 6)
                                      for key, value in sums[stage].items()} for stage in STUDENTS}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    values = measure(args.repo.resolve(), args.records.resolve(), "cuda:0", temperature=2.0)
    rendered = json.dumps(values, indent=2)
    (args.records / "teacher_kl.json").write_text(rendered + "\n", encoding="utf8")
    print(rendered)
