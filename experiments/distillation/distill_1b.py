"""AutoDL task SFT and answer-distribution distillation across two tokenizers."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluate_scenarios import challenge_rows, group_scores, recorded_family
from scripts.sharing_expanded import CASE, CONDITION, STUDENT, TEACHER
from scripts.sharing_from_scratch import (
    SYSTEM, evaluate, jsonl, load_model, reference, save_json, sha, weights,
)
from trainer.http_demo_reward import answer_checks

MODEL_ID = "openbmb/MiniCPM5-1B"
REVISION = "ac83f813ba7da263c51a9f3ec37c2072597e54af"
TEMPERATURE = 2.0
ANSWERS = sorted({reference(code, spec[0]) for code, spec in CASE.items()}
                 | {reference(code, action) for code, _, action in CONDITION})


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def training_rows(path):
    return [{"prompt": row["conversations"][1]["content"],
             "answer": row["conversations"][2]["content"]} for row in read_rows(path)]


def chat_ids(tokenizer, prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         open_thinking=False, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def target_ids(tokenizer, answer, mini):
    eos = tokenizer.eos_token_id if mini else tokenizer.convert_tokens_to_ids("<|im_end|>")
    return tokenizer.encode(answer, add_special_tokens=False) + [eos]


def token_cost(model, prefixes, targets, pad, mini, device):
    import torch
    import torch.nn.functional as F

    sequences = [prefix + target for prefix, target in zip(prefixes, targets, strict=True)]
    padded = max(map(len, sequences))
    tokens = [seq + [pad] * (padded - len(seq)) for seq in sequences]
    labels = [[-100] * (len(prefix) - 1) + target + [-100] * (padded - len(seq) + 1)
              for prefix, target, seq in zip(prefixes, targets, sequences, strict=True)]
    input_ids = torch.tensor(tokens, device=device)
    labels = torch.tensor(labels, device=device)
    mask = labels != -100
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        states = model.model(input_ids, use_cache=False)
        hidden = states[0] if mini else states.last_hidden_state
        logits = model.lm_head(hidden[mask])
    costs = F.cross_entropy(logits.float(), labels[mask], reduction="none")
    # Sum the negative log probabilities for the full answer string, including its terminator.
    return torch.zeros(len(tokens), device=device).scatter_add(0, mask.nonzero()[:, 0], costs)


def candidate_cost(model, prefixes, options, pad, mini, device):
    return token_cost(model, [prefix for prefix in prefixes for _ in options],
                      options * len(prefixes), pad, mini, device).reshape(len(prefixes), len(options))


def batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def atomic_torch_save(data, path):
    import torch

    temporary = path.with_suffix(".partial")
    torch.save(data, temporary)
    temporary.replace(path)


def fit_teacher(model, tokenizer, train, device, output):
    import torch

    model.requires_grad_(False)
    model.model.layers[-2:].requires_grad_(True).float()
    model.model.norm.requires_grad_(True).float()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    print(f"TEACHER total={sum(p.numel() for p in model.parameters()):,} "
          f"trainable={sum(p.numel() for p in trainable.values()):,} examples={len(train)}", flush=True)
    checkpoint = output / "teacher-tuned.pth"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(state) != set(trainable):
            raise ValueError("Checkpoint parameters do not match the trainable teacher layers")
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(state[name])
        print("TEACHER restored", flush=True)
        return
    model.train()
    optimizer = torch.optim.AdamW(trainable.values(), lr=3e-5, fused=True)
    for step, batch in enumerate(batches(train, 16), 1):
        prefixes = [chat_ids(tokenizer, row["prompt"]) for row in batch]
        targets = [target_ids(tokenizer, row["answer"], False) for row in batch]
        loss = token_cost(model, prefixes, targets, tokenizer.pad_token_id, False, device).sum()
        loss = loss / sum(map(len, targets))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            print(f"teacher_sft step={step} loss={loss.item():.4f}", flush=True)
    atomic_torch_save({name: p.detach().cpu() for name, p in trainable.items()}, checkpoint)


def teacher_targets(model, tokenizer, train, device, output, mini):
    import torch

    if output.exists():
        return torch.load(output, map_location="cpu", weights_only=True)
    model.eval()
    options = [target_ids(tokenizer, answer, mini) for answer in ANSWERS]
    parts = []
    with torch.inference_mode():
        for step, batch in enumerate(batches(train, 8), 1):
            prefixes = [chat_ids(tokenizer, row["prompt"]) for row in batch]
            nll = candidate_cost(model, prefixes, options, tokenizer.pad_token_id, mini, device)
            parts.append((-nll / TEMPERATURE).softmax(-1).cpu())
            if step == 1 or step % 40 == 0:
                print(f"{output.stem} targets={min(step * 8, len(train))}/{len(train)}", flush=True)
    result = torch.cat(parts)
    atomic_torch_save(result, output)
    return result


def fit_student(tag, tokenizer, train, device, probabilities):
    import torch
    import torch.nn.functional as F

    path = weights(tag, STUDENT)
    if path.exists():
        print(f"{tag} restored", flush=True)
        return
    student = native_model("expand_rl", STUDENT, device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, fused=True)
    options = [target_ids(tokenizer, answer, True) for answer in ANSWERS]
    for step, batch in enumerate(batches(train, 16), 1):
        prefixes = [chat_ids(tokenizer, row["prompt"]) for row in batch]
        labels = [ANSWERS.index(row["answer"]) for row in batch]
        gold = [options[index] for index in labels]
        if probabilities is None:
            ce = token_cost(student, prefixes, gold, tokenizer.pad_token_id, True, device).sum()
            ce = ce / sum(map(len, gold))
            loss = ce
        else:
            nll = candidate_cost(student, prefixes, options, tokenizer.pad_token_id, True, device)
            ce = nll[torch.arange(len(labels), device=device), labels].sum() / sum(map(len, gold))
            targets = probabilities[(step - 1) * 16:(step - 1) * 16 + len(batch)].to(device)
            kl = F.kl_div((-nll / TEMPERATURE).log_softmax(-1), targets, reduction="batchmean")
            loss = 0.5 * ce + 0.5 * TEMPERATURE ** 2 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            print(f"{tag} step={step} loss={loss.item():.4f} ce={ce.item():.4f}", flush=True)
    atomic_torch_save({k: v.detach().cpu() for k, v in student.state_dict().items()}, path)


def native_model(tag, shape, device):
    import torch

    model = load_model(shape)
    model.load_state_dict(torch.load(weights(tag, shape), map_location="cpu", weights_only=True))
    return model.to(device)


def teacher_generate(model, tokenizer, samples, device):
    import torch

    examples = []
    model.eval()
    tokenizer.padding_side = "left"
    with torch.inference_mode():
        for batch in batches(samples, 16):
            model_input = tokenizer.pad({"input_ids": [chat_ids(tokenizer, row["prompt"])
                                                       for row in batch]}, return_tensors="pt").to(device)
            tokens = model.generate(**model_input, max_new_tokens=32, do_sample=False)
            answers = tokenizer.batch_decode(tokens[:, model_input.input_ids.shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, answers, strict=True):
                generated = raw.strip()
                parsed, code, action, shape = answer_checks(generated, row["answer"])
                examples.append({**row, "generated": generated, "parsed_json": parsed,
                                 "correct_code": code, "correct_action": action,
                                 "valid_contract": shape, "pass": code and action and shape})
    return {"stage": "teacher_1b", "parameters": sum(p.numel() for p in model.parameters()),
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.device.startswith("cuda:") or not torch.cuda.is_bf16_supported():
        parser.error("Run this experiment on an AutoDL CUDA instance with bfloat16 support")
    if args.output.resolve() == args.run_dir.resolve():
        parser.error("Choose a dedicated output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    train_path = args.run_dir / "teacher_sft.jsonl"
    train = training_rows(train_path)
    random.Random(42).shuffle(train)
    subsets = {"original": read_rows(args.run_dir / "holdout.jsonl"), "challenge": challenge_rows()}
    if {row["prompt"] for row in train}.intersection(row["prompt"] for rows in subsets.values() for row in rows):
        raise ValueError("Held-out prompts leaked into the training data")

    model = AutoModelForCausalLM.from_pretrained(args.teacher_dir, torch_dtype=torch.bfloat16,
                                                local_files_only=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, local_files_only=True)
    if tokenizer.convert_tokens_to_ids("<|im_end|>") not in model.generation_config.eos_token_id:
        raise ValueError("Unexpected MiniCPM chat terminator")
    fit_teacher(model, tokenizer, train, args.device, args.output)
    model.to(dtype=torch.bfloat16)
    large_eval = {name: teacher_generate(model, tokenizer, rows, args.device) for name, rows in subsets.items()}
    save_json(args.output / "teacher-evaluation.json", large_eval)
    probs_large = teacher_targets(model, tokenizer, train, args.device, args.output / "targets-1b.pth", False)
    del model
    torch.cuda.empty_cache()

    mini_tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    small_teacher = native_model("expand_teacher", TEACHER, args.device)
    probs_small = teacher_targets(small_teacher, mini_tokenizer, train, args.device,
                                  args.output / "targets-64m.pth", True)
    del small_teacher
    fits = [("scenes_ce", None), ("scenes_kd64m", probs_small), ("scenes_kd1b", probs_large)]
    for tag, probs in fits:
        fit_student(tag, mini_tokenizer, train, args.device, probs)
    stages = [("pretrain", "expand_pre", STUDENT), ("sft", "expand_sft", STUDENT),
              ("rl", "expand_rl", STUDENT), ("teacher_64m", "expand_teacher", TEACHER),
              ("original_kd64m", "expand_kd", STUDENT),
              *((tag, tag, STUDENT) for tag, _ in fits)]
    for name, rows in subsets.items():
        holdout = args.output / f"{name}.jsonl"
        jsonl(holdout, rows)
        outputs = evaluate(holdout, stages, args.device, length=32) + [large_eval[name]]
        scores = group_scores(outputs, recorded_family if name == "original" else lambda row: row["family"])
        save_json(args.output / f"{name}-results.json", outputs)
        save_json(args.output / f"{name}-summary.json", scores)
    save_json(args.output / "manifest.json", {"teacher": MODEL_ID, "revision": REVISION,
              "base_config_sha256": sha(args.teacher_dir / "config.json"),
              "teacher_update_sha256": sha(args.output / "teacher-tuned.pth"),
              "training_data_sha256": sha(train_path), "training_examples": len(train),
              "temperature": TEMPERATURE, "answers": ANSWERS, "seed": 42,
              "weights": {label: sha(weights(tag, shape)) for label, tag, shape in stages}})
    print("EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
