"""Train a MiniMind student from random initialization through pretrain, SFT, GRPO and KD."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.http_demo_reward import answer_checks, score_answer

SEED = 42
STUDENT = (256, 4)
TEACHER = (384, 6)
SYSTEM = "你是前端故障助手。只返回一行 JSON，包含数字 code 和字符串 action。"
CASES = {
    200: ("ok", ["请求成功", "接口返回成功", "数据已正常读取", "请求已经完成", "资源响应正常"],
          ["数据读取成功，页面可以继续渲染", "接口已经正常返回用户信息"]),
    401: ("login", ["用户未认证", "令牌已过期", "请求缺少凭证", "用户没有登录", "登录凭证失效"],
          ["请求未通过认证，登录令牌过期", "当前会话未登录，需要重新认证"]),
    404: ("not_found", ["资源不存在", "路径没有匹配", "页面没有找到", "服务找不到资源", "路由不存在"],
          ["请求的页面路径找不到", "指定资源已不存在"]),
    503: ("retry", ["服务正在维护", "接口暂时不可用", "服务器短时过载", "服务暂时中断", "服务器暂时停用"],
          ["服务维护中，现在暂时不可用", "服务器过载，服务暂时无法访问"]),
}
TRAIN_ROUTES = ["订单接口", "图片服务", "登录入口", "列表页"]
HELD_ROUTES = ["导出任务", "收藏页面"]
TRAIN_PATTERNS = [
    "{route}出现这个情况：{case}。应该返回什么？",
    "在{route}联调时遇到{case}。请填 code 和 action。",
    "{route}的日志写着「{case}」。请按约定回复。",
    "{route}遇到 HTTP {code}（{case}）。请给出处理结果。",
]
RL_PATTERNS = [
    "排查{route}时看到「{case}」。请给出 JSON。",
    "Network 里{route}显示 {case}。判断该怎么处理。",
]
HELD_PATTERNS = [
    "联调{route}时，面板提示{case}。只输出一行 JSON，含 code 和 action。",
    "{route}收到 HTTP {code}，现场描述是「{case}」。按接口契约给出结果。",
]
PRETRAIN_PATTERNS = [
    "接口排错手册：{route}如果{case}，HTTP 状态码是 {code}，处理动作为 {action}。",
    "状态码速查：HTTP {code} 对应{case}。在{route}排查时，动作选择 {action}。",
    "联调记录：{route}发生{case}，看到 HTTP {code}，前端采用 {action}。",
    "FAQ：{route}出现{case}怎么处理？先确认 HTTP {code}，然后执行 {action}。",
    "接口测试中，HTTP {code} 表示{case}。对{route}，推荐动作是 {action}。",
]


def jsonl(path, rows):
    with path.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def reference(code, action):
    return json.dumps({"code": code, "action": action}, separators=(",", ":"))


def conversation(question, answer):
    return {"conversations": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def make_data(directory):
    pretrain, sft, teacher, rl, holdout = [], [], [], [], []
    for code, (action, cases, unseen) in CASES.items():
        answer = reference(code, action)
        for case, route, pattern in product(cases, TRAIN_ROUTES, PRETRAIN_PATTERNS):
            pretrain.append({"text": pattern.format(code=code, action=action, case=case, route=route)})
        for case, route, pattern in product(cases[:4], TRAIN_ROUTES, TRAIN_PATTERNS):
            q = pattern.format(code=code, case=case, route=route)
            sft.append(conversation(q, answer))
        for case, route, pattern in product(cases, TRAIN_ROUTES, TRAIN_PATTERNS + RL_PATTERNS):
            q = pattern.format(code=code, case=case, route=route)
            teacher.append(conversation(q, answer))
        for case, route, pattern in product(cases, TRAIN_ROUTES[:2], RL_PATTERNS):
            q = pattern.format(code=code, case=case, route=route)
            rl.append({**conversation(q, answer), "answer": answer})
        for case, route, pattern in product(unseen, HELD_ROUTES, HELD_PATTERNS):
            q = pattern.format(code=code, case=case, route=route)
            holdout.append({"prompt": q, "answer": answer})
    holdout.insert(0, {"prompt": "前端请求收到 HTTP 401。只输出一行 JSON，包含 code（数字）和 action（字符串）。",
                       "answer": reference(401, "login")})
    allowed = {row["conversations"][1]["content"] for row in sft + teacher + rl}
    if allowed.intersection(row["prompt"] for row in holdout):
        raise RuntimeError("Held-out prompts leaked into training")
    for name, rows in (("pretrain", pretrain * 3), ("sft", sft), ("teacher_sft", teacher),
                       ("rl", rl), ("holdout", holdout)):
        jsonl(directory / f"{name}.jsonl", rows)
    return {"pretrain": len(pretrain) * 3, "sft": len(sft), "teacher_sft": len(teacher),
            "rl": len(rl), "holdout": len(holdout)}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weights(name, shape):
    return ROOT / "out" / f"{name}_{shape[0]}.pth"


def command(script, data, name, device, shape=STUDENT, parent=None, **flags):
    options = {"device": device, "epochs": 1, "batch_size": 16, "num_workers": 0,
               "accumulation_steps": 1, "max_seq_len": 112,
               "save_interval": 1000, "log_interval": 20,
               "data_path": data, "save_weight": name,
               "hidden_size": shape[0], "num_hidden_layers": shape[1], **flags}
    if parent:
        options["from_weight"] = parent
    if script == "train_distillation.py":
        del options["hidden_size"], options["num_hidden_layers"]
        options.update(student_hidden_size=shape[0], student_num_layers=shape[1])
    args = [sys.executable, "-u", script]
    for key, value in options.items():
        args.append(f"--{key}")
        if value is not True:
            args.append(str(value))
    return args


def run_stage(args, log):
    print(f"$ {shlex.join(args)}", flush=True)
    with log.open("w", encoding="utf8") as stream:
        stream.write(f"$ {shlex.join(args)}\n")
        stream.flush()
        child = subprocess.Popen(args, cwd=ROOT / "trainer", stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf8",
                                 errors="replace")
        for line in child.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        if child.wait():
            raise subprocess.CalledProcessError(child.returncode, args)


def load_model(shape):
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    return MiniMindForCausalLM(MiniMindConfig(hidden_size=shape[0], num_hidden_layers=shape[1]))


def save_random():
    import torch

    path = weights("chain_random", STUDENT)
    path.parent.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    model = load_model(STUDENT)
    temporary = path.with_suffix(".partial")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def evaluate(holdout, records, device, length):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    samples = [json.loads(line) for line in holdout.read_text(encoding="utf8").splitlines()]
    stages = []
    with torch.inference_mode():
        for label, tag, shape in records:
            model = load_model(shape)
            model.load_state_dict(torch.load(weights(tag, shape), map_location="cpu",
                                            weights_only=True), strict=True)
            model.eval().to(device)
            outputs = []
            for offset in range(0, len(samples), 8):
                batch = samples[offset:offset + 8]
                prompts = [tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["prompt"]},
                ], tokenize=False, add_generation_prompt=True, open_thinking=False) for row in batch]
                encoded = tokenizer(prompts, padding=True, padding_side="left",
                                    return_tensors="pt").to(device)
                generated = model.generate(**encoded, max_new_tokens=length,
                                           do_sample=False, eos_token_id=tokenizer.eos_token_id)
                new_ids = generated[:, encoded.input_ids.shape[1]:]
                texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                for row, answer in zip(batch, texts):
                    answer = answer.strip()
                    parsed, code, action, shape_ok = answer_checks(answer, row["answer"])
                    outputs.append({**row, "generated": answer, "parsed_json": parsed,
                                    "correct_code": code, "correct_action": action,
                                    "valid_contract": shape_ok,
                                    "pass": code and action and shape_ok,
                                    "reward": round(score_answer(answer, row["answer"]), 4)})
            counts = {key: sum(row[key] for row in outputs)
                      for key in ("parsed_json", "correct_code", "correct_action", "valid_contract", "pass")}
            average = sum(row["reward"] for row in outputs) / len(outputs)
            stages.append({"stage": label, "weight": weights(tag, shape).name,
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "total": len(outputs), **counts, "mean_reward": round(average, 4),
                           "examples": outputs})
            print(f"{label}: {counts['pass']}/{len(outputs)} full contract, "
                  f"JSON {counts['parsed_json']}/{len(outputs)}, reward {average:.3f}", flush=True)
            del model
    return stages


def save_json(path, data):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "out" / "chain-records")
    parser.add_argument("--resume", action="store_true", help="Reuse stages marked complete in this run directory")
    args = parser.parse_args()

    import datasets  # Import pyarrow before torch, as in the upstream trainers.
    import torch

    if not re.fullmatch(r"cuda:\d+", args.device) or not torch.cuda.is_available():
        parser.error("This demonstration expects an available CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        parser.error("This demonstration expects BF16 support")
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "lineage.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error("--resume requires an existing lineage.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    else:
        if manifest_path.exists() or any((ROOT / "out").glob("chain_*_*.pth")):
            parser.error("Existing chain files detected. Use --resume or a fresh checkout.")
        manifest = {"seed": SEED, "data": make_data(directory), "stages": {}}
        save_json(manifest_path, manifest)

    def stage(label, tag, shape, source, flags=None):
        path = weights(tag, shape)
        saved = manifest["stages"].get(label)
        if saved:
            if not path.is_file() or sha(path) != saved["sha256"]:
                raise RuntimeError(f"Missing or changed checkpoint for {label}")
            print(f"Resume: verified {label}", flush=True)
            return
        print(f"STAGE {label}", flush=True)
        if flags is None:
            save_random()
        else:
            run_stage(flags, directory / (label + ".log"))
        manifest["stages"][label] = {
            "weight": path.name, "sha256": sha(path), "parent": source,
            "shape": list(shape), "command": flags,
        }
        save_json(manifest_path, manifest)

    def cmd(file, data, name, **opts):
        return command(file, directory / (data + ".jsonl"), name, args.device, **opts)

    stage("random", "chain_random", STUDENT, None)
    stage("pretrain", "chain_pre", STUDENT, "random",
          cmd("train_pretrain.py", "pretrain", "chain_pre", parent="chain_random",
              epochs=3, learning_rate=8e-4))
    stage("sft", "chain_sft", STUDENT, "pretrain",
          cmd("train_full_sft.py", "sft", "chain_sft", parent="chain_pre",
              epochs=12, learning_rate=5e-4))
    stage("rl", "chain_rl", STUDENT, "sft",
          cmd("train_grpo.py", "rl", "chain_rl", parent="chain_sft", epochs=3,
              batch_size=4, learning_rate=7e-5, max_gen_len=30, num_generations=6,
              thinking_ratio=0, beta=0.02, loss_type="grpo", http_demo_rewards=True,
              debug_mode=True, debug_interval=20, log_interval=4))
    stage("teacher_pretrain", "chain_teacher_pre", TEACHER, None,
          cmd("train_pretrain.py", "pretrain", "chain_teacher_pre", shape=TEACHER,
              epochs=3, learning_rate=7e-4))
    stage("teacher_sft", "chain_teacher_sft", TEACHER, "teacher_pretrain",
          cmd("train_full_sft.py", "teacher_sft", "chain_teacher_sft", shape=TEACHER,
              parent="chain_teacher_pre", epochs=10, learning_rate=4e-4))
    for label, tag, alpha in (("ce_control", "chain_ce", 1), ("distillation", "chain_kd", 0.5)):
        stage(label, tag, STUDENT, "rl",
              cmd("train_distillation.py", "teacher_sft", tag, epochs=1, learning_rate=7e-5,
                  batch_size=8, teacher_use_moe=0, teacher_hidden_size=TEACHER[0],
                  teacher_num_layers=TEACHER[1], from_student_weight="chain_rl",
                  from_teacher_weight="chain_teacher_sft", alpha=alpha, temperature=2))
    records = [(label, row["weight"].removesuffix(f"_{row['shape'][0]}.pth"),
                tuple(row["shape"])) for label, row in manifest["stages"].items()
               if label != "teacher_pretrain"]
    print("STAGE evaluation", flush=True)
    results = evaluate(directory / "holdout.jsonl", records, args.device, length=42)
    save_json(directory / "results.json", results)
    print(directory / "results.json", flush=True)


if __name__ == "__main__":
    main()
