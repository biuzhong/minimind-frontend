"""Bigger HTTP teaching experiment with probes specific to each training stage."""

import argparse
from itertools import product
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sharing_from_scratch import (
    command, conversation, evaluate, jsonl, load_model, reference,
    run_stage, save_json, sha, weights,
)

STUDENT = (384, 6)
TEACHER = (768, 8)
SEED = 42
OFFICIAL_SHA = "5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0"
CASE = {
    200: ("ok", ("接口返回成功", "读取成功", "请求正常完成", "业务返回正常", "操作成功", "数据已收到")),
    400: ("fix", ("请求参数有误", "必填字段缺失", "提交的格式错误", "输入校验失败", "请求参数不合法", "输入格式不正确")),
    401: ("login", ("未登录", "身份校验失败", "令牌已过期", "请求未认证", "凭证缺失", "会话已失效")),
    403: ("deny", ("账号没有权限", "访问被禁止", "权限检查不通过", "用户无权操作", "权限不足", "资源禁止访问")),
    404: ("missing", ("路径不存在", "页面找不到", "指定资源不存在", "路由未匹配", "资源未找到", "接口地址没有对应资源")),
    409: ("conflict", ("版本冲突", "资源当前状态产生冲突", "编辑版本不一致", "记录已被他人更新", "更新条件冲突", "提交发生并发冲突")),
    429: ("wait", ("请求过多", "客户端被限流", "触发请求频率限制", "调用超过限额", "短时间重复请求太多", "接口限流生效")),
    503: ("retry", ("服务正在维护", "服务暂时不可用", "上游过载", "接口暂时下线", "服务压力太大", "后端正在维护")),
}
TRAIN_ROUTE = ("订单", "登录", "结算", "图片", "通知", "库存", "搜索", "频道", "购物车", "账单",
               "订阅", "店铺", "发票", "积分", "消息", "审核", "预览", "报表", "评论", "偏好")
TEST_ROUTE = ("收藏", "导出", "质检", "统计", "标签", "地址", "对账", "投票")
PROSE = (
    ("联调手册：{route}接口提示{case}，HTTP 状态码是 ", "，前端动作是 {action}。"),
    ("故障排查：{route}检测到{case}，状态码记为 ", "，处理方式标记为 {action}。"),
    ("Network 记录：{route}出现{case}，响应码为 ", "，建议动作标记为 {action}。"),
    ("接口监控：{route}发生{case}，故障卡片上的状态码为 ", "，建议先采用 {action}。"),
    ("上线手册：{route}报告{case}，前端看到的状态码为 ", "，处理动作写作 {action}。"),
    ("响应说明：{route}遇到{case}，请记录 HTTP ", "，并将动作记为 {action}。"),
)
CHAT_TRAIN = (
    "{route}接口报出 HTTP {code}（{case}）。返回处理结果。",
    "请帮我处理{route}接口的故障。Network 显示 HTTP {code}，说明是{case}。",
    "在{route}联调看到{case}，HTTP {code}。只输出 JSON。",
)
CHAT_TEST = (
    "今天在{route}页面排错，面板显示 HTTP {code}，同时报告{case}。请填 code 和 action。",
    "审核{route}接口的日志时看到「{case}」，返回码是 {code}。请提供一行 JSON。",
)
CONDITION = (
    (401, "refresh_token=present", "refresh"),
    (401, "refresh_token=absent", "login"),
    (503, "method=GET", "retry"),
    (503, "method=POST", "stop"),
)
RULE_TRAIN = (
    "{route}接口收到 HTTP {code}，属性 {flag}。按自动处理规则给出结果。",
    "请只返回 JSON。Network 中{route}接口响应 HTTP {code}，{flag}。",
    "联调{route}时收到 {code}。此次请求的附加信息为 {flag}。",
    "排查{route}工单：HTTP {code}；{flag}。给出建议动作。",
)
RULE_TEST = (
    "线上{route}接口报错 HTTP {code}。请求元数据为 {flag}。请给出一行 JSON。",
    "{route}联调记录 HTTP {code}，条件为 {flag}。这个场景该执行什么动作？",
)


def build(directory):
    pre, prose_test, student, teacher, rollouts, held = [], [], [], [], [], []
    for code, (action, phrases) in CASE.items():
        for route, case, (prefix, suffix) in product(TRAIN_ROUTE, phrases, PROSE):
            lead = prefix.format(route=route, case=case)
            tail = f"{code}" + suffix.format(action=action)
            pre.append({"text": lead + tail})
        for route, case, (prefix, suffix) in product(TEST_ROUTE[:4], phrases[-2:], PROSE[:2]):
            prose_test.append({"prefix": prefix.format(route=route, case=case),
                               "target": f"{code}" + suffix.format(action=action)})
        for route, case, pattern in product(TRAIN_ROUTE, phrases[:4], CHAT_TRAIN):
            student.append(conversation(pattern.format(route=route, code=code, case=case),
                                        reference(code, action)))
        for route, case, pattern in product(TRAIN_ROUTE, phrases, CHAT_TRAIN):
            teacher.append(conversation(pattern.format(route=route, code=code, case=case),
                                        reference(code, action)))
        for route, case, pattern in product(TEST_ROUTE, phrases[-2:], CHAT_TEST):
            held.append({"kind": "basic", "prompt": pattern.format(route=route, code=code, case=case),
                         "answer": reference(code, action)})
    for code, flag, action in CONDITION:
        for route, pattern in product(TRAIN_ROUTE, RULE_TRAIN):
            q = pattern.format(route=route, code=code, flag=flag)
            a = reference(code, action)
            teacher.append(conversation(q, a))
            rollouts.append({**conversation(q, a), "answer": a})
        # Restricted demonstrations let the SFT model sample the conditional action labels.
        for route, pattern in product(TRAIN_ROUTE[:4], RULE_TRAIN[:2]):
            student.append(conversation(pattern.format(route=route, code=code, flag=flag),
                                        reference(code, action)))
        for route, pattern in product(TEST_ROUTE, RULE_TEST):
            held.append({"kind": "policy", "prompt": pattern.format(route=route, code=code, flag=flag),
                         "answer": reference(code, action)})
    held.insert(0, {"kind": "policy", "prompt": "登录请求收到 HTTP 401。refresh_token=present。请只返回一行 JSON，包含 code 和 action。",
                    "answer": reference(401, "refresh")})
    held.insert(1, {"kind": "policy", "prompt": "创建订单收到 HTTP 503，method=POST。请只返回一行 JSON，包含 code 和 action。",
                    "answer": reference(503, "stop")})
    train_questions = {row["conversations"][1]["content"] for row in student + teacher + rollouts}
    assert not train_questions.intersection(row["prompt"] for row in held)
    random.Random(SEED).shuffle(student)
    random.Random(SEED).shuffle(teacher)
    saved = {"pretrain": pre, "prose_holdout": prose_test, "sft": student, "teacher_sft": teacher,
             "rl": rollouts, "holdout": held}
    for name, rows in saved.items():
        jsonl(directory / f"{name}.jsonl", rows)
    counts = {key: len(value) for key, value in saved.items()}
    counts["basic_holdout"] = sum(row["kind"] == "basic" for row in held)
    counts["policy_holdout"] = sum(row["kind"] == "policy" for row in held)
    return counts


def save_random(path):
    import torch

    path.parent.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.save(load_model(STUDENT).state_dict(), path)


def prose_metrics(directory, stage_rows, device):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    rows = [json.loads(line) for line in (directory / "prose_holdout.jsonl").read_text(encoding="utf8").splitlines()]
    results = []
    with torch.inference_mode():
        for name, stage in stage_rows.items():
            if name not in {"random", "pretrain"}:
                continue
            model = load_model(STUDENT)
            model.load_state_dict(torch.load(ROOT / "out" / stage["weight"], map_location="cpu",
                                             weights_only=True), strict=True)
            model.eval().to(device)
            count, loss, samples = 0, 0.0, []
            for idx, row in enumerate(rows):
                prefix = [tokenizer.bos_token_id] + tokenizer.encode(row["prefix"], add_special_tokens=False)
                target = tokenizer.encode(row["target"], add_special_tokens=False)
                ids = torch.tensor([prefix + target], device=device)
                output = model(ids).logits[0, len(prefix) - 1:len(prefix) + len(target) - 1].float()
                loss += F.cross_entropy(output, ids[0, len(prefix):], reduction="sum").item()
                count += len(target)
                if idx < 4:
                    sequence = model.generate(torch.tensor([prefix], device=device), max_new_tokens=22,
                                              do_sample=False, eos_token_id=tokenizer.eos_token_id)
                    samples.append({**row, "generated": tokenizer.decode(sequence[0, len(prefix):],
                                                                         skip_special_tokens=True)})
            results.append({"stage": name, "continuation_ce": round(loss / count, 5),
                            "token_count": count, "total": len(rows), "examples": samples})
            del model
    return results


def summarize(results):
    table = []
    for row in results:
        summary = {"stage": row["stage"], "parameters": row["parameters"]}
        for kind in ("basic", "policy"):
            relevant = [item for item in row["examples"] if item["kind"] == kind]
            summary[kind] = {"total": len(relevant),
                             "json": sum(item["parsed_json"] for item in relevant),
                             "passed": sum(item["pass"] for item in relevant)}
        table.append(summary)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import datasets
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or not args.device.startswith("cuda:"):
        parser.error("An NVIDIA GPU with BF16 support is required")
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "lineage.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    else:
        if manifest_path.exists():
            parser.error("Existing run. Pass --resume or select a new run directory and checkout")
        manifest = {"seed": SEED, "data": build(directory), "official_teacher_sha256": OFFICIAL_SHA,
                    "stages": {}}
        save_json(manifest_path, manifest)
    official = weights("expand_teacher_base", TEACHER)
    official.parent.mkdir(exist_ok=True)
    if not official.exists():
        if sha(args.teacher_checkpoint) != OFFICIAL_SHA:
            raise ValueError("Official teacher checkpoint hash mismatch")
        shutil.copyfile(args.teacher_checkpoint, official)
    if sha(official) != OFFICIAL_SHA:
        raise ValueError("Cached teacher checkpoint hash mismatch")

    def stage(label, tag, shape, parent, flags=None):
        path = weights(tag, shape)
        if label in manifest["stages"]:
            assert sha(path) == manifest["stages"][label]["sha256"]
            print(f"Verified stage {label}", flush=True)
            return
        print(f"STAGE {label}", flush=True)
        if flags is None:
            save_random(path)
        else:
            run_stage(flags, directory / f"{label}.log")
        manifest["stages"][label] = {"weight": path.name, "sha256": sha(path),
                                     "parent": parent, "shape": shape, "command": flags}
        save_json(manifest_path, manifest)

    def cmd(script, data, tag, shape=STUDENT, **kwargs):
        return command(script, directory / f"{data}.jsonl", tag, args.device, shape=shape,
                       max_seq_len=128, **kwargs)

    stage("random", "expand_random", STUDENT, None)
    stage("pretrain", "expand_pre", STUDENT, "random",
          cmd("train_pretrain.py", "pretrain", "expand_pre", parent="expand_random",
              epochs=2, batch_size=32, learning_rate=7e-4))
    stage("sft", "expand_sft", STUDENT, "pretrain",
          cmd("train_full_sft.py", "sft", "expand_sft", parent="expand_pre",
              epochs=6, batch_size=32, learning_rate=3e-4))
    stage("rl", "expand_rl", STUDENT, "sft",
          cmd("train_grpo.py", "rl", "expand_rl", parent="expand_sft",
              epochs=3, batch_size=8, learning_rate=7e-5, max_gen_len=30, num_generations=6,
              thinking_ratio=0, beta=0.02, loss_type="grpo", http_demo_rewards=True,
              rollout_temperature=1.05, debug_mode=True, debug_interval=40, log_interval=8))
    stage("teacher", "expand_teacher", TEACHER, "official MiniMind full_sft_768",
          cmd("train_full_sft.py", "teacher_sft", "expand_teacher", shape=TEACHER,
              parent="expand_teacher_base", epochs=3, batch_size=24, learning_rate=1.5e-4))
    print("STAGE teacher_gate", flush=True)
    gate = evaluate(directory / "holdout.jsonl",
                    [("rl", "expand_rl", STUDENT), ("teacher", "expand_teacher", TEACHER)],
                    args.device, length=32)
    save_json(directory / "teacher_gate.json", summarize(gate))
    for label, tag, alpha in (("ce_control", "expand_ce", 1.0), ("distillation", "expand_kd", 0.5)):
        stage(label, tag, STUDENT, "rl",
              cmd("train_distillation.py", "teacher_sft", tag, epochs=1, batch_size=16,
                  learning_rate=5e-5, teacher_use_moe=0, teacher_hidden_size=TEACHER[0],
                  teacher_num_layers=TEACHER[1], from_student_weight="expand_rl",
                  from_teacher_weight="expand_teacher", alpha=alpha, temperature=2))
    records = [(label, row["weight"].removesuffix(f"_{row['shape'][0]}.pth"),
                tuple(row["shape"])) for label, row in manifest["stages"].items()]
    print("STAGE evaluation", flush=True)
    results = evaluate(directory / "holdout.jsonl", records, args.device, length=32)
    save_json(directory / "continuation.json", prose_metrics(directory, manifest["stages"], args.device))
    save_json(directory / "summary.json", summarize(results))
    save_json(directory / "results.json", results)


if __name__ == "__main__":
    main()
