"""Break down recorded results or run a held-out HTTP challenge on AutoDL."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sharing_from_scratch import evaluate, jsonl, reference, save_json, weights

RECORDED = Path(__file__).with_name("comparison") / "expanded" / "records"

# Each pair changes just the current status or one condition. Keep the original action vocabulary.
CHALLENGE = {
    "auth": ("团队空间续期", "HTTP 401，refresh_token={value}",
             [("present", 401, "refresh"), ("absent", 401, "login")]),
    "retry": ("同步面板", "HTTP 503，method={value}",
              [("GET", 503, "retry"), ("POST", 503, "stop")]),
    "permission": ("素材库读取", "HTTP {value}，method=GET",
                   [("403", 403, "deny"), ("404", 404, "missing")]),
    "form": ("表单提交", "HTTP {value}，method=POST",
             [("400", 400, "fix"), ("409", 409, "conflict")]),
    "rate_limit": ("自动补全请求", "HTTP {value}，method=GET",
                   [("429", 429, "wait"), ("503", 503, "retry")]),
    "save": ("草稿保存", "HTTP {value}，method=POST",
             [("200", 200, "ok"), ("409", 409, "conflict")]),
}
TEMPLATES = (
    "{scene}联调。上一条日志是别的请求，返回 HTTP 200。本次为 {current}。",
    "只处理最后一次{scene}。历史记录 HTTP 404。本次为 {current}。",
    "{scene}的批量日志里，旧请求是 HTTP 429，本次请求是 {current}。请处理本次。",
    "排查{scene}。本次是 {current}。同一面板中的 HTTP 400 来自无关请求。",
)


def challenge_rows():
    rows = []
    for family, (scene, pattern, variants) in CHALLENGE.items():
        for index, template in enumerate(TEMPLATES, 1):
            for value, code, action in variants:
                prompt = template.format(scene=scene, current=pattern.format(value=value))
                rows.append({"family": family, "pair_id": f"{family}-{index}",
                             "prompt": prompt + "只返回一行 JSON，包含 code 和 action。",
                             "answer": reference(code, action)})
    return rows


def group_scores(results, group_key):
    table = {}
    for stage in results:
        groups = {}
        for row in stage["examples"]:
            group = groups.setdefault(group_key(row), {"total": 0, "json": 0, "passed": 0,
                                                        "pairs": {}})
            group["total"] += 1
            group["json"] += row["parsed_json"]
            group["passed"] += row["pass"]
            if row.get("pair_id"):
                group["pairs"].setdefault(row["pair_id"], []).append(row["pass"])
        for values in groups.values():
            pairs = values.pop("pairs").values()
            if pairs:
                if any(len(pair) != 2 for pair in pairs):
                    raise ValueError("Expected exactly two questions per pair")
                values["pairs"] = len(pairs)
                values["pairs_passed"] = sum(all(pair) for pair in pairs)
        table[stage["stage"]] = groups
    return table


def recorded_family(row):
    code = json.loads(row["answer"])["code"]
    return f"{row['kind']}-{code}"


def load_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def evaluate_challenge(run_dir, output, device, extra_stages):
    lineage = load_json(run_dir / "lineage.json")
    stages = []
    for label, entry in lineage["stages"].items():
        shape = tuple(entry["shape"])
        tag = entry["weight"].removesuffix(f"_{shape[0]}.pth")
        stages.append((label, tag, shape))
    for label, tag, width, layers in extra_stages:
        if any(label == stage[0] for stage in stages):
            raise ValueError(f"Duplicate stage label: {label}")
        stages.append((label, tag, (int(width), int(layers))))
    for _, tag, shape in stages:
        if not weights(tag, shape).is_file():
            raise FileNotFoundError(weights(tag, shape))
    output.mkdir(parents=True, exist_ok=True)
    holdout = output / "challenge.jsonl"
    jsonl(holdout, challenge_rows())
    results = evaluate(holdout, stages, device, length=32)
    save_json(output / "results.json", results)
    summary = group_scores(results, lambda row: row["family"])
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    recorded = modes.add_parser("recorded", help="Rescore saved outputs without loading a model")
    recorded.add_argument("--results", type=Path, default=RECORDED / "results.json")
    challenge = modes.add_parser("challenge", help="Evaluate expanded checkpoints on new pairs")
    challenge.add_argument("--run-dir", type=Path, required=True)
    challenge.add_argument("--output", type=Path, required=True)
    challenge.add_argument("--device", default="cuda:0")
    challenge.add_argument("--extra-stage", action="append", nargs=4, default=[],
                           metavar=("LABEL", "WEIGHT_TAG", "WIDTH", "LAYERS"))
    args = parser.parse_args()
    if args.mode == "recorded":
        scores = group_scores(load_json(args.results), recorded_family)
    else:
        if not args.device.startswith("cuda:"):
            parser.error("Run the challenge on the AutoDL CUDA instance")
        if args.output.resolve() == args.run_dir.resolve():
            parser.error("Use a separate --output directory to preserve the expanded results")
        scores = evaluate_challenge(args.run_dir, args.output, args.device, args.extra_stage)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
