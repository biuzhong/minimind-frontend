"""Verify the written evidence and regenerated fixtures without loading a model."""

import ast
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile

from evaluate_scenarios import challenge_rows, group_scores, recorded_family

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECORDS = Path(__file__).with_name("recorded")


def read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf8"))


def check_data():
    for label, script, builder in (
        ("small", "sharing_from_scratch.py", "make_data"),
        ("expanded", "sharing_expanded.py", "build"),
    ):
        recorded = RECORDS / label / "records"
        run_data = runpy.run_path(str(ROOT / "scripts" / script))
        with tempfile.TemporaryDirectory() as temporary:
            created = Path(temporary)
            counts = run_data[builder](created)
            assert counts == read(recorded, "lineage.json")["data"], label
            for item in created.glob("*.jsonl"):
                expected = (recorded / item.name).read_text(encoding="utf8")
                assert item.read_text(encoding="utf8") == expected, item.name
    assert "torch" not in sys.modules


def check_results():
    expanded = RECORDS / "expanded" / "records"
    results = {r["stage"]: r for r in read(expanded, "results.json")}
    assert read(expanded, "teacher_kl.json")["reference_tokens"]["all"] == 5330
    assert [results[stage]["pass"] for stage in
            ("random", "pretrain", "sft", "rl", "teacher", "ce_control", "distillation")] == [
                0, 0, 238, 227, 322, 277, 270
            ]
    doc = (ROOT / "docs/share.md").read_text(encoding="utf8")
    for index in (274, 106, 166, 198, 292, 308):
        question = results["pretrain"]["examples"][index]["prompt"]
        assert question in doc
        for stage in ("pretrain", "sft", "rl", "distillation"):
            row = results[stage]["examples"][index]
            assert row["prompt"] == question
            if index == 274 or stage != "pretrain":
                assert row["generated"] in doc
    breakdown = group_scores(results.values(), recorded_family)
    assert breakdown["distillation"]["policy-401"]["passed"] == 24
    assert breakdown["distillation"]["policy-503"]["passed"] == 16
    assert breakdown["ce_control"]["policy-503"]["passed"] == 25


def check_challenge():
    rows = challenge_rows()
    assert len(rows) == len({row["prompt"] for row in rows}) == 48
    records = RECORDS / "expanded" / "records"
    existing = set()
    for name in ("sft", "teacher_sft", "rl", "holdout"):
        for line in (records / f"{name}.jsonl").read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            existing.add(row["prompt"] if "prompt" in row else row["conversations"][1]["content"])
    assert existing.isdisjoint(row["prompt"] for row in rows)
    examples = [{**row, "parsed_json": True, "pass": index % 8 != 1} for index, row in enumerate(rows)]
    scores = group_scores([{"stage": "scoring fixture", "examples": examples}],
                          lambda row: row["family"])["scoring fixture"]
    assert len(scores) == 6
    assert all(score == {"total": 8, "json": 8, "passed": 7, "pairs": 4, "pairs_passed": 3}
               for score in scores.values())


def check_links():
    for source in (ROOT / "README.md", ROOT / "README_en.md", ROOT / "docs/share.md",
                   RECORDS / "README.md"):
        text = source.read_text(encoding="utf8")
        for match in re.finditer(r"\]\((?:<([^>]+)>|([^)\n]+))\)", text):
            href = match[1] or match[2]
            if href.startswith(("http://", "https://", "#")):
                continue
            assert (source.parent / href.split("#", 1)[0]).exists(), (source.name, href)
    for source in Path(__file__).parent.glob("*.py"):
        ast.parse(source.read_text(encoding="utf8"))


if __name__ == "__main__":
    check_data()
    check_results()
    check_challenge()
    check_links()
    print("PASS: datasets, 322 recorded questions per stage, 48 new challenge questions, links, and syntax.")
