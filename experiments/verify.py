"""Verify the written evidence and regenerated fixtures without loading a model."""

import ast
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile

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
    question = results["pretrain"]["examples"][274]["prompt"]
    for stage in ("pretrain", "sft", "rl", "distillation"):
        row = results[stage]["examples"][274]
        assert row["prompt"] == question
        doc = (ROOT / "docs/share.md").read_text(encoding="utf8")
        assert question in doc and row["generated"] in doc


def check_links():
    for source in (ROOT / "README.md", ROOT / "README_en.md", ROOT / "docs/share.md",
                   RECORDS / "README.md"):
        text = source.read_text(encoding="utf8")
        for match in re.finditer(r"\]\((?:<([^>]+)>|([^)\n]+))\)", text):
            href = match[1] or match[2]
            if href.startswith(("http://", "https://", "#")):
                continue
            assert (source.parent / href.split("#", 1)[0]).exists(), (source.name, href)
    for source in (Path(__file__).with_name("run.py"), Path(__file__).with_name("holdout_ce.py"),
                   Path(__file__).with_name("measure_teacher_kl.py")):
        ast.parse(source.read_text(encoding="utf8"))


if __name__ == "__main__":
    check_data()
    check_results()
    check_links()
    print("PASS: deterministic datasets, 322 recorded outputs per stage, Markdown links, and Python syntax.")
