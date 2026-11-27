"""Replay the frontend MiniMind experiments from a committed Git snapshot."""

import argparse
from datetime import datetime
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path(__file__).with_name("inputs.json")
EXPERIMENTS = ("smoke", "official", "from-scratch", "expanded")
SNAPSHOT_PATHS = (
    "dataset", "model", "scripts", "trainer",
    "experiments/pretrain/train_pretrain.py",
    "experiments/sft/train_full_sft.py",
    "experiments/rl/train_grpo.py",
    "experiments/distillation/train_distillation.py",
    "experiments/distillation/distill_1b.py",
    "experiments/distillation/measure_teacher_kl.py",
    "experiments/evaluate_scenarios.py",
    "experiments/holdout_ce.py",
    "experiments/measure_teacher_kl.py",
    "experiments/distill_1b.py",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def fetch(name, cache):
    entry = json.loads(INPUTS.read_text(encoding="utf8"))[name]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if path.exists():
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for cached file: {path}. Move the file and try again.")
        return path
    errors = []
    for url in entry["urls"]:
        descriptor, filename = tempfile.mkstemp(dir=cache)
        temporary = Path(filename)
        try:
            headers = {"User-Agent": "minimind-frontend/reproduction"}
            if "lines" in entry:
                headers["Range"] = "bytes=0-4194303"
            with os.fdopen(descriptor, "wb") as stream, urlopen(
                Request(url, headers=headers), timeout=45
            ) as response:
                if "lines" in entry:
                    count = 0
                    for line in islice(response, entry["lines"]):
                        if "conversations" not in json.loads(line):
                            raise ValueError("Invalid SFT input")
                        stream.write(line)
                        count += 1
                    if count != entry["lines"]:
                        raise ValueError(f"Only {count} SFT lines received")
                else:
                    shutil.copyfileobj(response, stream, length=1 << 20)
            if sha256(temporary) != entry["sha256"]:
                raise ValueError("Downloaded content does not match the pinned SHA-256")
            temporary.replace(path)
            print(f"Verified {name} from {url}", flush=True)
            return path
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {error}")
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not recover {name}.\n" + "\n".join(errors))


def git(*command):
    return subprocess.check_output(["git", "-C", str(ROOT), *command], text=True).strip()


def prepare(run_dir, experiment, device, resume=False):
    metadata = run_dir / "run.json"
    source = run_dir / "source"
    records = run_dir / "records"
    if resume:
        if not metadata.is_file():
            raise ValueError("--resume requires the original run directory and its run.json")
        state = json.loads(metadata.read_text(encoding="utf8"))
        if state["experiment"] != experiment or state["device"] != device:
            raise ValueError("The experiment and device must match the original run")
        if not source.is_dir() or not records.is_dir():
            raise ValueError("The original source snapshot and records directory must be present")
        return source, records, state
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"{run_dir} is not empty. Choose another directory or use --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    revision = git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(dir=run_dir) as staging:
        archive = Path(staging) / "source.zip"
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=zip", f"--output={archive}",
             revision, *SNAPSHOT_PATHS],
            check=True,
        )
        extracted = Path(staging) / "source"
        with ZipFile(archive) as snapshot:
            snapshot.extractall(extracted)
        extracted.replace(source)
    records.mkdir()
    state = {"experiment": experiment, "code_revision": revision, "device": device, "complete": False}
    write_json(metadata, state)
    return source, records, state


def launch(command, directory, log, env):
    print(f"$ {shlex.join(str(item) for item in command)}", flush=True)
    with log.open("w", encoding="utf8") as output:
        output.write(f"$ {shlex.join(str(item) for item in command)}\n")
        output.flush()
        with subprocess.Popen(command, cwd=directory, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace") as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)


def preflight(device, records, env):
    probe = (
        "import sys,datasets,transformers,torch; from transformers import PreTrainedModel; "
        "assert torch.cuda.is_available(), 'Install a CUDA build of PyTorch'; "
        f"torch.cuda.set_device({device!r}); "
        "assert torch.cuda.is_bf16_supported(), 'A GPU supporting BF16 is required'; "
        "print(sys.version); print('PyTorch',torch.__version__,'CUDA',torch.version.cuda); "
        "print('GPU',torch.cuda.get_device_name()); "
        "print('datasets',datasets.__version__,'transformers',transformers.__version__)"
    )
    launch([sys.executable, "-c", probe], ROOT, records / "environment.log", env)


def official(source, records, cache, device, env, state):
    weights = source / "out"
    weights.mkdir(exist_ok=True)
    for name in ("pretrain_768.pth", "full_sft_768.pth", "grpo_768.pth", "grpo_768_moe.pth"):
        target = weights / name
        asset = fetch(name, cache)
        if not target.is_file():
            shutil.copyfile(asset, target)
        elif sha256(target) != sha256(asset):
            raise RuntimeError(f"Original model was modified: {target}")
    data = fetch("sharing_kd_512.jsonl", cache)
    common = [
        "--device", device, "--dtype", "bfloat16", "--epochs", "1", "--batch_size", "2",
        "--num_workers", "0", "--max_seq_len", "512", "--accumulation_steps", "8",
        "--save_interval", "64", "--log_interval", "16", "--teacher_use_moe", "1",
        "--from_student_weight", "grpo", "--from_teacher_weight", "grpo",
        "--data_path", str(data), "--temperature", "2",
    ]
    saved = state.setdefault("official_weights", {})
    for tag, alpha in (("cloud_ce", "1"), ("cloud_kd", "0.5")):
        result_path = weights / f"{tag}_768.pth"
        if tag in saved:
            if not result_path.is_file() or sha256(result_path) != saved[tag]:
                raise RuntimeError(f"Recorded checkpoint was modified: {result_path}")
            continue
        launch([sys.executable, "-u", "train_distillation.py", *common,
                "--alpha", alpha, "--save_weight", tag], source / "trainer", records / f"{tag}.log", env)
        saved[tag] = sha256(result_path)
        write_json(records.parent / "run.json", state)
    for tag, mode in (("stage_outputs", "native"), ("stage_outputs_all_chat", "chat")):
        launch([sys.executable, "scripts/compare_stage_outputs.py", "--device", device,
                "--template", mode, "--include-ce-control", "--output", str(records/f"{tag}.json")],
               source, records/f"{tag}.txt", env)


def execute(args, source, records, state):
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(records.parent / "hf"),
        "HF_DATASETS_CACHE": str(records.parent / "hf" / "datasets"),
    }
    preflight(args.device, records, env)
    if args.experiment == "official":
        official(source, records, args.assets_dir, args.device, env, state)
        return
    runners = {"smoke": "sharing_demo.py", "from-scratch": "sharing_from_scratch.py",
               "expanded": "sharing_expanded.py"}
    command = [sys.executable, "-u", f"scripts/{runners[args.experiment]}",
               "--device", args.device, "--run-dir", str(records)]
    if args.experiment == "smoke":
        command.extend(["--epochs", "1"])
    elif (records / "lineage.json").is_file():
        command.append("--resume")
    if args.experiment == "expanded":
        teacher = fetch("full_sft_768.pth", args.assets_dir)
        command.extend(["--teacher-checkpoint", str(teacher)])
    launch(command, source, records / "runner.log", env)
    if args.experiment == "from-scratch":
        launch([sys.executable, "experiments/holdout_ce.py", "--repo", str(source),
                "--records", str(records), "--device", args.device], source, records / "reference_ce.log", env)
    if args.experiment == "expanded":
        launch([sys.executable, "experiments/measure_teacher_kl.py", "--repo", str(source),
                "--records", str(records), "--device", args.device], source, records / "teacher_kl.log", env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / ".assets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Archive the committed source and create a run directory without downloads or GPU use")
    args = parser.parse_args()
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must look like cuda:0")
    if args.run_dir is None:
        args.run_dir = ROOT / "runs" / f"{args.experiment}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir = args.run_dir.resolve()
    args.assets_dir = args.assets_dir.resolve()
    try:
        source, records, state = prepare(run_dir, args.experiment, args.device, args.resume)
        print(f"Run directory: {run_dir}", flush=True)
        if args.prepare_only:
            return
        if state["complete"]:
            print("This run already finished. Select a fresh --run-dir to repeat it.")
            return
        execute(args, source, records, state)
        state["complete"] = True
        write_json(run_dir / "run.json", state)
        print(f"Results: {records}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
