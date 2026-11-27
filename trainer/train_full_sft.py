"""Compatibility entry point for the stage training script."""

from importlib import import_module
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("experiments.sft.train_full_sft", run_name="__main__")
else:
    sys.modules[__name__] = import_module("experiments.sft.train_full_sft")
