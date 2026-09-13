#!/usr/bin/env python3
"""Full-1024 completion-aligned MUSE depth/capacity sweep.

This is a thin, provenance-aware specialization of the frozen completion-
aligned implementation. It preserves six training exposures from the 128-row
pilot by scaling A1/A2 steps fourfold for 512 rows per corpus.
"""
import os
from pathlib import Path
import subprocess

import authorize_muse_recap_full1024 as authorization
import sweep_muse_recap_completion_aligned as aligned


aligned.VERSION = "muse-completion-aligned-full1024-v1"
aligned.VARIANTS = {
    "d4": {"depth": 4, "rank": 32, "a1_steps": 1536, "a2_steps": 1152},
    "d6": {"depth": 6, "rank": 32, "a1_steps": 1536, "a2_steps": 1152},
}
aligned.POINTS = {
    "light": (0.50, 0.50, 0.001),
    "a1light": (0.75, 0.50, 0.001),
    "balanced": (0.75, 0.75, 0.001),
    "strong": (1.00, 0.75, 0.001),
}
# Both validate_source() and train() call through this module reference.
aligned.offline.validate_authorized_rows = authorization.validate_rows


def run_child(args, mode, variant, corpus, values, gpu):
    """Re-enter this specialization for full-1024 training children."""
    run = aligned.variant_run(args.run, variant)
    label = values.get("role", values.get("point"))
    log = args.run / "logs" / f"{mode}_{variant}_{corpus}_{label}.log"
    log.parent.mkdir(exist_ok=True)
    if mode == "train":
        command = [
            args.train_python,
            "-u",
            str(Path(__file__).resolve()),
            "train",
            "--run",
            str(run),
            "--corpus",
            corpus,
        ]
    else:
        command = [
            args.train_python,
            "-u",
            str(Path(aligned.base.__file__).resolve()),
            "evaluate",
            "--run",
            str(run),
            "--corpus",
            corpus,
            "--eval-python",
            args.eval_python,
        ]
    for key, value in values.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    aligned.base.event(
        f"aligned_{mode}_start variant={variant} corpus={corpus} "
        f"item={label} GPU={gpu}"
    )
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    with log.open("a") as stream:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    aligned.base.event(
        f"aligned_{mode}_done variant={variant} corpus={corpus} "
        f"item={label} GPU={gpu}"
    )


aligned.run_child = run_child


if __name__ == "__main__":
    aligned.main()
