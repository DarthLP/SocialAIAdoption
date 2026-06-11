"""
Script summary:
Sequential post-estimation pipeline after the post-window contamination fix baseline DiD rerun.

Functionality:
- Optionally waits for an in-flight baseline ``did_event_study.py`` process to finish.
- Runs, in order: exbantopic DiD (full bootstrap + figures), exbantopic comparison,
  design-based placebo-in-time, scan-wide BH audit (F10), and post-window reconcile report.
- Baseline ``did_event_study.py`` must already include bootstrap (WCB + placebo-in-space);
  this script does not re-estimate baseline rows.

How to apply/run:
  # After baseline did_event_study is running or complete:
  .venv/bin/python scripts/diagnostics/run_post_window_inference_pipeline.py \\
    --config config/italy_polarization_setup.yaml --wait-pid 74450

  # Baseline already finished:
  .venv/bin/python scripts/diagnostics/run_post_window_inference_pipeline.py \\
    --config config/italy_polarization_setup.yaml --skip-wait
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute path to project root.
    """
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _parse_args() -> argparse.Namespace:
    """Function summary: CLI for ordered post-window inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Run exbantopic DiD, placebo-in-time, scan audit, reconcile after baseline."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--wait-pid",
        type=int,
        default=None,
        help="Poll until this PID exits (baseline did_event_study).",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Do not wait; assume baseline did_event_study already finished.",
    )
    parser.add_argument(
        "--skip-exbantopic",
        action="store_true",
        help="Skip exbantopic did_event_study and compare_exbantopic steps.",
    )
    parser.add_argument(
        "--skip-placebo-in-time",
        action="store_true",
        help="Skip placebo_in_time.py.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=120,
        help="Seconds between wait-pid polls.",
    )
    return parser.parse_args()


def _pid_alive(pid: int) -> bool:
    """Function summary: True when process pid is still running.

    Parameters:
    - pid: OS process id.

    Returns:
    - True if kill(pid, 0) succeeds.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_pid(pid: int, poll_seconds: int) -> None:
    """Function summary: block until pid exits, logging periodically.

    Parameters:
    - pid: process to wait on.
    - poll_seconds: sleep between checks.
    """
    print(f"[pipeline] waiting for baseline did_event_study pid={pid}", flush=True)
    while _pid_alive(pid):
        time.sleep(poll_seconds)
        print(f"[pipeline] still waiting on pid={pid} ...", flush=True)
    print(f"[pipeline] pid={pid} finished", flush=True)


def _run_step(label: str, cmd: List[str]) -> None:
    """Function summary: run one subprocess step; raise on non-zero exit.

    Parameters:
    - label: log label.
    - cmd: argv list (executable + args).
    """
    print(f"[pipeline] START {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"[pipeline] DONE {label}", flush=True)


def main() -> None:
    """Function summary: execute ordered post-baseline inference pipeline."""
    args = _parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Missing venv python: {PYTHON}")

    if args.wait_pid and not args.skip_wait:
        _wait_for_pid(args.wait_pid, args.poll_seconds)
    elif not args.skip_wait:
        print(
            "[pipeline] no --wait-pid; proceeding immediately (use --skip-wait if baseline done)",
            flush=True,
        )

    cfg = args.config

    if not args.skip_exbantopic:
        _run_step(
            "exbantopic did_event_study (bootstrap)",
            [
                str(PYTHON),
                "scripts/analysis/did_event_study.py",
                "--config",
                cfg,
                "--exclude-ban-topic",
                "--families",
                "lexical,semantic_axis",
            ],
        )
        _run_step(
            "compare_exbantopic_coefficients",
            [
                str(PYTHON),
                "scripts/analysis/compare_exbantopic_coefficients.py",
                "--config",
                cfg,
            ],
        )

    if not args.skip_placebo_in_time:
        _run_step(
            "placebo_in_time",
            [str(PYTHON), "scripts/analysis/placebo_in_time.py", "--config", cfg],
        )

    _run_step(
        "export_scan_audit",
        [str(PYTHON), "scripts/diagnostics/export_scan_audit.py", "--config", cfg],
    )
    _run_step(
        "reconcile_post_window_fix",
        [str(PYTHON), "scripts/diagnostics/reconcile_post_window_fix.py", "--config", cfg],
    )
    print("[pipeline] all steps complete", flush=True)


if __name__ == "__main__":
    main()
