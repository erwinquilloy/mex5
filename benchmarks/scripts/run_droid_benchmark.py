"""Reproduce Table 6 of arXiv:2605.02881 with MolmoAct2-DROID on the real Franka.

Usage:
    # full Table 6 (5 tasks x 15 trials)
    export FRANKA_HOST=192.168.1.131
    export FRANKA_BENCH_EXT_INDEX=0           # USB webcam /dev/video0
    # (omit FRANKA_BENCH_WRIST_SERIAL to use whatever D457 is connected)
    python -m benchmarks.scripts.run_droid_benchmark --molmoact-url http://localhost:8000

    # quick smoke test: one trial of one task
    python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from benchmarks.benchmark.droid_runner import run_droid_benchmark


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task ids; defaults to all 5 from Table 6")
    ap.add_argument("--trials", type=int, default=None,
                    help="override trials per task (default 15 per Table 6)")
    ap.add_argument("--molmoact-url", default="http://localhost:8000")
    ap.add_argument("--franka-host", default=None,
                    help="Franka IP/hostname (or set FRANKA_HOST)")
    ap.add_argument("--num-steps", type=int, default=10,
                    help="flow-matching steps the model takes per inference call")
    ap.add_argument("--chunk-step-dt", type=float, default=0.1,
                    help="commanded duration per action row (sec)")
    ap.add_argument("--exec-rows", type=int, default=3,
                    help="rows of each predicted chunk to execute before re-querying with a fresh frame (receding-horizon). 0 = run the whole chunk (old behavior).")
    ap.add_argument("--results-dir", default="benchmarks/results")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    run = run_droid_benchmark(
        task_ids=args.tasks,
        trials_per_task=args.trials,
        molmoact_url=args.molmoact_url,
        franka_host=args.franka_host,
        num_steps=args.num_steps,
        chunk_step_dt_s=args.chunk_step_dt,
        results_dir=args.results_dir,
        exec_rows=args.exec_rows,
    )
    summary = run.summary()
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    # Table 6 comparison
    from benchmarks.benchmark.droid_tasks import by_id
    print("\n===== Table 6 comparison (MolmoAct2-DROID) =====")
    print(f"{'task':<28} {'paper':>8} {'ours':>8} {'n':>4}")
    for task_id, per in summary["per_task"].items():
        try:
            paper = by_id(task_id).paper_success_rate
            print(f"{task_id:<28} {paper:>7.1f}% {per['success_rate']*100:>7.1f}% {per['n']:>4}")
        except KeyError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
