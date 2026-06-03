"""Legacy CLI entrypoint (cartesian-delta runner; superseded by run_droid_benchmark.py).

Examples:
    # full spatial+goal subset, 5 trials each
    python -m benchmarks.scripts.run_benchmark

    # one task, dry run with file camera, no human-in-the-loop success
    FRANKA_BENCH_IMAGES=./fixtures \
        python -m benchmarks.scripts.run_benchmark \
            --tasks spatial_red_left --trials 2 --no-interactive
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from benchmarks.benchmark.runner import run_benchmark


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task ids; defaults to all (spatial + goal subsets)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--franka-ip", default="192.168.2.1")
    ap.add_argument("--franka-port", type=int, default=34568)
    ap.add_argument("--molmoact-url", default="http://localhost:8000")
    ap.add_argument("--n-action-chunk", type=int, default=1,
                    help="how many actions to request per inference")
    ap.add_argument("--step-time", type=float, default=1.0,
                    help="commanded motion duration per action step (sec)")
    ap.add_argument("--results-dir", default="benchmarks/results")
    ap.add_argument("--no-interactive", action="store_true",
                    help="skip human-in-the-loop success prompt (success defaults to False)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run = run_benchmark(
        task_ids=args.tasks,
        trials_per_task=args.trials,
        franka_ip=args.franka_ip,
        franka_port=args.franka_port,
        molmoact_url=args.molmoact_url,
        n_action_chunk=args.n_action_chunk,
        step_time_s=args.step_time,
        results_dir=args.results_dir,
        interactive_success=not args.no_interactive,
    )
    print("\n===== SUMMARY =====")
    print(json.dumps(run.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
