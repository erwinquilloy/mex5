"""LIBERO sim CLI for MolmoAct2.

Examples:
    # full LIBERO-Spatial suite (10 tasks), 5 trials each
    python -m benchmarks.scripts.run_libero_benchmark --suite libero_spatial --trials 5

    # one task, model already outputs [-1, 1] normalized actions
    python -m benchmarks.scripts.run_libero_benchmark \
        --suite libero_goal --tasks 0 --trials 3 --action-scale 1.0

    # list suite tasks (read-only, exits)
    python -m benchmarks.scripts.run_libero_benchmark --list libero_spatial
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from benchmarks.benchmark.sim_runner import run_libero_benchmark


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object", "libero_goal",
                             "libero_10", "libero_90"])
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="task indices within the suite; default: all")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--molmoact-url", default="http://localhost:8000")
    ap.add_argument("--n-action-chunk", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="multiplier applied to model's [dx..dyaw] before env.step "
                         "(LIBERO expects [-1,1]; set to model-specific scale)")
    ap.add_argument("--action-clip", type=float, default=1.0)
    ap.add_argument("--camera-height", type=int, default=256)
    ap.add_argument("--camera-width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="benchmarks/results")
    ap.add_argument("--list", metavar="SUITE", default=None,
                    help="list (task_index, instruction) for SUITE and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.list:
        from benchmarks.benchmark.libero_env import list_suite_tasks
        for i, lang in list_suite_tasks(args.list):
            print(f"{i:3d}  {lang}")
        return 0

    run = run_libero_benchmark(
        suite=args.suite,
        task_indices=args.tasks,
        trials_per_task=args.trials,
        molmoact_url=args.molmoact_url,
        n_action_chunk=args.n_action_chunk,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        action_clip=args.action_clip,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
        results_dir=args.results_dir,
    )
    print("\n===== SUMMARY =====")
    print(json.dumps(run.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
