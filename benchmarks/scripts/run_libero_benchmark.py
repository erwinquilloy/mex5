"""LIBERO sim CLI for MolmoAct2-LIBERO (in-process, runs on the HPC GPU).

Examples:
    # full libero_spatial suite (10 tasks), 5 trials each
    python -m benchmarks.scripts.run_libero_benchmark --suite libero_spatial --trials 5

    # one task, smoke test
    python -m benchmarks.scripts.run_libero_benchmark \
        --suite libero_goal --tasks 0 --trials 1

    # list suite tasks (read-only, exits)
    python -m benchmarks.scripts.run_libero_benchmark --list libero_spatial
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object", "libero_goal",
                             "libero_10", "libero_90"])
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="task indices within the suite; default: all")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--model-id", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num-steps", type=int, default=10,
                    help="flow-matching steps the model takes per inference call")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="environment steps cap per trial")
    ap.add_argument("--camera-height", type=int, default=256)
    ap.add_argument("--camera-width", type=int, default=256)
    ap.add_argument("--no-cuda-graph", action="store_true",
                    help="disable CUDA graphs (reduces GPU memory, slower per-step)")
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

    # Heavy imports deferred so --list / --help don't pull in torch.
    from benchmarks.benchmark.sim_runner import run_libero_benchmark
    run = run_libero_benchmark(
        suite=args.suite,
        task_indices=args.tasks,
        trials_per_task=args.trials,
        model_id=args.model_id,
        dtype=args.dtype,
        num_steps=args.num_steps,
        max_steps=args.max_steps,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        enable_cuda_graph=not args.no_cuda_graph,
        seed=args.seed,
        results_dir=args.results_dir,
    )
    print("\n===== SUMMARY =====")
    print(json.dumps(run.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
