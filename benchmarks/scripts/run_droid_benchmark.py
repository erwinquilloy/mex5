"""Reproduce Table 6 of arXiv:2605.02881 with MolmoAct2-DROID on the real Franka.

Two robot transports are supported:

  --transport fci   panda_py/libfranka direct (default; FRANKA_HOST = robot FCI IP).
  --transport rest  motion_server REST (FRANKA_REST_HOST = the box running
                    motion_server, repo default 192.168.2.1).

Usage:
    # full Table 6 over FCI (5 tasks x 15 trials)
    export FRANKA_HOST=192.168.1.131           # robot FCI IP
    export FRANKA_BENCH_EXT_INDEX=0            # USB webcam /dev/video0
    python -m benchmarks.scripts.run_droid_benchmark --molmoact-url http://localhost:8000

    # same, but routed through motion_server REST
    export FRANKA_REST_HOST=192.168.2.1        # motion_server, NOT the robot
    python -m benchmarks.scripts.run_droid_benchmark --transport rest --rest-step-time-s 2.5

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
    ap.add_argument("--grasp-commit-grip-frac", type=float, default=0.5,
                    help="when at least this fraction of the chunk commands gripper-close, run the WHOLE chunk instead of just --exec-rows (don't interrupt a grasp).")
    ap.add_argument("--fine-refinement-travel-rad", type=float, default=0.2,
                    help="if total joint-space travel within a chunk is below this (radians), treat as fine refinement and only run --exec-rows. Larger chunks are run full open-loop. Default 0.2 ~= 11 deg total chunk travel.")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="override the per-task safety cap on action chunks per trial (default per droid_tasks.py: 30). Raise this when using small --exec-rows so the trial doesn't run out of budget before the policy finishes.")
    ap.add_argument("--results-dir", default="benchmarks/results")
    ap.add_argument("--transport", choices=["fci", "rest", "mcp"], default="fci",
                    help="fci = direct panda_py/libfranka (default, joint-position streaming). "
                         "rest = motion_server REST API (cartesian xyz + Euler-delta-deg per row). "
                         "mcp = same as rest, but routed through franka/python/mcp_server.py "
                         "(fastmcp -> REST). Useful when an MCP agent should drive the same arm.")
    ap.add_argument("--rest-host", default=None,
                    help="motion_server host for --transport=rest (or FRANKA_REST_HOST). "
                         "Repo default in existing clients: 192.168.2.1.")
    ap.add_argument("--rest-port", type=int, default=34568,
                    help="motion_server port (default 34568).")
    ap.add_argument("--rest-step-time-s", type=float, default=2.5,
                    help="Commanded per-row motion time on REST/MCP paths (sec). "
                         "Server now allows 0.5s+ after the patch; default 2.5.")
    ap.add_argument("--mcp-url", default=None,
                    help="MCP server URL for --transport=mcp (or FRANKA_MCP_URL). "
                         "Default per franka/python/mcp_server.py: http://<host>:8085/franka.")
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
        grasp_commit_grip_frac=args.grasp_commit_grip_frac,
        fine_refinement_travel_rad=args.fine_refinement_travel_rad,
        max_chunks=args.max_chunks,
        transport=args.transport,
        rest_host=args.rest_host,
        rest_port=args.rest_port,
        rest_step_time_s=args.rest_step_time_s,
        mcp_url=args.mcp_url,
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
