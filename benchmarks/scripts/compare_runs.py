"""Side-by-side comparison of N benchmark result files.

Each result file is the JSON dumped by RunRecord.dump (one per run). Use this
to A/B/C the same task suite across transports (fci vs rest vs mcp) or any
other axis. Pass file paths explicitly, or a glob like benchmarks/results/*.json.

Example:
    python -m benchmarks.scripts.compare_runs \
        benchmarks/results/droid-20260603-*-fci.json \
        benchmarks/results/droid-20260603-*-rest.json \
        benchmarks/results/droid-20260603-*-mcp.json
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics
import sys
from pathlib import Path


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = int(round((len(xs) - 1) * q))
    return xs[i]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _summarize(run: dict) -> dict:
    trials = run.get("trials", [])
    steps = [s for t in trials for s in t.get("steps", [])]
    per_task: dict[str, dict] = {}
    for t in trials:
        d = per_task.setdefault(t["task_id"], {"n": 0, "successes": 0})
        d["n"] += 1
        d["successes"] += int(t.get("success", False))
    for d in per_task.values():
        d["success_rate"] = d["successes"] / d["n"] if d["n"] else 0.0

    def col(key: str) -> list[float]:
        return [float(s.get(key, 0.0)) for s in steps]

    return {
        "n_trials": len(trials),
        "n_steps": len(steps),
        "overall_success_rate": (
            sum(int(t.get("success", False)) for t in trials) / len(trials) if trials else 0.0
        ),
        "per_task": per_task,
        "infer_server_ms": col("infer_server_ms"),
        "e2e_ms": col("e2e_ms"),
        "motion_rest_ms": col("motion_rest_ms"),
    }


def _transport_tag(endpoint: str) -> str:
    if ":" in endpoint:
        return endpoint.split(":", 1)[0]
    return "?"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


_MS_COL_W = 8 + 1 + 8 + 1 + 8   # "mean p50 p95" with single-space separators


def _fmt_ms_block(xs: list[float]) -> str:
    if not xs:
        return f"{'-':>8} {'-':>8} {'-':>8}"
    return (
        f"{statistics.fmean(xs):8.0f} "
        f"{_percentile(xs, 0.50):8.0f} "
        f"{_percentile(xs, 0.95):8.0f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="result JSON files (globs OK)")
    ap.add_argument("--results-dir", default="benchmarks/results",
                    help="if no paths given, compare all *.json under this dir")
    ap.add_argument("--label", action="append", default=[],
                    help="optional column labels, one per input file (in order). "
                         "Defaults to <transport>:<run_id_tail>.")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    if args.paths:
        for p in args.paths:
            expanded = _glob.glob(p)
            if expanded:
                paths.extend(Path(x) for x in expanded)
            elif Path(p).exists():
                paths.append(Path(p))
            else:
                print(f"warn: no match for {p}", file=sys.stderr)
    else:
        paths = sorted(Path(args.results_dir).glob("*.json"))

    if not paths:
        print("no result files found", file=sys.stderr)
        return 1

    runs = []
    for p in paths:
        try:
            r = _load(p)
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        s = _summarize(r)
        s["_path"] = p
        s["_run_id"] = r.get("run_id", p.stem)
        s["_endpoint"] = r.get("franka_endpoint", "?")
        s["_transport"] = _transport_tag(s["_endpoint"])
        runs.append(s)

    if not runs:
        return 1

    labels: list[str] = []
    for i, run in enumerate(runs):
        if i < len(args.label):
            labels.append(args.label[i])
        else:
            tail = run["_run_id"].split("-")[-1] if "-" in run["_run_id"] else run["_run_id"]
            labels.append(f"{run['_transport']}:{tail}")

    col_w = max(14, max(len(l) for l in labels) + 1)

    print("\n===== Inputs =====")
    for run, lab in zip(runs, labels):
        print(f"  [{lab}] {run['_path']} ({run['_endpoint']}, "
              f"n_trials={run['n_trials']}, n_steps={run['n_steps']})")

    print("\n===== Overall success =====")
    print(f"{'metric':<24} " + "".join(f"{l:>{col_w}}" for l in labels))
    print(f"{'overall_success_rate':<24} "
          + "".join(f"{_fmt_pct(r['overall_success_rate']):>{col_w}}" for r in runs))
    print(f"{'n_trials':<24} "
          + "".join(f"{r['n_trials']:>{col_w}d}" for r in runs))

    all_tasks: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for t in run["per_task"]:
            if t not in seen:
                seen.add(t)
                all_tasks.append(t)
    if all_tasks:
        print("\n===== Per-task success =====")
        print(f"{'task':<24} " + "".join(f"{l:>{col_w}}" for l in labels))
        for t in all_tasks:
            cells = []
            for run in runs:
                d = run["per_task"].get(t)
                if d is None:
                    cells.append(f"{'-':>{col_w}}")
                else:
                    cell_text = _fmt_pct(d["success_rate"]) + f" ({d['n']})"
                    cells.append(f"{cell_text:>{col_w}}")
            print(f"{t:<24} " + "".join(cells))

    print("\n===== Latency (mean / p50 / p95, ms) =====")
    sep = "  "
    print(f"{'metric':<18} " + sep.join(f"{l:>{_MS_COL_W}}" for l in labels))
    sub = f"{'mean':>8} {'p50':>8} {'p95':>8}"
    print(f"{' ':<18} " + sep.join(sub for _ in labels))
    for key in ("infer_server_ms", "e2e_ms", "motion_rest_ms"):
        cells = [_fmt_ms_block(run[key]) for run in runs]
        print(f"{key:<18} " + sep.join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
