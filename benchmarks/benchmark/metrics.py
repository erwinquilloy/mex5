"""Latency timers + per-task / per-step records."""
from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    step: int
    camera_ms: float
    infer_server_ms: float
    infer_rtt_ms: float
    motion_rest_ms: float
    motion_cmd_ms: float        # commanded motion duration
    e2e_ms: float               # capture -> motion REST return
    action: list[float]
    instruction: str


@dataclass
class TrialRecord:
    task_id: str
    trial: int
    success: bool
    n_steps: int
    wallclock_s: float
    steps: list[StepRecord] = field(default_factory=list)
    notes: str = ""


@dataclass
class RunRecord:
    run_id: str
    started_at: float
    model_id: str
    franka_endpoint: str
    molmoact_endpoint: str
    trials: list[TrialRecord] = field(default_factory=list)

    def dump(self, out_dir: str | Path) -> Path:
        out = Path(out_dir) / f"{self.run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2))
        return out

    def summary(self) -> dict[str, Any]:
        steps = [s for t in self.trials for s in t.steps]
        def p(xs, q):
            xs = sorted(xs)
            if not xs:
                return 0.0
            i = int(round((len(xs) - 1) * q))
            return xs[i]
        per_task: dict[str, dict[str, Any]] = {}
        for t in self.trials:
            d = per_task.setdefault(t.task_id, {"n": 0, "successes": 0})
            d["n"] += 1
            d["successes"] += int(t.success)
        for d in per_task.values():
            d["success_rate"] = d["successes"] / d["n"] if d["n"] else 0.0
        infer = [s.infer_server_ms for s in steps]
        e2e = [s.e2e_ms for s in steps]
        return {
            "n_trials": len(self.trials),
            "n_steps": len(steps),
            "overall_success_rate": (
                sum(t.success for t in self.trials) / len(self.trials) if self.trials else 0.0
            ),
            "per_task": per_task,
            "infer_server_ms": {
                "mean": statistics.fmean(infer) if infer else 0.0,
                "p50": p(infer, 0.50), "p95": p(infer, 0.95), "p99": p(infer, 0.99),
            },
            "e2e_ms": {
                "mean": statistics.fmean(e2e) if e2e else 0.0,
                "p50": p(e2e, 0.50), "p95": p(e2e, 0.95), "p99": p(e2e, 0.99),
            },
        }


class Stopwatch:
    def __init__(self) -> None:
        self.ms: float = 0.0
    @contextmanager
    def __call__(self):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.ms = (time.perf_counter() - t0) * 1000.0
