"""HTTP client for the upstream MolmoAct2-DROID server.

The upstream `examples/droid/host_server_droid.py` exposes:
  GET  /act          health blob (repo_id, norm_tag, device, dtype)
  GET  /healthz      liveness
  POST /act          json_numpy-encoded {external_cam, wrist_cam, instruction,
                     state, num_steps?, enable_cuda_graph?}
                     -> {actions: (N, 8) float32, dt_ms: float}

`json_numpy` adds a default/object_hook pair so numpy arrays serialize as
{"__numpy__": "<b64 bytes>", "dtype": ..., "shape": ...} -- both sides must use
the same package version, which is why we depend on the pinned `json-numpy`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import json_numpy
import numpy as np
import requests


@dataclass
class DroidPrediction:
    actions: np.ndarray         # (N, 8) float32
    server_dt_ms: float         # GPU-side time the server measured
    rtt_ms: float               # full HTTP round-trip
    n_bytes_sent: int


class DroidClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def health(self) -> dict:
        r = self._session.get(f"{self.base_url}/act", timeout=5)
        r.raise_for_status()
        return r.json()

    def healthz(self) -> dict:
        r = self._session.get(f"{self.base_url}/healthz", timeout=5)
        r.raise_for_status()
        return r.json()

    def act(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = 10,
        enable_cuda_graph: bool = False,
    ) -> DroidPrediction:
        if external_cam.dtype != np.uint8 or wrist_cam.dtype != np.uint8:
            raise ValueError("cameras must be uint8 RGB")
        if state.shape != (8,) or state.dtype != np.float32:
            raise ValueError(f"state must be float32 (8,), got {state.dtype} {state.shape}")

        payload = json_numpy.dumps({
            "external_cam": external_cam,
            "wrist_cam":    wrist_cam,
            "instruction":  instruction,
            "state":        state,
            "num_steps":    int(num_steps),
            "enable_cuda_graph": bool(enable_cuda_graph),
        })
        t0 = time.perf_counter()
        r = self._session.post(
            f"{self.base_url}/act",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_s,
        )
        rtt = (time.perf_counter() - t0) * 1000.0
        if not r.ok:
            raise RuntimeError(
                f"server {r.status_code} on /act: {r.text[:500]} "
                f"(sent ext={external_cam.shape}, wrist={wrist_cam.shape}, "
                f"state={state.shape})"
            )
        body = json_numpy.loads(r.content)
        if "error" in body:
            raise RuntimeError(f"server error: {body['error']}")
        return DroidPrediction(
            actions=np.asarray(body["actions"], dtype=np.float32),
            server_dt_ms=float(body["dt_ms"]),
            rtt_ms=rtt,
            n_bytes_sent=len(payload),
        )
