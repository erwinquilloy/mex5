"""HTTP client to the MolmoAct2 FastAPI server (tunneled over SSH)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import requests
from PIL import Image

from .camera import image_to_b64


@dataclass
class Prediction:
    actions: list[list[float]]   # [n, 7]
    server_dt_ms: float          # GPU + processor time as reported by the server
    rtt_ms: float                # full HTTP round-trip from the client


class MolmoActClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def health(self) -> dict:
        return self._session.get(f"{self.base_url}/health", timeout=5).json()

    def reset(self) -> None:
        self._session.post(f"{self.base_url}/reset", timeout=5)

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[Sequence[float]] = None,
        n_actions: int = 1,
    ) -> Prediction:
        payload = {
            "image_b64": image_to_b64(image),
            "instruction": instruction,
            "n_actions": n_actions,
        }
        if state is not None:
            payload["state"] = list(state)

        t0 = time.perf_counter()
        r = self._session.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout_s)
        rtt = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        body = r.json()
        return Prediction(actions=body["actions"], server_dt_ms=body["dt_ms"], rtt_ms=rtt)
