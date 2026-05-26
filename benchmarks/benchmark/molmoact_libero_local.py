"""In-process MolmoAct2-LIBERO inference.

LIBERO sim and the model both live on the HPC, so we skip the HTTP layer
entirely. This module just wraps `model.predict_action(...)` from the
upstream MolmoAct2 transformers code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LiberoPrediction:
    actions: np.ndarray         # (N, 7) float32, LIBERO-scale; pass to env.step
    server_dt_ms: float         # forward-pass wallclock
    rtt_ms: float               # equal to server_dt_ms here (in-process)


class MolmoActLiberoLocal:
    """Loads `allenai/MolmoAct2-LIBERO` once and exposes a thin act() call."""

    def __init__(
        self,
        model_id: str = "allenai/MolmoAct2-LIBERO",
        dtype: str = "bf16",
        device: str = "cuda",
        enable_cuda_graph: bool = True,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self._torch = torch
        self.model_id = model_id
        self._enable_cuda_graph = enable_cuda_graph

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch_dtype,
        ).to(device).eval()
        self._device = next(self._model.parameters()).device

    def health(self) -> dict:
        return {
            "ok": True,
            "model_id": self.model_id,
            "device": str(self._device),
        }

    def reset(self) -> None:
        # Stateless inference; left as a hook for future temporal buffers.
        pass

    def act(
        self,
        agentview: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        state8: np.ndarray,
        num_steps: int = 10,
    ) -> LiberoPrediction:
        if agentview.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("camera frames must be uint8 RGB")
        if state8.shape != (8,):
            raise ValueError(f"state8 must be shape (8,), got {state8.shape}")

        t0 = time.perf_counter()
        with self._torch.inference_mode():
            out = self._model.predict_action(
                processor=self._processor,
                images=[agentview, wrist],
                task=instruction,
                state=state8.astype(np.float32),
                norm_tag="libero",
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=self._enable_cuda_graph,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(out.actions, dtype=np.float32).reshape(-1, 7)
        return LiberoPrediction(actions=actions, server_dt_ms=dt_ms, rtt_ms=dt_ms)
