"""
MolmoAct2 inference server (runs on the A100 HPC node).

Exposes:
  GET  /health
  POST /predict   {image_b64, instruction, state?, n_actions?} -> {actions, dt_ms}
  POST /reset                                                   -> {ok}

Launch (on ai-n002):
    python serve_molmoact2.py --host 0.0.0.0 --port 8000

Then from the workstation:
    ssh -N -L 8000:localhost:8000 erwin.quilloy@ai-n002.hpc.coe.upd.edu.ph
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor

log = logging.getLogger("molmoact2.serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class PredictRequest(BaseModel):
    image_b64: str
    instruction: str
    state: Optional[list[float]] = None  # 7-DoF proprio if the model uses it
    n_actions: int = 1                    # action chunk length to return


class PredictResponse(BaseModel):
    actions: list[list[float]]            # [n_actions, 7] -> [dx,dy,dz,droll,dpitch,dyaw,grip]
    dt_ms: float


_state = {"model": None, "processor": None, "device": None, "model_id": None}


def _load_model(model_id: str, dtype: str):
    log.info("loading %s (dtype=%s)", model_id, dtype)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    _state["model"] = model
    _state["processor"] = processor
    _state["device"] = next(model.parameters()).device
    _state["model_id"] = model_id
    log.info("loaded on %s", _state["device"])


app = FastAPI(title="MolmoAct2")


@app.get("/health")
def health():
    return {
        "ok": _state["model"] is not None,
        "model_id": _state["model_id"],
        "device": str(_state["device"]),
        "cuda": torch.cuda.is_available(),
    }


@app.post("/reset")
def reset():
    # Stateless inference for now. If the model carries a temporal buffer
    # (e.g. action-chunk consumer), reset it here per-episode.
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(503, "model not loaded")

    try:
        img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"bad image_b64: {e}")

    proc = _state["processor"]
    model = _state["model"]
    device = _state["device"]

    t0 = time.perf_counter()
    inputs = proc(images=[img], text=req.instruction, return_tensors="pt").to(device)
    if req.state is not None and hasattr(proc, "encode_state"):
        inputs["state"] = torch.tensor([req.state], device=device)

    with torch.inference_mode():
        # MolmoAct exposes generate_actions on its custom HF class; if not present,
        # fall back to generate() + processor.decode_actions().
        if hasattr(model, "generate_actions"):
            actions = model.generate_actions(**inputs, n_actions=req.n_actions)
        else:
            out = model.generate(**inputs, max_new_tokens=req.n_actions * 8)
            actions = proc.decode_actions(out, n_actions=req.n_actions)
        actions = actions.detach().float().cpu().tolist()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return PredictResponse(actions=actions, dt_ms=dt_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-id", default=os.environ.get("MOLMOACT2_MODEL", "allenai/MolmoAct-2-7B"))
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    _load_model(args.model_id, args.dtype)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
