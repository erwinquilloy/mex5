# HPC — MolmoAct2-DROID inference server

We use **Allen AI's upstream FastAPI server** (`examples/droid/host_server_droid.py`
from [`allenai/molmoact2`](https://github.com/allenai/molmoact2)) rather than a custom
one. The schema is fixed by the model card: two cameras + `state[8]` in, `actions[N,8]`
out (absolute joint targets + gripper).

## One-time setup on `ai-n002.hpc.coe.upd.edu.ph`

```bash
# Use uv (much faster + matches the upstream lockfile)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL    # pick up PATH

git clone https://github.com/allenai/molmoact2.git ~/molmoact2
cd ~/molmoact2
uv sync                                   # creates .venv, installs torch/transformers/etc.
uv run hf download allenai/MolmoAct2-DROID
```

If `~` is quota-limited, move HF cache off home **before** the download:

```bash
mkdir -p $HOME/hf-cache
export HF_HOME=$HOME/hf-cache              # add to ~/.bashrc to persist
```

## Required upstream patch — `action_mode` kwarg rename

Upstream `examples/droid/host_server_droid.py` passes `action_mode="continuous"`
to `predict_action(...)`, but the current MolmoAct2 model code on HF expects
`inference_action_mode`. Without the rename the server still starts and serves
`GET /act` 200, but **every real `/act` call returns 500** and the startup log
contains `TypeError: predict_action() got an unexpected keyword argument 'action_mode'`
from the warmup. Re-apply this patch on every fresh clone of `~/molmoact2`:

```bash
sed -i 's/action_mode="continuous"/inference_action_mode="continuous"/' \
  ~/molmoact2/examples/droid/host_server_droid.py

# verify
grep -n 'inference_action_mode="continuous"' \
  ~/molmoact2/examples/droid/host_server_droid.py    # expect: one match
```

After relaunch, look for `Warmup OK (XXXX ms)` in the server log to confirm.

## Per-session: launch the server

```bash
cd ~/molmoact2
# pick a free port (port 8000 is often taken on shared HPC nodes; this picks one):
PORT=$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "MolmoAct2-DROID server port: $PORT"
uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port $PORT --dtype bfloat16
# wait for: "Uvicorn running on http://0.0.0.0:<PORT>"
```

Verify (in another shell on the HPC):
```bash
curl http://localhost:$PORT/act    # GET /act returns the health blob with norm_tag etc.
```

`GET /act` 200 alone is **not** enough — re-read the server log and confirm
`Warmup OK (...)` is present. If you only see `Listening on 0.0.0.0:<PORT>`
without a warmup-OK line, the upstream `action_mode` patch above is missing.

## Workstation: SSH tunnel

```bash
# replace PORT with the port the server printed
ssh -N -L 8000:localhost:PORT erwin.quilloy@ai-n002.hpc.coe.upd.edu.ph
# now http://localhost:8000 on the workstation reaches the HPC server
```

## Notes

- **Memory**: `bfloat16` keeps the model under 16GB and is the recommended default.
  Use `--dtype float32` only if you're chasing reproducibility of the paper numbers
  exactly (≈26GB, A100 80GB only).
- **CUDA visibility**: if you're on a Slurm node, `srun --gres=gpu:a100:1 --pty bash`
  before launching, and confirm with `nvidia-smi`.
- **Two conda envs**: the molmoact2-upstream env has its own torch/transformers
  pinned via `uv`. Don't `pip install` into it from elsewhere — that's how we hit
  the `cached_path` URL bug earlier.
