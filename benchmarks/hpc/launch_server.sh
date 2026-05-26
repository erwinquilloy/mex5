#!/usr/bin/env bash
# Launch MolmoAct2 server on the A100 HPC node and open a local tunnel.
#
# One-time (on HPC):
#   conda create -n molmoact2 python=3.10 -y
#   conda activate molmoact2
#   pip install -r benchmarks/hpc/requirements_hpc.txt
#
# Per-session:
#   # terminal A on HPC (or via srun if slurm-managed)
#   conda activate molmoact2
#   python benchmarks/hpc/serve_molmoact2.py --port 8000
#
#   # terminal B on workstation
#   ssh -N -L 8000:localhost:8000 erwin.quilloy@ai-n002.hpc.coe.upd.edu.ph
#
# Verify:
#   curl http://localhost:8000/health
set -euo pipefail

HPC_USER="${HPC_USER:-erwin.quilloy}"
HPC_HOST="${HPC_HOST:-ai-n002.hpc.coe.upd.edu.ph}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

case "${1:-tunnel}" in
  tunnel)
    exec ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${HPC_USER}@${HPC_HOST}"
    ;;
  remote-serve)
    ssh -t "${HPC_USER}@${HPC_HOST}" \
      "cd ~/molmoact2-bench && conda run -n molmoact2 python benchmarks/hpc/serve_molmoact2.py --port ${REMOTE_PORT}"
    ;;
  *)
    echo "usage: $0 {tunnel|remote-serve}" >&2
    exit 2
    ;;
esac
