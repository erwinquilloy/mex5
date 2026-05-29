"""Single-shot diagnostic for the MolmoAct2-LIBERO wiring.

Renders one LIBERO task, runs one inference, dumps the camera frames + state
+ action chunk so we can sanity-check image orientation, state convention,
and action magnitudes.

Run:
    CUDA_VISIBLE_DEVICES=3 python -m benchmarks.scripts.debug_libero_one_shot
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from benchmarks.benchmark.libero_env import LiberoEnv
from benchmarks.benchmark.molmoact_libero_local import MolmoActLiberoLocal


def main() -> None:
    env = LiberoEnv("libero_spatial", 0)
    obs = env.reset(init_index=0)

    Image.fromarray(obs.agentview).save("/tmp/agent.png")
    Image.fromarray(obs.wrist).save("/tmp/wrist.png")
    print("saved /tmp/agent.png and /tmp/wrist.png")
    print("agentview shape/dtype:", obs.agentview.shape, obs.agentview.dtype)
    print("wrist shape/dtype:    ", obs.wrist.shape, obs.wrist.dtype)
    print("state8:", obs.state8)
    print("instruction:", env.spec.instruction)

    molmo = MolmoActLiberoLocal(enable_cuda_graph=False)
    pred = molmo.act(
        agentview=obs.agentview,
        wrist=obs.wrist,
        instruction=env.spec.instruction,
        state8=obs.state8,
        num_steps=32,
    )
    a = np.asarray(pred.actions)
    print("action chunk shape:", a.shape)
    print("first action:      ", a[0])
    print("action min:        ", a.min(axis=0))
    print("action max:        ", a.max(axis=0))
    print("action abs-mean:   ", np.abs(a).mean(axis=0))
    print("server_dt_ms:      ", pred.server_dt_ms)

    env.close()


if __name__ == "__main__":
    main()
