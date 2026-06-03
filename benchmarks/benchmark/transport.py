"""Transport autodetect + driver factory.

Three robot transports share a duck-typed surface (home, state_vec8,
send_chunk, close). This module picks one from env vars and constructs it.
The dashboard uses autodetect; droid_runner uses the explicit form so its
CLI flags still rule.
"""
from __future__ import annotations

import os
from typing import Optional


_PRECEDENCE = ("mcp", "rest", "fci")


def autodetect_transport() -> str:
    """Return the transport implied by env vars.

    Precedence: FRANKA_MCP_URL > FRANKA_REST_HOST > FRANKA_HOST.
    The more specific names beat the general FRANKA_HOST so a leftover
    FCI env from a previous session doesn't override a deliberate
    FRANKA_REST_HOST.
    """
    if os.environ.get("FRANKA_MCP_URL"):
        return "mcp"
    if os.environ.get("FRANKA_REST_HOST"):
        return "rest"
    if os.environ.get("FRANKA_HOST"):
        return "fci"
    raise RuntimeError(
        "No transport env var set. Set one of: "
        "FRANKA_MCP_URL=http://<host>:8085/franka (mcp), "
        "FRANKA_REST_HOST=<motion_server IP> (rest), or "
        "FRANKA_HOST=<robot FCI IP> (fci)."
    )


def make_driver(
    transport: str,
    *,
    fci_host: Optional[str] = None,
    rest_host: Optional[str] = None,
    rest_port: int = 34568,
    mcp_url: Optional[str] = None,
    step_time_s: float = 2.5,
):
    """Build a driver. Imports are local so missing optional deps don't kill
    the whole module (e.g. fastmcp not installed and the user picked fci)."""
    if transport == "fci":
        from .panda_driver import PandaDriver
        return PandaDriver(hostname=fci_host)
    if transport == "rest":
        from .franka_rest_driver import FrankaRestDriver
        return FrankaRestDriver(host=rest_host, port=rest_port, step_time_s=step_time_s)
    if transport == "mcp":
        from .franka_mcp_driver import FrankaMcpDriver
        return FrankaMcpDriver(mcp_url=mcp_url, step_time_s=step_time_s)
    raise ValueError(f"unknown transport: {transport!r} (expected 'fci', 'rest', or 'mcp')")
