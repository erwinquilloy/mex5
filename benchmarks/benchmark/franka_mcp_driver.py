"""MCP-transport driver: talks to franka/python/mcp_server.py via fastmcp.

The MCP server wraps motion_server's REST endpoint as a single MCP tool
(`send_command_tool`). Mechanically it adds one hop on top of REST, so this
driver subclasses FrankaRestDriver and only overrides the transport call.
Everything else (FK, Euler decomposition, gripper sequencing, send_chunk
loop) is inherited.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional, Sequence

from .franka_rest_driver import FrankaRestDriver


def _unwrap(result: Any) -> dict:
    """Pull a dict out of whatever fastmcp's call_tool returned.

    fastmcp 2.x returns a CallToolResult with .data (structured) and
    .content (list of content blocks). Older versions only populate .content
    with a TextContent whose .text is the JSON-serialized return value.
    """
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        block = content[0]
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    if isinstance(result, dict):
        return result
    return {}


class FrankaMcpDriver(FrankaRestDriver):
    def __init__(
        self,
        mcp_url: Optional[str] = None,
        step_time_s: float = 2.5,
        timeout_s: float = 60.0,
    ):
        url = mcp_url or os.environ.get("FRANKA_MCP_URL")
        if not url:
            raise RuntimeError(
                "FrankaMcpDriver needs the MCP server URL. "
                "Pass --mcp-url or set FRANKA_MCP_URL. "
                "Default per franka/python/mcp_server.py: http://<host>:8085/franka"
            )
        try:
            from fastmcp import Client  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "fastmcp not installed. pip install fastmcp"
            ) from e
        try:
            import panda_py  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "FrankaMcpDriver needs panda_py for local FK. "
                "Install panda-py on this machine (kinematics only, no FCI)."
            ) from e

        # Bypass FrankaRestDriver.__init__'s host plumbing — we don't
        # POST to motion_server directly.
        self._mcp_url = url
        self._timeout_s = float(timeout_s)
        self._step_time_s = float(step_time_s)
        self._last_grip = None

    def _post(self, command: str, params: Sequence[float]) -> dict:
        from fastmcp import Client

        payload = {"command": command, "parameters": [float(x) for x in params]}

        async def _call() -> Any:
            async with Client(self._mcp_url) as client:
                return await client.call_tool("send_command_tool", payload)

        result = asyncio.run(_call())
        return _unwrap(result)
