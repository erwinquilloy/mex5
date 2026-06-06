"""WebRTC live-video broadcaster used by the dashboard.

Browser flow:
  1. browser POSTs {sdp, type} to /offer/<cam>
  2. server attaches a VideoStreamTrack that pulls the latest RGB frame
     from a caller-supplied getter, returns {sdp, type} (answer)
  3. browser sets remote desc; <video> starts playing

Single asyncio loop runs in a background thread; Flask handlers hand
offers in via run_coroutine_threadsafe and block for the answer.

Requires: pip install aiortc av
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("dashboard.webrtc")


FrameGetter = Callable[[], Optional[np.ndarray]]


def _import_aiortc():
    """Defer the aiortc/av import so the module is loadable on hosts that
    haven't installed them yet (the dashboard still works via MJPEG)."""
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        import av
        return RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, av
    except ImportError as e:
        raise RuntimeError(
            "WebRTC support needs `pip install aiortc av`. Import error: " + str(e)
        ) from e


def _make_track(frame_getter: FrameGetter):
    _, _, VideoStreamTrack, av = _import_aiortc()

    class _LatestFrameTrack(VideoStreamTrack):
        kind = "video"

        async def recv(self):
            # next_timestamp paces at aiortc's default 30 fps and supplies the
            # pts/time_base aiortc's RTP packetizer expects.
            pts, time_base = await self.next_timestamp()
            rgb = frame_getter()
            if rgb is None:
                rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            vf.pts = pts
            vf.time_base = time_base
            return vf

    return _LatestFrameTrack()


class WebRTCBroadcaster:
    """One per dashboard process. Each handle_offer() spawns its own
    RTCPeerConnection wired to the requested camera's frame getter."""

    def __init__(self) -> None:
        # Validate aiortc up front so the dashboard can fall back to MJPEG
        # at startup rather than failing on the first /offer.
        self._RTCPeerConnection, self._RTCSessionDescription, _, _ = _import_aiortc()
        self._loop = asyncio.new_event_loop()
        self._pcs: set = set()
        self._thread = threading.Thread(
            target=self._run_loop, name="webrtc-loop", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _accept_offer(
        self, sdp: str, type_: str, frame_getter: FrameGetter
    ) -> dict:
        pc = self._RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self._pcs.discard(pc)

        pc.addTrack(_make_track(frame_getter))
        await pc.setRemoteDescription(
            self._RTCSessionDescription(sdp=sdp, type=type_)
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    def handle_offer(self, sdp: str, type_: str, frame_getter: FrameGetter,
                     timeout_s: float = 10.0) -> dict:
        fut = asyncio.run_coroutine_threadsafe(
            self._accept_offer(sdp, type_, frame_getter), self._loop
        )
        return fut.result(timeout=timeout_s)

    def close(self) -> None:
        async def _close_all() -> None:
            await asyncio.gather(*(pc.close() for pc in list(self._pcs)),
                                 return_exceptions=True)
        try:
            asyncio.run_coroutine_threadsafe(_close_all(), self._loop).result(timeout=3.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
