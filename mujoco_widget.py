"""MuJoCo browser widget: Python commands + asynchronous request/reply."""
from pathlib import Path
import asyncio
import itertools
import threading
import anywidget
import traitlets

ROOT = Path(__file__).resolve().parent

class _BaseWidget(anywidget.AnyWidget):
    asset_base = traitlets.Unicode("").tag(sync=True)
    status = traitlets.Unicode("Waiting for browser...").tag(sync=True)

    def __init__(self, asset_base, **kwargs):
        if not asset_base.startswith("https://"):
            raise ValueError("asset_base must be the public HTTPS URL of your docs/mujoco/ directory")
        super().__init__(asset_base=asset_base, **kwargs)

class MuJoCoSmokeTest(_BaseWidget):
    _esm = ROOT / "dist" / "smoke.js"

class MuJoCoViewer(_BaseWidget):
    _esm = ROOT / "dist" / "viewer.js"
    xml = traitlets.Unicode("").tag(sync=True)
    sim_time = traitlets.Float(0).tag(sync=True)

    def __init__(self, xml, asset_base, **kwargs):
        self._pending = {}
        self._request_ids = itertools.count(1)
        super().__init__(asset_base=asset_base, xml=xml, **kwargs)
        self.on_msg(self._receive_message)

    def _receive_message(self, widget, content, buffers):
        
        print("[MuJoCo] Received from JavaScript:", content)

        if content.get("type") != "reply":
            return

        request_id = content.get("id")

        print("Received request ID:", request_id)
        print("Pending request IDs:", list(self._pending.keys()))
        print("Request ID type:", type(request_id))


        pending = self._pending.get(request_id)

        print("Found pending request:", pending is not None)

        if pending is None:
            print(f"[MuJoCo] No pending request for id={request_id}")
            return

        loop, future = pending

        import threading
        print("Reply thread:", threading.get_ident())
        print("Reply loop:", id(loop))
        print("Future done:", future.done())

        def resolve():
            print("[MuJoCo] resolve() executing")
            print("[MuJoCo] Future already done:", future.done())

            if future.done():
                return

            if content.get("error"):
                future.set_exception(
                    RuntimeError(str(content["error"]))
                )
            else:
                future.set_result(content.get("result"))

            print("[MuJoCo] Future resolved:", future.done())

        # Schedule completion on the event loop that owns the future.
        loop.call_soon_threadsafe(resolve)


    def _send_command(self, op, **kwargs):
        """Fire-and-forget command; frontend must be initialized."""
        self.send({"op": op, **kwargs})

    async def _request(self, op, timeout=10.0, **kwargs):
        loop = asyncio.get_running_loop()
        import threading
        print("Request thread:", threading.get_ident())
        print("Request loop:", id(loop))
        request_id = next(self._request_ids)

        future = loop.create_future()
        self._pending[request_id] = (loop, future)

        try:
            self.send({
                "op": op,
                "id": request_id,
                **kwargs
            })

            return await asyncio.wait_for(future, timeout=timeout)

        finally:
            self._pending.pop(request_id, None)

    def play(self): self._send_command("play")
    def pause(self): self._send_command("pause")
    def reset(self): self._send_command("reset")
    def set_gravity(self, value): self._send_command("gravity", value=float(value))
    def set_control(self, index, value):
        self._send_command("control", index=int(index), value=float(value))

    async def get_sim_state(self):
        return await self._request("get_state")

    async def step(self, n=1, timeout=10.0):
        if not isinstance(n, int) or not 1 <= n <= 10000:
            raise ValueError("n must be an integer between 1 and 10000")
        return await self._request("step", n=n, timeout=timeout)
