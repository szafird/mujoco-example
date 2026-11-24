"""MuJoCo browser widget with a single nonblocking API for Jupyter, Colab and VS Code.

Physics and rendering run in the browser. Python sends commands and receives
asynchronous event replies *after the calling notebook cell yields*. In
particular, no Python API method blocks waiting for a widget response.
"""
from pathlib import Path
import itertools
import traceback
import anywidget
import traitlets

ROOT = Path(__file__).resolve().parent


class _BaseWidget(anywidget.AnyWidget):
    asset_base = traitlets.Unicode("").tag(sync=True)
    status = traitlets.Unicode("Waiting for browser...").tag(sync=True)

    def __init__(self, asset_base, **kwargs):
        if not isinstance(asset_base, str) or not asset_base.startswith("https://"):
            raise ValueError("asset_base must be an HTTPS URL to the hosted mujoco/ directory")
        super().__init__(asset_base=asset_base, **kwargs)


class MuJoCoSmokeTest(_BaseWidget):
    _esm = ROOT / "dist" / "smoke.js"


class ControllerRun:
    """Mutable handle for a browser-stepped, Python-defined controller.

    Check ``run.done``, ``run.completed``, ``run.state`` and ``run.error`` from
    a later cell. The controller executes from widget callbacks, not a blocking
    notebook cell. The viewer retains the handle until stopped/completed.
    """
    def __init__(self, viewer, controller, steps, substeps, on_done=None):
        self.viewer = viewer
        self.controller = controller
        self.steps = steps
        self.substeps = substeps
        self.on_done = on_done
        self.completed = 0
        self.done = False
        self.cancelled = False
        self.error = None
        self.state = None

    def cancel(self):
        """Stop scheduling further controller steps; current chunk may finish."""
        if not self.done:
            self.cancelled = True
            self.done = True
            if self.viewer._controller is self:
                self.viewer._controller = None
            self.viewer._complete_controller(self)

    def __repr__(self):
        return (f"ControllerRun(completed={self.completed}/{self.steps}, "
                f"done={self.done}, cancelled={self.cancelled}, error={self.error!r})")


class MuJoCoViewer(_BaseWidget):
    _esm = ROOT / "dist" / "viewer.js"
    xml = traitlets.Unicode("").tag(sync=True)
    # Optional public HTTPS base directory for STL assets referenced by the MJCF.
    mesh_source_url = traitlets.Unicode("").tag(sync=True)
    sim_time = traitlets.Float(0.0).tag(sync=True)
    # Optional close-up camera for small tabletop robots; zero preserves V4 default.
    camera_distance = traitlets.Float(0.0).tag(sync=True)
    camera_target = traitlets.List(trait=traitlets.Float(), default_value=[0.0, 0.0, 0.5]).tag(sync=True)

    def __init__(self, xml, asset_base, **kwargs):
        self._requests = {}
        self._request_ids = itertools.count(1)
        self._state_listeners = []
        self._latest_sim_state = None
        self._controller = None
        self.last_error = None
        super().__init__(asset_base=asset_base, xml=xml, **kwargs)
        self.on_msg(self._receive_message)

    @property
    def latest_sim_state(self):
        """Last reply snapshot, or None until a response is processed."""
        return self._latest_sim_state

    def on_sim_state(self, callback, remove=False):
        """Register callback(state) for successful state/step responses."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        if remove:
            if callback in self._state_listeners:
                self._state_listeners.remove(callback)
        elif callback not in self._state_listeners:
            self._state_listeners.append(callback)

    def _send_command(self, op, **kwargs):
        self.send({"op": op, **kwargs})

    def _send_request(self, op, callback=None, **kwargs):
        """Send a request without waiting; callback(result, error) runs later."""
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        request_id = next(self._request_ids)
        self._requests[request_id] = callback
        try:
            self.send({"op": op, "id": request_id, **kwargs})
        except Exception:
            self._requests.pop(request_id, None)
            raise
        return request_id

    def _receive_message(self, widget, content, buffers):
        if content.get("type") != "reply":
            return
        request_id = content.get("id")
        if request_id not in self._requests:
            return  # e.g., stale response after a kernel restart
        callback = self._requests.pop(request_id)
        error = content.get("error")
        result = content.get("result")
        if error:
            self.last_error = str(error)
        elif isinstance(result, dict):
            self._latest_sim_state = result
            for listener in tuple(self._state_listeners):
                try:
                    listener(result)
                except Exception:
                    traceback.print_exc()
        if callback is not None:
            try:
                callback(result, str(error) if error else None)
            except Exception:
                traceback.print_exc()

    def play(self):
        """Play real-time physics in the browser."""
        if self._controller and not self._controller.done:
            raise RuntimeError("Cancel the active Python controller before playing")
        self._send_command("play")

    def pause(self):
        self._send_command("pause")

    def reset(self):
        if self._controller and not self._controller.done:
            self._controller.cancel()
        self._send_command("reset")

    def set_gravity(self, value):
        self._send_command("gravity", value=float(value))

    def set_control(self, index, value):
        self._send_command("control", index=int(index), value=float(value))

    def request_sim_state(self, callback=None):
        """Request state; returns request ID immediately.

        Read latest_sim_state in a later notebook cell, or use on_sim_state().
        Optional callback signature: callback(state, error).
        """
        return self._send_request("get_state", callback=callback)

    def step(self, n=1, callback=None):
        """Advance n physics steps while paused; returns request ID.

        The resulting state is available later through latest_sim_state.
        """
        if not isinstance(n, int) or not (1 <= n <= 10000):
            raise ValueError("n must be an integer in 1..10000")
        if self._controller and not self._controller.done:
            raise RuntimeError("Cancel the active Python controller before manual stepping")
        return self._send_request("step", callback=callback, n=n)

    def run_controller(self, controller, steps=200, substeps=5, on_done=None):
        """Run a Python feedback controller through nonblocking widget callbacks.

        ``controller(state)`` returns a scalar (single actuator), a sequence of
        actuator values, or None (keep existing controls). The browser pauses
        real-time playback and steps exactly ``steps * substeps`` physics steps.
        Each controller invocation is based on a returned MuJoCo state. Cells
        must *not* block waiting for this handle; check it from later cells.

        This callback round-trip is designed for demonstrations and teaching,
        not hard real-time or high-frequency control.
        """
        if not callable(controller):
            raise TypeError("controller must be callable")
        if not isinstance(steps, int) or not (1 <= steps <= 100000):
            raise ValueError("steps must be 1..100000")
        if not isinstance(substeps, int) or not (1 <= substeps <= 10000):
            raise ValueError("substeps must be 1..10000")
        if self._controller and not self._controller.done:
            raise RuntimeError("A controller is already running. Cancel it first")
        run = ControllerRun(self, controller, steps, substeps, on_done)
        self._controller = run

        def fail(exc):
            run.error = str(exc)
            run.done = True
            if self._controller is run:
                self._controller = None
            self._complete_controller(run)

        def tick(state, error):
            if run.done:
                return
            if error:
                fail(error)
                return
            run.state = state
            if run.completed >= run.steps:
                run.done = True
                if self._controller is run:
                    self._controller = None
                self._complete_controller(run)
                return
            try:
                control = run.controller(state)
                if control is None:
                    ctrl_values = None
                elif isinstance(control, (int, float)):
                    ctrl_values = [float(control)]
                else:
                    ctrl_values = [float(value) for value in control]
                run.completed += 1
                self._send_request("controller_step", callback=tick,
                                   n=run.substeps, controls=ctrl_values)
            except Exception as exc:
                fail(exc)

        # The browser pauses physics before taking the initial snapshot.
        self._send_request("controller_start", callback=tick)
        return run

    @staticmethod
    def _complete_controller(run):
        if run.on_done is not None:
            try:
                run.on_done(run)
            except Exception:
                traceback.print_exc()

