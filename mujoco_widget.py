from pathlib import Path
import json
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
    command = traitlets.Unicode("").tag(sync=True)
    sim_time = traitlets.Float(0).tag(sync=True)

    def __init__(self, xml, asset_base, **kwargs):
        self._seq = 0
        super().__init__(asset_base=asset_base, xml=xml, **kwargs)

    def _send(self, op, **kwargs):
        self._seq += 1
        self.command = json.dumps(dict(seq=self._seq, op=op, **kwargs))

    def pause(self): self._send("pause")
    def play(self): self._send("play")
    def reset(self): self._send("reset")
    def set_gravity(self, value): self._send("gravity", value=float(value))
    def set_control(self, index, value): self._send("control", index=int(index), value=float(value))
