from pathlib import Path
import json
import anywidget
import traitlets

class MuJoCoViewer(anywidget.AnyWidget):
    """Browser-side MuJoCo WASM physics and Three.js visualization."""
    _esm = Path(__file__).with_name('viewer.js')
    xml = traitlets.Unicode('').tag(sync=True)
    command = traitlets.Unicode('').tag(sync=True)
    status = traitlets.Unicode('Loading…').tag(sync=True)
    sim_time = traitlets.Float(0).tag(sync=True)
    def __init__(self, xml, **kwargs):
        self._seq = 0
        super().__init__(xml=xml, **kwargs)
    def _send(self, op, **kwargs):
        self._seq += 1
        self.command = json.dumps(dict(seq=self._seq, op=op, **kwargs))
    def pause(self): self._send('pause')
    def play(self): self._send('play')
    def reset(self): self._send('reset')
    def set_gravity(self, gravity): self._send('gravity', value=float(gravity))
    def set_control(self, index, value): self._send('control', index=int(index), value=float(value))
