# MuJoCo Intro — Binder and Colab

A minimal Jupyter notebook demonstrating MJCF, physics stepping, plots, and inline offscreen rendering.

## Binder
Open in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/szafird/mujoco-example/HEAD?labpath=mujoco_intro.ipynb)   

Binder installs `apt.txt` and `requirements.txt` automatically. The notebook uses OSMesa software rendering.

## Google Colab
Upload `mujoco_intro.ipynb` to Colab, then run cells in order. The first code cell installs dependencies if missing. **Colab Linux may not have OSMesa installed**. If rendering fails, the notebook still plots the physics trajectory. To enable rendering on Colab, run this cell **before importing MuJoCo**, then restart the runtime if MuJoCo was already imported:

```python
!apt-get -qq update && apt-get -qq install -y libosmesa6 libgl1-mesa-dri
```

Then rerun from the top. Colab runtimes vary; software rendering availability is environment-dependent.
