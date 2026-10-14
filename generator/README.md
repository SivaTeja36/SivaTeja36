# Banner generator

Source of truth for `dark.svg` / `light.svg`. Edit the scripts, regenerate, commit the SVGs.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python portrait.py inputs/photo.jpg --crop 160 0 619 520   # -> out/portrait_*.npy
python logos.py logos/python.svg logos/fastapi.svg logos/terraform.svg
python banner.py                                           # -> ../dark.svg ../light.svg
```

- `inputs/photo.jpg` is git-ignored. Drop a new head-and-shoulders photo there and re-run.
- `portrait.py`: crop, GrabCut background removal, autocontrast + 1.3x contrast + unsharp mask,
  serpentine Floyd–Steinberg on a 300x340 grid. `DARK_GAIN`/`DARK_GAMMA` compress highlights.
- `logos.py`: traces Simple Icons paths into 900 evenly spread points per logo.
- `banner.py`: assembles the SVG (intro shimmer, drift bands, optimal-transport travellers,
  locked info panel) and prints the evenness / straight-boundary metrics.
- Text rows live in `ROWS` in `banner.py`. Palette in `THEMES`.
