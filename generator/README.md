# Banner generator

Source of truth for `dark.svg` / `light.svg`. Edit the scripts, regenerate, commit the SVGs.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python portrait.py inputs/photo.jpg --crop 160 0 619 520   # -> out/portrait_*.npy
python logos.py logos/python.svg logos/fastapi.svg logos/terraform.svg
python banner.py                                           # -> ../dark.svg ../light.svg
python stack.py                                            # -> ../stack.svg  (tech stack panel)
python connect.py                                          # -> ../connect.svg (contact panel)
```

- `inputs/photo.jpg` is git-ignored. Drop a new head-and-shoulders photo there and re-run.
- `portrait.py`: crop, MediaPipe person segmentation (GrabCut-refined edge), autocontrast + 1.3x contrast + unsharp mask,
  serpentine Floyd–Steinberg on a 300x340 grid. `DARK_GAIN`/`DARK_GAMMA` compress highlights.
- `logos.py`: traces Simple Icons paths into 900 evenly spread points per logo.
- `banner.py`: assembles the SVG (intro shimmer, drift bands, optimal-transport travellers,
  locked info panel) and prints the evenness / straight-boundary metrics.
- Text rows live in `ROWS` in `banner.py`. Palette in `THEMES`.
- `panels.py`: shared frame, embedded fonts (assets/fonts, OFL) and brand icons (assets/icons, Simple Icons) for
  `stack.py` (orbiting icons + grouped chips, edit `GROUPS`) and `connect.py` (portrait + contact `CARDS`).
