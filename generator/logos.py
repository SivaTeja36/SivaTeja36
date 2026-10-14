"""Trace logo SVGs (Simple Icons paths) into N evenly spread sample points.

Each logo becomes out/logo_<name>.npy, shape (N, 2), in portrait coordinates
(the 300x340 dot grid), centred at LOGO_CENTRE and scaled to LOGO_SIZE.

Usage: python logos.py logos/python.svg logos/fastapi.svg logos/terraform.svg
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from svgpathtools import parse_path

N_DOTS = 900
RASTER = 480
LOGO_SIZE = 210.0           # width/height of the logo bounding square in portrait units
LOGO_CENTRE = (150.0, 170.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def raster_mask(svg_file: str) -> np.ndarray:
    root = ET.parse(svg_file).getroot()
    vb = root.get("viewBox", "0 0 24 24").split()
    vx, vy, vw, vh = map(float, vb)
    acc = np.zeros((RASTER, RASTER), bool)
    sc = RASTER / max(vw, vh)
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        path = parse_path(el.get("d"))
        for sub in path.continuous_subpaths():
            pts = []
            for seg in sub:
                n = max(2, int(seg.length() * sc / 1.0))
                for t in np.linspace(0, 1, n, endpoint=False):
                    z = seg.point(t)
                    pts.append(((z.real - vx) * sc, (z.imag - vy) * sc))
            if len(pts) < 3:
                continue
            im = Image.new("L", (RASTER, RASTER), 0)
            ImageDraw.Draw(im).polygon(pts, fill=255)
            acc ^= np.asarray(im) > 127   # even-odd: inner subpaths become holes
    return acc


def spread_points(mask: np.ndarray, n: int, seed: int = 7, iters: int = 25) -> np.ndarray:
    """Pick n points from the mask, then Lloyd-relax them so spacing is even."""
    ys, xs = np.nonzero(mask)
    pix = np.stack([xs, ys], 1).astype(float)
    rng = np.random.default_rng(seed)
    pts = pix[rng.choice(len(pix), n, replace=False)]
    for _ in range(iters):
        tree = cKDTree(pts)
        _, owner = tree.query(pix)
        new = pts.copy()
        for i in range(n):
            sel = pix[owner == i]
            if len(sel):
                new[i] = sel.mean(0)
        pts = new
    return pts


def to_portrait(pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ext = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = LOGO_SIZE / ext
    out = (pts - [cx, cy]) * s + LOGO_CENTRE
    return out


def main(files):
    os.makedirs(OUT, exist_ok=True)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        m = raster_mask(f)
        pts = spread_points(m, N_DOTS)
        pp = to_portrait(pts, m)
        np.save(os.path.join(OUT, f"logo_{name}.npy"), pp)
        pv = Image.new("L", (300, 340), 0)
        d = ImageDraw.Draw(pv)
        for x, y in pp:
            d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=255)
        pv.resize((600, 680), Image.NEAREST).save(os.path.join(OUT, f"preview_logo_{name}.png"))
        print(f"{name}: mask fill {m.mean():.3f}, {len(pp)} pts, "
              f"bbox x[{pp[:,0].min():.0f},{pp[:,0].max():.0f}] y[{pp[:,1].min():.0f},{pp[:,1].max():.0f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
