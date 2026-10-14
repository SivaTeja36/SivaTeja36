"""Assemble dark.svg / light.svg from the portrait + logo .npy data.

Layout: one terminal window 1180x610 titled `profile.sh --live`.
  left  ~38%  VISUAL.MAP  portrait frame (dither dots, 300x340 grid at 1.4x)
  right       SYSTEM.INFO readout, dotted leaders, pulsing LIVE badge, handle pill

Animation:
  intro  (~3.2s, once)  60 interleaved random groups fade in over ~2s
  loop   (14.2s)        portrait 3.0 | ->1.3 | logo1 2.0 | ->1.3 | logo2 2.0 | ->1.3 | logo3 2.0 | ->1.3
    layer A: full-density portrait, ~94 drift bands translate 42% toward logo1 centroid and fade
    layer B: 900 travellers morph logo1 -> logo2 -> logo3 (optimal-transport matched)

Usage: python banner.py            (writes ../dark.svg and ../light.svg)
"""
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.dirname(HERE)

PW, PH = 300, 340               # portrait grid
SCALE = 1.4
WIDTH, HEIGHT = 1180, 610
LOGOS = ["python", "react", "cloud", "ai"]

# ----- timing -------------------------------------------------------------
INTRO_DUR = 3.2
INTRO_GROUPS = 60
INTRO_SPREAD = 2.0
INTRO_FADE = 0.9
PORTRAIT_HOLD, LOGO_HOLD, TRANSITION = 3.0, 2.0, 1.3
TIMES = [0.0, PORTRAIT_HOLD]
for _ in LOGOS:                                   # -> formed, hold end, per logo
    TIMES += [TIMES[-1] + TRANSITION, TIMES[-1] + TRANSITION + LOGO_HOLD]
TIMES.append(TIMES[-1] + TRANSITION)              # return to portrait
LOOP = round(TIMES[-1], 2)
N_KEYS = len(TIMES)                               # 2N + 3
KEYTIMES = ";".join("1" if t == TIMES[-1] else f"{t / LOOP:.4f}" for t in TIMES)
EASE = ";".join(["0.42 0 0.58 1"] * (len(TIMES) - 1))

DRIFT = 0.42
N_BANDS = 94
NOISE_SIGMA = 4.0

THEMES = {
    "dark": dict(bg="#0A101F", chrome="#22D3EE", portrait="#A78BFA", accent="#10B981",
                 text="#E2E8F0", muted="#475569", title="#94A3B8", live="#EF4444",
                 frame_fill="#22D3EE", frame_fill_op="0.04", border_op="0.35"),
    "light": dict(bg="#F8FAFC", chrome="#0891B2", portrait="#7C3AED", accent="#059669",
                  text="#0F172A", muted="#94A3B8", title="#64748B", live="#DC2626",
                  frame_fill="#0891B2", frame_fill_op="0.05", border_op="0.45"),
}

ROWS = [
    ("Subject", "V. Siva Teja"),
    ("Role", "Full Stack · Azure Cloud · AI/ML Engineer"),
    ("Origin", "India"),
    ("Education", "B.Tech, Mechanical Engineering"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code · Git · Docker · Postman · Terraform"),
    None,
    ("Core.Lang", "Python · JavaScript · SQL"),
    ("Core.Frontend", "React.js · Streamlit"),
    ("Core.Backend", "FastAPI · Django · Flask · SQLAlchemy"),
    ("Core.Database", "PostgreSQL · PgVector · SQLite"),
    ("Core.Infra", "Azure · Terraform · Docker · Azure DevOps"),
    None,
    ("Grid.Mail", "vantasivateja@gmail.com"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.LinkedIn", "linkedin.com/in/siva-teja-vanta-99213b257"),
    ("Grid.GitHub", "github.com/SivaTeja36"),
]
HANDLE = "@SivaTeja36"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# info panel geometry
ROW_FS, HDR_FS, LIVE_FS, PILL_FS = 14, 13, 12, 14
ROW_GAP = 23
CW = ROW_FS * 0.6                 # locked char width
X0 = 500
COLS = 77
XR = X0 + COLS * CW
Y_HDR, Y_PILL, Y_ROW0 = 82, 98, 150

# portrait frame geometry
FX, FY, FW, FH = 24, 60, 448, 524
PX, PY = 38, 96


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ----- dots -> path runs ----------------------------------------------------
def runs_path(ys: np.ndarray, xs: np.ndarray) -> str:
    """Horizontal runs of unit dots as one path d-string."""
    if len(ys) == 0:
        return ""
    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    parts = []
    y0, x0, n = ys[0], xs[0], 1
    for y, x in zip(ys[1:], xs[1:]):
        if y == y0 and x == x0 + n:
            n += 1
        else:
            parts.append(f"M{x0} {y0}h{n}v1h-{n}z")
            y0, x0, n = y, x, 1
    parts.append(f"M{x0} {y0}h{n}v1h-{n}z")
    return "".join(parts)


# ----- grouping -------------------------------------------------------------
def intro_groups(n: int, rng) -> np.ndarray:
    return rng.integers(0, INTRO_GROUPS, size=n)


def evenness(groups: np.ndarray, ys, xs) -> float:
    """Mean total-variation distance between each group's spatial distribution
    and the whole portrait's, over a 5x5 cell grid. ~0.05 random, ~0.7 patchy."""
    cell = (ys * 5 // PH) * 5 + (xs * 5 // PW)
    tot = np.bincount(cell, minlength=25) / len(cell)
    tv = []
    for g in range(INTRO_GROUPS):
        sel = groups == g
        if sel.sum() == 0:
            continue
        p = np.bincount(cell[sel], minlength=25) / sel.sum()
        tv.append(0.5 * np.abs(p - tot).sum())
    return float(np.mean(tv))


def drift_bands(ys, xs, centroid, rng, sigma=NOISE_SIGMA):
    """Assign each dot to a band by quantising its (noisy) drift vector.
    Returns (band_id per dot, translate vector per band)."""
    p = np.stack([xs, ys], 1).astype(float)
    pn = p + rng.normal(0, sigma, p.shape)
    d = DRIFT * (np.asarray(centroid) - pn)
    lo, hi = 2.0, 60.0
    for _ in range(40):                      # bisect quantum so we land near N_BANDS
        q = (lo + hi) / 2
        cells = np.floor(d / q).astype(int)
        keys = cells[:, 0] * 100000 + cells[:, 1]
        n = len(np.unique(keys))
        if n > N_BANDS:
            lo = q
        else:
            hi = q
    uniq, inv = np.unique(keys, return_inverse=True)
    dtrue = DRIFT * (np.asarray(centroid) - p)
    vec = np.zeros((len(uniq), 2))
    for b in range(len(uniq)):
        vec[b] = dtrue[inv == b].mean(0)
    return inv, vec


def straight_boundary(band: np.ndarray, ys, xs, min_run=6) -> float:
    """Fraction of band-boundary pixels that lie on axis-aligned straight runs
    of >= min_run. Low = organic; high = you rebuilt a grid."""
    lab = -np.ones((PH, PW), int)
    lab[ys, xs] = band
    # Fill the gaps between dither dots with the nearest dot's band so the
    # band boundary is a continuous curve we can measure.
    _, (iy, ix) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    lab = lab[iy, ix]
    total = straight = 0
    for axis in (0, 1):
        a = lab if axis == 0 else lab.T
        b = (a[:, :-1] != a[:, 1:]) & (a[:, :-1] >= 0) & (a[:, 1:] >= 0)   # boundary between cols
        # straight runs go along the other axis: consecutive rows with a boundary at same col
        for col in range(b.shape[1]):
            colv = b[:, col]
            total += int(colv.sum())
            run = 0
            for v in colv:
                if v:
                    run += 1
                else:
                    if run >= min_run:
                        straight += run
                    run = 0
            if run >= min_run:
                straight += run
    return straight / max(total, 1)


# ----- optimal transport ---------------------------------------------------
def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Permutation such that a[i] -> b[perm[i]] minimises total squared travel."""
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    perm = np.empty(len(a), int)
    perm[r] = c
    return perm


# ----- SVG pieces -----------------------------------------------------------
def info_rows(t) -> str:
    out = []
    y = Y_ROW0
    for row in ROWS:
        if row is not None:
            label, value = row
            nl, nv = len(label), len(value)
            nlead = COLS - nl - nv - 2
            out.append(
                f'<text x="{X0}" y="{y}" fill="{t["chrome"]}" textLength="{nl * CW:.1f}" '
                f'lengthAdjust="spacingAndGlyphs">{esc(label)}</text>'
                f'<text x="{X0 + (nl + 1) * CW:.1f}" y="{y}" fill="{t["muted"]}" '
                f'textLength="{nlead * CW:.1f}" lengthAdjust="spacingAndGlyphs">{"." * nlead}</text>'
                f'<text x="{XR - nv * CW:.1f}" y="{y}" fill="{t["text"]}" textLength="{nv * CW:.1f}" '
                f'lengthAdjust="spacingAndGlyphs">{esc(value)}</text>')
        y += ROW_GAP
    return "\n".join(out)


def chrome(t, theme) -> tuple[str, str]:
    pill_w = len(HANDLE) * PILL_FS * 0.6 + 26
    head = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">
<title>V. Siva Teja - profile.sh --live ({theme})</title>
<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{HEIGHT - 1}" rx="14" fill="{t["bg"]}" stroke="{t["chrome"]}" stroke-opacity="{t["border_op"]}"/>
<circle cx="28" cy="20" r="6" fill="#FF5F56"/><circle cx="48" cy="20" r="6" fill="#FFBD2E"/><circle cx="68" cy="20" r="6" fill="#27C93F"/>
<text x="{WIDTH / 2}" y="25" text-anchor="middle" font-size="13" fill="{t["title"]}">profile.sh --live</text>
<line x1="0" y1="40.5" x2="{WIDTH}" y2="40.5" stroke="{t["chrome"]}" stroke-opacity="{t["border_op"]}"/>
<rect x="{FX}" y="{FY}" width="{FW}" height="{FH}" rx="8" fill="{t["frame_fill"]}" fill-opacity="{t["frame_fill_op"]}" stroke="{t["chrome"]}" stroke-opacity="0.5"/>
<text x="{FX + 14}" y="{FY + 22}" font-size="12" letter-spacing="2" fill="{t["chrome"]}">VISUAL.MAP</text>
<text x="{X0}" y="{Y_HDR}" font-size="{HDR_FS}" letter-spacing="2" fill="{t["chrome"]}">SYSTEM.INFO</text>
<circle cx="{XR - 46:.1f}" cy="{Y_HDR - 4}" r="4" fill="{t["live"]}"><animate attributeName="opacity" values="1;0.15;1" dur="1.4s" repeatCount="indefinite"/></circle>
<text x="{XR:.1f}" y="{Y_HDR}" text-anchor="end" font-size="{LIVE_FS}" letter-spacing="2" fill="{t["live"]}">LIVE</text>
<rect x="{X0}" y="{Y_PILL}" width="{pill_w:.1f}" height="26" rx="13" fill="{t["accent"]}"/>
<text x="{X0 + 13}" y="{Y_PILL + 18}" font-size="{PILL_FS}" font-weight="700" fill="{t["bg"]}" textLength="{len(HANDLE) * PILL_FS * 0.6:.1f}" lengthAdjust="spacingAndGlyphs">{HANDLE}</text>
<g font-size="{ROW_FS}">
{info_rows(t)}
</g>
'''
    return head, "</svg>\n"


def build(theme: str) -> dict:
    t = THEMES[theme]
    rng = np.random.default_rng(36)
    dots = np.load(os.path.join(OUT, f"portrait_{theme}.npy"))
    ys, xs = np.nonzero(dots)
    logos = [np.load(os.path.join(OUT, f"logo_{n}.npy")) for n in LOGOS]
    centroid = logos[0].mean(0)

    # --- intro layer
    g = intro_groups(len(ys), rng)
    ev = evenness(g, ys, xs)
    intro = [f'<g id="intro" fill="{t["portrait"]}" shape-rendering="crispEdges">',
             f'<set attributeName="opacity" to="0" begin="{INTRO_DUR}s" fill="freeze"/>']
    for k in range(INTRO_GROUPS):
        sel = g == k
        d = runs_path(ys[sel], xs[sel])
        begin = k * INTRO_SPREAD / INTRO_GROUPS
        intro.append(f'<g opacity="0"><animate attributeName="opacity" values="0;1" begin="{begin:.3f}s" '
                     f'dur="{INTRO_FADE}s" fill="freeze"/><path d="{d}"/></g>')
    intro.append("</g>")

    # --- loop layer A: drift bands
    band, vec = drift_bands(ys, xs, centroid, rng)
    sb = straight_boundary(band, ys, xs)
    sb_grid = straight_boundary(drift_bands(ys, xs, centroid, rng, sigma=0.0)[0], ys, xs)
    op_vals = ";".join(["1", "1"] + ["0"] * (N_KEYS - 3) + ["1"])
    loop = [f'<g id="portrait" opacity="0" fill="{t["portrait"]}" shape-rendering="crispEdges">',
            f'<set attributeName="opacity" to="1" begin="{INTRO_DUR}s" fill="freeze"/>',
            f'<g><animate attributeName="opacity" values="{op_vals}" keyTimes="{KEYTIMES}" '
            f'calcMode="spline" keySplines="{EASE}" dur="{LOOP}s" begin="{INTRO_DUR}s" repeatCount="indefinite"/>']
    for b in range(len(vec)):
        sel = band == b
        dx, dy = vec[b]
        mv = f"{dx:.1f} {dy:.1f}"
        tv = ";".join(["0 0", "0 0"] + [mv] * (N_KEYS - 3) + ["0 0"])
        loop.append(f'<g><animateTransform attributeName="transform" type="translate" values="{tv}" '
                    f'keyTimes="{KEYTIMES}" calcMode="spline" keySplines="{EASE}" dur="{LOOP}s" '
                    f'begin="{INTRO_DUR}s" repeatCount="indefinite"/><path d="{runs_path(ys[sel], xs[sel])}"/></g>')
    loop.append("</g></g>")

    # --- loop layer B: travellers
    # chain optimal-transport matchings: traveller i visits logos[0][i] -> logos[1][idx] -> ...
    idx = np.arange(len(logos[0]))
    stops = [logos[0]]
    for a, b in zip(logos, logos[1:]):
        idx = match(a, b)[idx]
        stops.append(b[idx])
    trav_op = ";".join(["0", "0"] + ["1"] * (N_KEYS - 3) + ["0"])
    trav = [f'<g id="travellers" opacity="0" fill="{t["portrait"]}">',
            f'<animate attributeName="opacity" values="{trav_op}" keyTimes="{KEYTIMES}" '
            f'dur="{LOOP}s" begin="{INTRO_DUR}s" repeatCount="indefinite"/>']
    for i in range(len(logos[0])):
        pts = [f"{s[i][0]:.1f} {s[i][1]:.1f}" for s in stops]
        tv = ";".join([pts[0]] * 4 + [p for p in pts[1:] for _ in (0, 1)] + [pts[0]])
        trav.append(f'<rect x="-1" y="-1" width="2" height="2"><animateTransform attributeName="transform" '
                    f'type="translate" values="{tv}" keyTimes="{KEYTIMES}" dur="{LOOP}s" '
                    f'begin="{INTRO_DUR}s" repeatCount="indefinite"/></rect>')
    trav.append("</g>")

    head, tail = chrome(t, theme)
    body = (f'<g transform="translate({PX} {PY}) scale({SCALE})">\n'
            + "\n".join(intro) + "\n" + "\n".join(loop) + "\n" + "\n".join(trav) + "\n</g>\n")
    svg = head + body + tail
    path = os.path.join(ROOT, f"{theme}.svg")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    return dict(path=path, dots=len(ys), bands=len(vec), evenness=ev, straight=sb,
                straight_grid=sb_grid, kb=len(svg.encode("utf-8")) / 1024)


if __name__ == "__main__":
    for theme in (sys.argv[1:] or ["dark", "light"]):
        r = build(theme)
        print(f"{theme}: {r['dots']} dots, {r['bands']} bands, {r['kb']:.0f} KB | "
              f"intro evenness {r['evenness']:.3f} (random ~0.05) | "
              f"straight-boundary {r['straight']:.3f} (no-noise would be {r['straight_grid']:.3f})")
