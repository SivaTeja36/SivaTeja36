"""connect.svg -- "Let's build something together": dithered portrait with a
handle sticker on the left, contact cards on the right.  Usage: python connect.py"""
import os

import numpy as np

from banner import runs_path
from panels import (THEME, out_path, BG2, CARD, CARD_STROKE, CYAN, EMERALD, MONO, MUTED, PINK, PURPLE, ROOT, SANS,
                    TEXT, defs, esc, eyebrow, fade_in, frame, icon, icon_svg, title)

W, H = 1280, 470
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

CARDS = [  # (icon, colour, title, subtitle, href)
    ("github", TEXT, "GitHub", "SivaTeja36", "https://github.com/SivaTeja36"),
    ("gmail", "#EA4335", "Email", "vantasivateja@gmail.com", "mailto:vantasivateja@gmail.com"),
    ("linkedin", "#0A66C2", "LinkedIn", "in/siva-teja-vanta-99213b257", "https://www.linkedin.com/in/siva-teja-vanta-99213b257/"),
    ("globe", EMERALD, "Portfolio", "coming soon", None),
]
QUOTE = "“Build it right, ship it, then make it faster.”"


def portrait(x, y, scale) -> str:
    dots = np.load(os.path.join(OUT, f"portrait_{THEME}.npy"))   # lit-on-dark / dark-on-light
    ys, xs = np.nonzero(dots)
    d = runs_path(ys, xs)
    h, w = dots.shape
    cx, cy = x + w * scale / 2, y + h * scale * 0.42
    return (f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="150" fill="{PURPLE}" fill-opacity="0.22" filter="url(#glow)">'
            f'<animate attributeName="fill-opacity" values="0.16;0.3;0.16" dur="5s" repeatCount="indefinite"/></circle>'
            f'<g transform="translate({x} {y}) scale({scale})" fill="{PURPLE}" shape-rendering="crispEdges">'
            f'<path d="{d}"/></g>')


def sticker(x, y) -> str:
    text = "@SivaTeja36"
    w = len(text) * 20 * 0.6 + 44
    return (f'<g transform="translate({x} {y})">'
            f'<g transform="rotate(-8)"><animateTransform attributeName="transform" type="rotate" '
            f'values="-8;-5;-8" dur="6s" repeatCount="indefinite"/>'
            f'<rect x="0" y="-22" width="{w:.0f}" height="44" rx="22" fill="{BG2}" stroke="{PINK}" stroke-width="2"/>'
            f'<rect x="4" y="-18" width="{w - 8:.0f}" height="36" rx="18" fill="{PINK}" fill-opacity="0.12"/>'
            f'<text x="22" y="7" font-family="{MONO}" font-size="20" font-weight="700" fill="{PINK}">{text}</text>'
            f'<path d="M{w - 6:.0f} -26 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3z" fill="{CYAN}">'
            f'<animate attributeName="opacity" values="1;0.3;1" dur="2.2s" repeatCount="indefinite"/></path>'
            f'</g></g>')


def card(x, y, w, h, ic, col, head, sub, href, begin) -> str:
    inner = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{CARD}" stroke="{CARD_STROKE}"/>'
             f'<rect x="{x + 18}" y="{y + (h - 44) / 2}" width="44" height="44" rx="11" fill="{col}" fill-opacity="0.14" '
             f'stroke="{col}" stroke-opacity="0.45"/>'
             f'{icon_svg(ic, x + 18 + 11, y + (h - 44) / 2 + 11, 22, col)}'
             f'<text x="{x + 78}" y="{y + h / 2 - 4}" font-family="{SANS}" font-size="16" font-weight="700" fill="{TEXT}">{esc(head)}</text>'
             f'<text x="{x + 78}" y="{y + h / 2 + 17}" font-family="{MONO}" font-size="12" fill="{MUTED}">{esc(sub)}</text>')
    if href:
        inner = f'<a href="{href}" target="_blank">{inner}</a>'
    return f'<g opacity="0">{fade_in(begin, 0.6, 12)}{inner}</g>'


def build() -> str:
    x0 = 500
    cw, ch, gap = 362, 84, 18
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Let\'s connect">',
             "<title>Let's connect</title>",
             "<desc>Siva Teja's dithered portrait with his handle, beside cards linking to GitHub, email, LinkedIn and portfolio.</desc>",
             defs(), frame(W, H),
             portrait(70, 44, 1.12), sticker(262, 352),
             eyebrow(x0, 112, "// LET'S CONNECT", PINK),
             title(x0, 152, "Let’s build something together", 30),
             f'<text x="{x0}" y="{182}" font-family="{SANS}" font-size="15" font-weight="500" fill="{MUTED}">'
             f'Open to backend, Azure and AI/ML collaborations, side projects and good problems.</text>']
    for i, (ic, col, head, sub, href) in enumerate(CARDS):
        cx = x0 + (i % 2) * (cw + gap)
        cy = 212 + (i // 2) * (ch + gap)
        parts.append(card(cx, cy, cw, ch, ic, col, head, sub, href, 0.3 + i * 0.12))
    parts.append(f'<text x="{x0}" y="{H - 38}" font-family="{MONO}" font-size="13" fill="{PINK}">{esc(QUOTE)}</text>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    out = out_path("connect")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    print(f"{os.path.basename(out)} {W}x{H} {len(svg.encode()) / 1024:.0f} KB")
    return out


if __name__ == "__main__":
    build()
