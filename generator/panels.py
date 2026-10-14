"""Shared pieces for the card-style SVG panels (stack.svg, connect.svg):
embedded fonts, brand icons, the panel frame, and small helpers."""
import base64
import os
import re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
ROOT = os.path.dirname(HERE)

# palette (matches the banner)
BG = "#0D1220"
BG2 = "#121A2C"
CARD = "#161D30"
CARD_STROKE = "#273049"
TEXT = "#E6E9F2"
MUTED = "#8B93A7"
CYAN = "#22D3EE"
PURPLE = "#A78BFA"
PINK = "#F472B6"
EMERALD = "#34D399"
AMBER = "#FBBF24"
BLUE = "#60A5FA"

SANS = "'SG', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JBM', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# icons that simple-icons does not ship, drawn by hand (24x24 box)
CUSTOM_ICONS = {
    "azure": ("M5.483 21.3H24L14.025 4.013l-3.038 8.347 5.836 6.938L5.483 21.3zM13.23 2.7L6.105 "
              "8.677 0 19.253h5.505v.014L13.23 2.7z", "#0078D4"),
    "sql": ("M12 2C7.03 2 3 3.57 3 5.5v13C3 20.43 7.03 22 12 22s9-1.57 9-3.5v-13C21 3.57 16.97 2 12 2zm0 2c4.42 "
            "0 7 1.35 7 1.5S16.42 7 12 7 5 5.65 5 5.5 7.58 4 12 4zm7 14.5c0 .15-2.58 1.5-7 1.5s-7-1.35-7-1.5v-3.1C6.6 "
            "18.3 9.2 19 12 19s5.4-.7 7-1.6v1.1zm0-4.5c0 .15-2.58 1.5-7 1.5S5 14.15 5 14v-3.1C6.6 11.8 9.2 12.5 12 "
            "12.5s5.4-.7 7-1.6V14zm0-4.5c0 .15-2.58 1.5-7 1.5S5 9.65 5 9.5V7.4C6.6 8.3 9.2 9 12 9s5.4-.7 7-1.6v2.1z",
            "#60A5FA"),
    "globe": ("M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm6.93 6h-2.95a15.7 15.7 0 0 0-1.38-3.56A8.03 8.03 0 0 1 "
              "18.93 8zM12 4.04c.83 1.2 1.48 2.53 1.91 3.96h-3.82c.43-1.43 1.08-2.76 1.91-3.96zM4.26 14A8 8 0 0 1 4 "
              "12c0-.69.1-1.36.26-2h3.38a16.5 16.5 0 0 0 0 4H4.26zm.81 2h2.95c.32 1.25.78 2.45 1.38 3.56A8 8 0 0 1 "
              "5.07 16zM8.03 8H5.07a8 8 0 0 1 4.33-3.56A15.7 15.7 0 0 0 8.03 8zM12 19.96c-.83-1.2-1.48-2.53-1.91-"
              "3.96h3.82c-.43 1.43-1.08 2.76-1.91 3.96zM14.34 14H9.66a14.7 14.7 0 0 1 0-4h4.68a14.7 14.7 0 0 1 0 "
              "4zm.25 5.56c.6-1.11 1.06-2.31 1.38-3.56h2.95a8.03 8.03 0 0 1-4.33 3.56zM16.36 14a16.5 16.5 0 0 0 "
              "0-4h3.38c.16.64.26 1.31.26 2s-.1 1.36-.26 2h-3.38z", "#34D399"),
    "linkedin": ("M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.13 1.45-2.13 2.94v5.67H9.35V9h3.41v1.56h.05"
                 "c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 "
                 "2.06 0 0 1 0 4.13zM7.12 20.45H3.55V9h3.57v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 "
                 "24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z", "#0A66C2"),
}

# brands whose official colour is too dark for a dark panel
COLOR_OVERRIDE = {
    "flask": "#E6E9F2", "github": "#E6E9F2", "ollama": "#E6E9F2", "django": "#44B78B",
    "pandas": "#A78BFA", "numpy": "#4DABCF", "sqlite": "#4DA8DA", "langchain": "#2DD4BF",
    "linux": "#FCC624",
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def icon(name: str):
    """-> (path d, colour) for a simple-icons slug or a custom icon."""
    if name in CUSTOM_ICONS:
        return CUSTOM_ICONS[name]
    root = ET.parse(os.path.join(ASSETS, "icons", f"{name}.svg")).getroot()
    d = next(el.get("d") for el in root.iter() if el.tag.split("}")[-1] == "path")
    col = COLOR_OVERRIDE.get(name, root.get("fill", "#E6E9F2"))
    return d, col


def icon_svg(name: str, x: float, y: float, size: float, color: str | None = None) -> str:
    d, col = icon(name)
    return (f'<g transform="translate({x:.1f} {y:.1f}) scale({size / 24:.4f})">'
            f'<path d="{d}" fill="{color or col}"/></g>')


def font_css() -> str:
    out = []
    for fam, fn in (("SG", "SpaceGrotesk.woff2"), ("JBM", "JetBrainsMono.woff2")):
        with open(os.path.join(ASSETS, "fonts", fn), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        out.append(f"@font-face{{font-family:'{fam}';font-weight:100 900;"
                   f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "".join(out)


def defs(extra: str = "") -> str:
    return f'''<defs>
<style>{font_css()}</style>
<linearGradient id="edge" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="{CYAN}"/><stop offset="0.5" stop-color="{PURPLE}"/><stop offset="1" stop-color="{PINK}"/>
</linearGradient>
<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
  <circle cx="1" cy="1" r="1" fill="{TEXT}" fill-opacity="0.045"/>
</pattern>
<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
  <feGaussianBlur stdDeviation="14"/>
</filter>
<filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
  <feGaussianBlur stdDeviation="2.5"/>
</filter>
{extra}
</defs>'''


def frame(w: int, h: int) -> str:
    """Panel background: dark rounded card, dot texture, gradient edge."""
    return (f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="24" fill="{BG}"/>'
            f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="24" fill="url(#dots)"/>'
            f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="24" fill="none" '
            f'stroke="url(#edge)" stroke-width="1.5" stroke-opacity="0.85"/>')


def eyebrow(x: float, y: float, text: str, color: str) -> str:
    return (f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="12" font-weight="700" '
            f'letter-spacing="3" fill="{color}">{esc(text)}</text>')


def title(x: float, y: float, text: str, size: int = 30) -> str:
    return (f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" font-weight="700" '
            f'fill="{TEXT}">{esc(text)}</text>')


def mono_width(text: str, size: float) -> float:
    return len(text) * size * 0.6


def fade_in(begin: float, dur: float = 0.5, dy: float = 8) -> str:
    """Opacity + small upward slide, run once."""
    return (f'<animate attributeName="opacity" values="0;1" begin="{begin:.2f}s" dur="{dur}s" fill="freeze"/>'
            f'<animateTransform attributeName="transform" type="translate" values="0 {dy};0 0" '
            f'begin="{begin:.2f}s" dur="{dur}s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>')
