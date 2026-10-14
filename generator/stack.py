"""stack.svg -- "Tools I build with": orbiting brand icons around a code orb on
the left, grouped tech chips on the right.  Usage: python stack.py"""
import math
import os

from panels import (THEME, out_path, AMBER, BG2, BLUE, CARD, CARD_STROKE, CYAN, EMERALD, MONO, MUTED, PINK,
                    PURPLE, ROOT, SANS, TEXT, defs, esc, eyebrow, fade_in, frame, icon,
                    icon_svg, mono_width, title)

W = 1280
GROUPS = [
    ("LANGUAGES & FRONTEND", CYAN, [("Python", "python"), ("TypeScript", "typescript"), ("JavaScript", "javascript"), ("SQL", "sql"),
                                    ("HTML5", "html5"), ("CSS", "css"), ("React", "react"),
                                    ("Streamlit", "streamlit")]),
    ("BACKEND", PURPLE, [("FastAPI", "fastapi"), ("Django", "django"), ("Flask", "flask"),
                         ("SQLAlchemy", "sqlalchemy"), ("Pytest", "pytest"), ("RabbitMQ", "rabbitmq")]),
    ("CLOUD & DEVOPS", AMBER, [("Azure", "azure"), ("Terraform", "terraform"), ("Docker", "docker"),
                               ("Azure DevOps", "azure"), ("Linux", "linux"), ("Git", "git"),
                               ("GitHub", "github")]),
    ("AI / ML", PINK, [("Scikit-learn", "scikitlearn"), ("OpenCV", "opencv"), ("LangChain", "langchain"), ("LangGraph", "langgraph"),
                       ("Ollama", "ollama"), ("n8n", "n8n"), ("NumPy", "numpy"), ("Pandas", "pandas")]),
    ("DATA", BLUE, [("PostgreSQL", "postgresql"), ("PgVector", "postgresql"), ("Redis", "redis"), ("SQLite", "sqlite")]),
]
ORBIT_ICONS = [  # (icon, orbit index)
    ("python", 0), ("fastapi", 0), ("postgresql", 0),
    ("azure", 1), ("terraform", 1), ("docker", 1),
    ("react", 2), ("langchain", 2), ("rabbitmq", 2),
]

# layout
X_RIGHT = 470
CHIP_H, CHIP_FS, CHIP_PAD, CHIP_GAP, ROW_GAP = 34, 13, 12, 10, 10
GROUP_GAP = 16
RIGHT_W = W - X_RIGHT - 44
Y_GROUPS = 150


def chip(x, y, label, ic, begin) -> tuple[str, float]:
    d, col = icon(ic)
    w = CHIP_PAD + 16 + 8 + mono_width(label, CHIP_FS) + CHIP_PAD
    s = (f'<g opacity="0">{fade_in(begin)}'
         f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{CHIP_H}" rx="9" fill="{col}" fill-opacity="0.10" '
         f'stroke="{col}" stroke-opacity="0.55"/>'
         f'{icon_svg(ic, x + CHIP_PAD, y + (CHIP_H - 16) / 2, 16)}'
         f'<text x="{x + CHIP_PAD + 24:.1f}" y="{y + CHIP_H / 2 + 4.5:.1f}" font-family="{MONO}" font-size="{CHIP_FS}" '
         f'font-weight="500" fill="{TEXT}">{esc(label)}</text></g>')
    return s, w


def right_panel() -> tuple[str, int]:
    out = []
    y = Y_GROUPS
    t = 0.2
    for name, col, items in GROUPS:
        out.append(f'<g opacity="0">{fade_in(t)}'
                   f'<rect x="{X_RIGHT}" y="{y - 4}" width="12" height="3" rx="1.5" fill="{col}"/>'
                   f'<text x="{X_RIGHT + 20}" y="{y}" font-family="{MONO}" font-size="11" font-weight="700" '
                   f'letter-spacing="3" fill="{MUTED}">{esc(name)}</text></g>')
        y += 12
        x = X_RIGHT
        for label, ic in items:
            t += 0.06
            s, w = chip(x, y, label, ic, t)
            if x + w > X_RIGHT + RIGHT_W:          # wrap
                x = X_RIGHT
                y += CHIP_H + ROW_GAP
                s, w = chip(x, y, label, ic, t)
            out.append(s)
            x += w + CHIP_GAP
        y += CHIP_H + GROUP_GAP + 14
    return "\n".join(out), y


def orbit_cluster(cx, cy) -> str:
    out = [f'<g transform="translate({cx} {cy})">']
    # glow + orb
    out.append(f'<circle r="70" fill="{PURPLE}" fill-opacity="0.35" filter="url(#glow)">'
               f'<animate attributeName="fill-opacity" values="0.25;0.45;0.25" dur="4s" repeatCount="indefinite"/></circle>')
    out.append('<g><animateTransform attributeName="transform" type="scale" values="1;1.04;1" dur="4s" repeatCount="indefinite"/>'
               f'<circle r="52" fill="url(#orb)"/>'
               f'<circle r="52" fill="none" stroke="{TEXT}" stroke-opacity="0.18"/>'
               f'<text y="9" text-anchor="middle" font-family="{MONO}" font-size="26" font-weight="700" fill="{TEXT}">&lt;/&gt;</text></g>')
    # dashed ring rotating
    out.append(f'<circle r="68" fill="none" stroke="{CYAN}" stroke-opacity="0.35" stroke-dasharray="3 9">'
               f'<animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="40s" repeatCount="indefinite"/></circle>')
    # orbits
    orbits = [(165, 62, -22, 18.0), (150, 58, 42, 24.0), (178, 50, 104, 30.0)]
    for i, (rx, ry, ang, dur) in enumerate(orbits):
        path = f"M{rx} 0 A{rx} {ry} 0 1 1 {-rx} 0 A{rx} {ry} 0 1 1 {rx} 0 Z"
        col = [CYAN, PURPLE, PINK][i]
        out.append(f'<g transform="rotate({ang})">'
                   f'<path id="orb{i}" d="{path}" fill="none" stroke="{col}" stroke-opacity="0.28" stroke-width="1"/>')
        riders = [r for r in ORBIT_ICONS if r[1] == i]
        for k, (ic, _) in enumerate(riders):
            d, c = icon(ic)
            begin = -dur * k / len(riders)
            out.append(f'<g><animateMotion dur="{dur}s" begin="{begin:.2f}s" repeatCount="indefinite" rotate="0">'
                       f'<mpath href="#orb{i}"/></animateMotion>'
                       f'<g transform="rotate({-ang})">'
                       f'<circle r="17" fill="{BG2}" stroke="{c}" stroke-opacity="0.8"/>'
                       f'<circle r="17" fill="{c}" fill-opacity="0.12"/>'
                       f'{icon_svg(ic, -9, -9, 18)}</g></g>')
        out.append('</g>')
    out.append('</g>')
    return "\n".join(out)


def build() -> str:
    right, y_end = right_panel()
    H = int(math.ceil((y_end + 10) / 10) * 10)
    extra = (f'<radialGradient id="orb" cx="0.35" cy="0.3" r="0.8">'
             f'<stop offset="0" stop-color="#C4B5FD"/><stop offset="0.55" stop-color="{PURPLE}"/>'
             f'<stop offset="1" stop-color="#6D28D9"/></radialGradient>')
    cy = (Y_GROUPS - 20 + H) / 2 + 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Tech stack">',
             '<title>Tech stack</title>',
             '<desc>Brand icons orbiting a code orb, and a grouped grid of languages, backend, cloud, AI/ML and data tools.</desc>',
             defs(extra), frame(W, H),
             eyebrow(44, 62, "// TECH STACK", CYAN), title(44, 100, "Tools I build with", 30),
             f'<line x1="{X_RIGHT - 26}" y1="{Y_GROUPS - 20}" x2="{X_RIGHT - 26}" y2="{H - 36}" stroke="{CARD_STROKE}"/>',
             orbit_cluster(232, cy), right, '</svg>']
    svg = "\n".join(parts)
    out = out_path("stack")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    print(f"{os.path.basename(out)} {W}x{H} {len(svg.encode()) / 1024:.0f} KB")
    return out


if __name__ == "__main__":
    build()
