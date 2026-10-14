"""Portrait pipeline: crop -> segment -> tone -> 1-bit Floyd-Steinberg dither.

Outputs (in generator/out/):
  portrait_dark.npy   bool (H, W)  dots = lit parts of the subject, background removed
  portrait_light.npy  bool (H, W)  dots = dark parts of the photo
  mask.npy            bool (H, W)  subject mask
  preview_*.png       6x previews for eyeballing

Usage:  python portrait.py <photo> [--crop x0 y0 x1 y1]
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

W, H = 300, 340
DARK_GAIN = 0.92                 # overall density multiplier
DARK_KNEE, DARK_KNEE_SLOPE = 0.55, 0.3   # highlights above the knee compressed
DARK_FLOOR = 0.10
CLAHE_CLIP = 1.6
PERSON_THRESHOLD = 0.35          # MediaPipe prob; ears/jaw against dark wood sit below 0.5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def crop_head_shoulders(img: Image.Image, box=None) -> Image.Image:
    """Crop to the portrait aspect (300:340). Default: centred, head + shoulders."""
    if box is None:
        w, h = img.size
        target = W / H
        if w / h > target:  # too wide
            cw = int(h * target)
            box = ((w - cw) // 2, 0, (w + cw) // 2, h)
        else:
            ch = int(w / target)
            box = (0, 0, w, ch)
    return img.crop(box)


def person_probability(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel subject probability from MediaPipe selfie segmentation.
    A learned person model, so skin against wood (ears, jaw) is not a colour
    guess the way it is for GrabCut."""
    import mediapipe as mp
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as seg:
        return seg.process(np.ascontiguousarray(rgb)).segmentation_mask


def segment_subject(rgb: np.ndarray, fg_hints=()) -> np.ndarray:
    """MediaPipe person mask, edge refined by GrabCut, then morphological clean-up.
    fg_hints: ellipses (cx, cy, rx, ry) in this image's pixels that are certainly
    subject -- used for parts the model drops (ears in shadow against dark wood).
    Returns bool mask (True = subject)."""
    h, w = rgb.shape[:2]
    prob = person_probability(rgb)
    m = largest_component(prob > PERSON_THRESHOLD)
    hint = np.zeros((h, w), bool)
    yy, xx = np.mgrid[:h, :w]
    for cx, cy, rx, ry in fg_hints:
        hint |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    m |= hint

    # Refine the edge with GrabCut: trust the model deep inside / far outside,
    # let the colour model re-decide only a thin band around the boundary.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    band = int(0.012 * w)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[ndimage.binary_dilation(m, iterations=band)] = cv2.GC_PR_BGD
    mask[m] = cv2.GC_PR_FGD
    mask[ndimage.binary_erosion(m, iterations=band)] = cv2.GC_FGD
    mask[~ndimage.binary_dilation(m, iterations=band)] = cv2.GC_BGD
    mask[hint] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    m = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD) | hint

    m = ndimage.binary_closing(m, structure=np.ones((5, 5)), iterations=2)
    m = ndimage.binary_fill_holes(m)
    m = largest_component(m)
    # Smooth the stair-stepped edge: blur the mask and re-threshold.
    sm = ndimage.gaussian_filter(m.astype(float), sigma=max(1.0, w / 300))
    return sm > 0.5


def largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def tone(gray: Image.Image) -> np.ndarray:
    """CLAHE (mild) -> autocontrast(cutoff=1) -> contrast 1.3x -> UnsharpMask(3, 140). Returns float 0..1."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    g = Image.fromarray(clahe.apply(np.asarray(gray)))
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.3)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    return np.asarray(g, dtype=np.float64) / 255.0


def floyd_steinberg(v: np.ndarray) -> np.ndarray:
    """1-bit serpentine Floyd-Steinberg. v in 0..1 where 1 = dot. Returns bool."""
    a = v.copy()
    h, w = a.shape
    out = np.zeros((h, w), bool)
    for y in range(h):
        rng = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        step = 1 if y % 2 == 0 else -1
        for x in rng:
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0.5
            err = old - new
            if 0 <= x + step < w:
                a[y, x + step] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    a[y + 1, x - step] += err * 3 / 16
                a[y + 1, x] += err * 5 / 16
                if 0 <= x + step < w:
                    a[y + 1, x + step] += err * 1 / 16
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--light-keep-bg", action="store_true",
                    help="light mode keeps the photo background (prompt default)")
    ap.add_argument("--fg", action="append", nargs=4, type=float, default=[],
                    metavar=("CX", "CY", "RX", "RY"),
                    help="ellipse in CROP pixels that is certainly subject (repeatable)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    img = Image.open(args.photo).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = crop_head_shoulders(img, tuple(args.crop) if args.crop else None)
    print("crop size", img.size)

    # Segment at ~800px wide for GrabCut accuracy, then downsample.
    seg_w = 800
    seg = img.resize((seg_w, int(seg_w * H / W)), Image.LANCZOS)
    f = seg_w / img.size[0]
    hints = [(cx * f, cy * f, rx * f, ry * f) for cx, cy, rx, ry in args.fg]
    mask_hi = segment_subject(np.asarray(seg), hints)
    mask = np.asarray(Image.fromarray(mask_hi.astype(np.uint8) * 255)
                      .resize((W, H), Image.LANCZOS)) > 127
    print("subject coverage %.1f%%" % (100 * mask.mean()))

    small = img.resize((W, H), Image.LANCZOS)
    gray = small.convert("L")

    # ---- dark mode: lit subject on panel; background hard-black so no error bleed
    v = tone(gray)
    # Compress highlights (a white shirt at full density becomes a solid slab)
    # and add a density floor inside the subject so black hair and shadowed
    # ears still trace the head silhouette instead of vanishing into the panel.
    # Soft knee: skin mid-tones keep their density, only highlights above the
    # knee are compressed, so the face stays denser than a gamma curve allows.
    knee = DARK_KNEE
    v_knee = np.minimum(v, knee) + np.maximum(v - knee, 0.0) * DARK_KNEE_SLOPE
    v_dark = np.where(mask, DARK_FLOOR + (1 - DARK_FLOOR) * DARK_GAIN * v_knee, 0.0)
    dots_dark = floyd_steinberg(v_dark)
    mask_core = ndimage.binary_erosion(mask, iterations=1)
    dots_dark &= mask_core  # hard-clear bleed at the mask edge

    # ---- light mode: dark parts of the photo
    v_light = 1.0 - v
    if not args.light_keep_bg:
        v_light = np.where(mask, v_light, 0.0)
    dots_light = floyd_steinberg(v_light)
    if not args.light_keep_bg:
        dots_light &= mask_core

    for name, d in (("dark", dots_dark), ("light", dots_light)):
        np.save(os.path.join(OUT, f"portrait_{name}.npy"), d)
        print(f"{name}: {int(d.sum())} dots, ink {d.mean():.3f}")
        pv = Image.fromarray((~d if name == "light" else d).astype(np.uint8) * 255)
        pv = pv.resize((W * 2, H * 2), Image.NEAREST)
        pv.save(os.path.join(OUT, f"preview_{name}.png"))
    np.save(os.path.join(OUT, "mask.npy"), mask)
    Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(OUT, "preview_mask.png"))
    small.save(os.path.join(OUT, "preview_crop.png"))


if __name__ == "__main__":
    main()
