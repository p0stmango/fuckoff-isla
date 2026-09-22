"""
Generate a print-ready A4 PNG of an Australian 5 km/h speed sign
with the adversarial patch composited onto it at true physical scale.

Includes a rectangular backing plate to match real carpark signs.

Output: printable_sign.png  (A4 @ 300 DPI = 2480 × 3508 px)

Usage:
    python3 make_printable.py --patch patch.png
    python3 make_printable.py --patch patch.png --patch-x 23 --patch-y -10
    python3 make_printable.py --patch patch.png --test   # also runs surrogate model

Print at 100% scale (no fit-to-page) on A4.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── physical constants ────────────────────────────────────────────────────────
DPI           = 300
A4_W_MM       = 210
A4_H_MM       = 297
REAL_SIGN_MM  = 450    # real AU carpark sign outer diameter
SIGN_DIAM_MM  = 190    # rendered circle diameter on A4
REAL_PATCH_MM = 80     # real patch diameter
PATCH_DIAM_MM = REAL_PATCH_MM * SIGN_DIAM_MM / REAL_SIGN_MM   # ~33.8 mm

# Backing plate: roughly 1.3× sign diameter, rounded corners
PLATE_W_MM    = SIGN_DIAM_MM * 1.25
PLATE_H_MM    = SIGN_DIAM_MM * 1.25
PLATE_R_MM    = 8     # corner radius

MM_TO_PX = DPI / 25.4

A4_W_PX    = int(A4_W_MM     * MM_TO_PX)
A4_H_PX    = int(A4_H_MM     * MM_TO_PX)
SIGN_PX    = int(SIGN_DIAM_MM * MM_TO_PX)
PATCH_PX   = int(PATCH_DIAM_MM * MM_TO_PX)
PLATE_W_PX = int(PLATE_W_MM  * MM_TO_PX)
PLATE_H_PX = int(PLATE_H_MM  * MM_TO_PX)
PLATE_R_PX = int(PLATE_R_MM  * MM_TO_PX)

_WHITE      = (255, 255, 255)
_OFF_WHITE  = (235, 235, 232)   # backing plate — slightly off-white like real ali blanks
_RED        = (206, 17,  38)
_BLACK      = (10,  10,  10)
_PLATE_EDGE = (190, 190, 188)   # subtle border on plate

FONT_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def rounded_rect(draw, xy, radius, fill, outline=None, outline_width=2):
    """Draw a filled rounded rectangle."""
    x0, y0, x1, y1 = xy
    r = radius
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)
    draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)
    draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)
    draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)
    if outline:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                outline=outline, width=outline_width)


def render_sign_on_plate(sign_diam_px: int, plate_w_px: int, plate_h_px: int,
                          plate_r_px: int) -> Image.Image:
    """Render sign circle centred on a rectangular backing plate, RGBA."""
    img  = Image.new("RGBA", (plate_w_px, plate_h_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Backing plate
    rounded_rect(draw, [0, 0, plate_w_px - 1, plate_h_px - 1],
                 radius=plate_r_px, fill=_OFF_WHITE + (255,),
                 outline=_PLATE_EDGE + (255,), outline_width=max(2, plate_r_px // 2))

    # Sign circle, centred on plate
    cx = plate_w_px / 2
    cy = plate_h_px / 2
    r  = sign_diam_px / 2 - 1
    ring_w = max(6, int(sign_diam_px * 0.07))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_WHITE + (255,))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=_RED + (255,), width=ring_w)

    # "5"
    font_size = int(sign_diam_px * 0.52)
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), "5", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]),
              "5", fill=_BLACK + (255,), font=font)

    return img


def make_rectangular_patch(patch_img: Image.Image, size_px: int) -> Image.Image:
    """Resize patch to size_px × size_px, no masking — full rectangle."""
    return patch_img.convert("RGB").resize((size_px, size_px), Image.LANCZOS)


def make_printable(patch_path: str, out_path: str,
                   patch_x_mm: float, patch_y_mm: float) -> Image.Image:
    """Returns the canvas so --test can crop and run the model over it."""
    print(f"A4        : {A4_W_PX}×{A4_H_PX} px  ({A4_W_MM}×{A4_H_MM} mm @ {DPI} DPI)")
    print(f"Plate     : {PLATE_W_PX}×{PLATE_H_PX} px  ({PLATE_W_MM:.0f}×{PLATE_H_MM:.0f} mm)")
    print(f"Sign      : {SIGN_PX} px  ({SIGN_DIAM_MM} mm)")
    print(f"Patch     : {PATCH_PX} px  ({PATCH_DIAM_MM:.1f} mm)")
    print(f"Patch pos : ({patch_x_mm:+.1f} mm, {patch_y_mm:+.1f} mm) from sign centre")

    canvas = Image.new("RGB", (A4_W_PX, A4_H_PX), (255, 255, 255))

    # Centre plate on page
    plate = render_sign_on_plate(SIGN_PX, PLATE_W_PX, PLATE_H_PX, PLATE_R_PX)
    plate_x = (A4_W_PX - PLATE_W_PX) // 2
    plate_y = (A4_H_PX - PLATE_H_PX) // 2
    canvas.paste(plate, (plate_x, plate_y), mask=plate.split()[3])

    # Sign centre in canvas coords
    scx = plate_x + PLATE_W_PX // 2
    scy = plate_y + PLATE_H_PX // 2

    # Composite patch — rectangular, pasted directly
    patch_rect = make_rectangular_patch(Image.open(patch_path), PATCH_PX)
    px_left = int(scx + patch_x_mm * MM_TO_PX) - PATCH_PX // 2
    py_top  = int(scy + patch_y_mm * MM_TO_PX) - PATCH_PX // 2
    canvas.paste(patch_rect, (px_left, py_top))

    # Footer
    draw = ImageDraw.Draw(canvas)
    ff   = load_font(int(30 * MM_TO_PX / 10))
    footer = "PRINT AT 100% SCALE — DO NOT FIT TO PAGE"
    fb = draw.textbbox((0, 0), footer, font=ff)
    draw.text(((A4_W_PX - (fb[2] - fb[0])) / 2, A4_H_PX - int(12 * MM_TO_PX)),
              footer, fill=(200, 100, 100), font=ff)

    canvas.save(out_path, dpi=(DPI, DPI))
    print(f"Saved: {out_path}")
    return canvas


def test_surrogate(canvas: Image.Image, patch_path: str, model_path: str = "surrogate.pt",
                   n_runs: int = 20):
    """
    Crop the sign region from the canvas, resize to 224×224, and run the
    surrogate model over it n_runs times (with random EOT each time).
    Reports prediction distribution.
    """
    import torch
    import torchvision.transforms as T
    from collections import Counter

    try:
        from dataset import ALL_SPEEDS, val_transforms, NORMALIZE
        from model import load as load_model, get_device
        from eot import eot_batch
    except ImportError as e:
        print(f"Cannot import project modules: {e}")
        print("Run this from your project directory.")
        return

    device = get_device()
    model  = load_model(model_path).to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Crop just the sign circle (what the car's detector would pass to the classifier)
    # Sign is centred on canvas — crop to SIGN_PX with a small margin
    cw, ch = canvas.size
    cx, cy = cw // 2, ch // 2
    margin = int(SIGN_PX * 0.12)   # ~12% margin around sign
    half   = SIGN_PX // 2 + margin
    crop   = canvas.crop((cx - half, cy - half, cx + half, cy + half))
    crop_224 = crop.resize((224, 224), Image.LANCZOS)

    arr = np.array(crop_224).astype(np.float32) / 255.
    t   = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    img_norm = (t - mean) / std

    preds = []
    with torch.no_grad():
        for _ in range(n_runs):
            # EOT
            t01 = img_norm * std + mean
            t01 = eot_batch(t01)
            t_eot = (t01 - mean) / std
            pred = model(t_eot).argmax(1).item()
            preds.append(ALL_SPEEDS[pred])

    counts = Counter(preds)
    print(f"\n── Surrogate predictions on printable ({n_runs} EOT runs) ──")
    for speed, count in counts.most_common():
        bar = "█" * count
        print(f"  {speed:>4} km/h : {bar} ({count}/{n_runs})")

    # Save the cropped region for inspection
    crop_224.save("test_crop.png")
    print(f"Cropped sign saved to test_crop.png — check this looks right")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch",   required=True)
    p.add_argument("--out",     default="printable_sign.png")
    p.add_argument("--patch-x", type=float, default=23.0,
                   help="Patch X offset from sign centre in mm (+ = right)")
    p.add_argument("--patch-y", type=float, default=-10.0,
                   help="Patch Y offset from sign centre in mm (+ = down)")
    p.add_argument("--test",    action="store_true",
                   help="Run surrogate model over the rendered printable")
    p.add_argument("--model",   default="surrogate.pt")
    p.add_argument("--n-runs",  type=int, default=20)
    args = p.parse_args()

    canvas = make_printable(args.patch, args.out, args.patch_x, args.patch_y)
    if args.test:
        test_surrogate(canvas, args.patch, args.model, args.n_runs)
