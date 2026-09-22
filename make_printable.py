"""
Generate a print-ready A4 PNG of an Australian 5 km/h speed sign
with the adversarial patch composited onto it at true physical scale.

Includes a rectangular backing plate to match real carpark signs.

Output: printable_sign.png  (A4 @ 300 DPI = 2480 × 3508 px)

Usage:
    python3 make_printable.py --patch patch.png
    python3 make_printable.py --patch patch.png --patch-x 23 --patch-y -10
    python3 make_printable.py --patch patch.png --test   # also runs surrogate ensemble

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
REAL_SIGN_MM  = 300    # real AU carpark sign diameter (typical 300mm round, not 450mm)
SIGN_DIAM_MM  = 190    # rendered circle diameter on A4
REAL_PATCH_MM = 80     # real patch side length (square)
PATCH_MM      = REAL_PATCH_MM * SIGN_DIAM_MM / REAL_SIGN_MM   # ~50.7 mm on A4

# Backing plate: roughly 1.25× sign diameter, rounded corners
PLATE_W_MM    = SIGN_DIAM_MM * 1.25
PLATE_H_MM    = SIGN_DIAM_MM * 1.25
PLATE_R_MM    = 8     # corner radius

MM_TO_PX = DPI / 25.4

A4_W_PX    = int(A4_W_MM     * MM_TO_PX)
A4_H_PX    = int(A4_H_MM     * MM_TO_PX)
SIGN_PX    = int(SIGN_DIAM_MM * MM_TO_PX)
PATCH_PX   = int(PATCH_MM    * MM_TO_PX)   # square side in pixels
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


def prepare_square_patch(patch_img: Image.Image, size_px: int) -> Image.Image:
    """
    Resize patch to a square at the correct physical print size.
    The attack was optimised as a square patch — composite as square, NOT circle.
    Applying a circular mask here would clip pixels that the model was trained on,
    producing a different effective patch than what was optimised.
    """
    return patch_img.convert("RGBA").resize((size_px, size_px), Image.LANCZOS)


def make_printable(patch_path: str, out_path: str,
                   patch_x_mm: float, patch_y_mm: float) -> Image.Image:
    """Returns the canvas so --test can crop and run the model over it."""
    print(f"A4        : {A4_W_PX}×{A4_H_PX} px  ({A4_W_MM}×{A4_H_MM} mm @ {DPI} DPI)")
    print(f"Plate     : {PLATE_W_PX}×{PLATE_H_PX} px  ({PLATE_W_MM:.0f}×{PLATE_H_MM:.0f} mm)")
    print(f"Sign      : {SIGN_PX} px  ({SIGN_DIAM_MM} mm)")
    print(f"Patch     : {PATCH_PX}×{PATCH_PX} px  ({PATCH_MM:.1f}×{PATCH_MM:.1f} mm)  [SQUARE]")
    print(f"Patch pos : ({patch_x_mm:+.1f} mm, {patch_y_mm:+.1f} mm) from sign centre")
    print(f"Scale     : REAL_SIGN_MM={REAL_SIGN_MM}mm, REAL_PATCH_MM={REAL_PATCH_MM}mm")

    canvas = Image.new("RGB", (A4_W_PX, A4_H_PX), (255, 255, 255))

    # Centre plate on page
    plate = render_sign_on_plate(SIGN_PX, PLATE_W_PX, PLATE_H_PX, PLATE_R_PX)
    plate_x = (A4_W_PX - PLATE_W_PX) // 2
    plate_y = (A4_H_PX - PLATE_H_PX) // 2
    canvas.paste(plate, (plate_x, plate_y), mask=plate.split()[3])

    # Sign centre in canvas coords
    scx = plate_x + PLATE_W_PX // 2
    scy = plate_y + PLATE_H_PX // 2

    # Composite patch as square (matches what the attack was optimised on)
    patch_sq = prepare_square_patch(Image.open(patch_path), PATCH_PX)
    px_left = int(scx + patch_x_mm * MM_TO_PX) - PATCH_PX // 2
    py_top  = int(scy + patch_y_mm * MM_TO_PX) - PATCH_PX // 2
    canvas.paste(patch_sq, (px_left, py_top), mask=patch_sq.split()[3])

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


def test_surrogate(canvas: Image.Image, patch_path: str,
                   surrogate_paths: dict | None = None,
                   n_runs: int = 20):
    """
    Crop the sign region from the canvas, apply the patch via apply_patch()
    at the correct 75px footprint, run EOT, and report predictions.
    Tests all three surrogate models (R18, MBV3-Small, EB0).

    Uses apply_patch() + eot_batch() — the same pipeline as optimisation —
    so the patch footprint is identical to what was trained (75px in 224px input).
    """
    import torch
    import torchvision.transforms as T
    from collections import Counter

    if surrogate_paths is None:
        surrogate_paths = {
            "resnet18":           ("surrogate.pt",                    "resnet18"),
            "mobilenet_v3_small": ("surrogate_mobilenet_v3_small.pt", "mobilenet_v3_small"),
            "efficientnet_b0":    ("surrogate_efficientnet_b0.pt",    "efficientnet_b0"),
        }

    try:
        from dataset import ALL_SPEEDS, KMH_TO_LABEL, val_transforms
        from model import load as load_model, get_device
        from eot import eot_batch
        from patch_attack import apply_patch
    except ImportError as e:
        print(f"Cannot import project modules: {e}")
        print("Run this from your project directory.")
        return

    device = get_device()

    # Load all surrogate models with correct arch
    models = {}
    for name, (path, arch) in surrogate_paths.items():
        try:
            m = load_model(path, arch=arch).to(device)
            m.eval()
            models[name] = m
            print(f"Loaded {name} from {path}")
        except Exception as e:
            print(f"Warning: could not load {name} ({path}): {e}")

    if not models:
        print("No models loaded — aborting test.")
        return

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Load and normalise the patch tensor (same as eval_patch.py)
    import numpy as np
    arr = np.array(Image.open(patch_path).convert("RGB")).astype(np.float32) / 255.0
    patch_01   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    patch_norm = (patch_01 - mean) / std

    # target_patch_px: must match what was used during optimisation.
    # Default: 75px — computed as max(4, int(int(224*0.80) * 80 / 190))
    # where 80mm is the real patch size and 190mm is the sign rendered diameter.
    target_patch_px = max(4, int(int(224 * 0.80) * REAL_PATCH_MM / SIGN_DIAM_MM))
    print(f"Patch footprint in model input: {target_patch_px}px  "
          f"(REAL_PATCH_MM={REAL_PATCH_MM}, SIGN_DIAM_MM={SIGN_DIAM_MM})")

    # Render a CLEAN sign (no patch) so apply_patch() starts from a blank slate.
    # If we crop from canvas, the patch is already baked in at a fixed position
    # and apply_patch() would double-stamp it — randomise_placement would do nothing.
    clean_plate = render_sign_on_plate(SIGN_PX, PLATE_W_PX, PLATE_H_PX, PLATE_R_PX)
    clean_canvas = Image.new("RGB", (PLATE_W_PX, PLATE_H_PX), (255, 255, 255))
    clean_canvas.paste(clean_plate, (0, 0), mask=clean_plate.split()[3])

    # Crop with same margin as round_trip_test / eval_patch
    margin = int(SIGN_PX * 0.12)
    half   = SIGN_PX // 2 + margin
    cx, cy = PLATE_W_PX // 2, PLATE_H_PX // 2
    clean_crop = clean_canvas.crop((cx - half, cy - half, cx + half, cy + half))
    clean_crop.save("test_crop.png")

    # val_transforms: resize to 224×224, ToTensor, Normalize
    sign_tensor = val_transforms(clean_crop.convert("RGB")).unsqueeze(0).to(device)

    target_pred = KMH_TO_LABEL[80]
    preds_per_model = {name: [] for name in models}

    with torch.no_grad():
        for _ in range(n_runs):
            # Each run: place patch at a random position on the clean sign, then EOT
            patched     = apply_patch(sign_tensor, patch_norm,
                                      randomise_placement=True,
                                      target_patch_px=target_patch_px)
            patched_01  = patched * std + mean
            patched_01  = eot_batch(patched_01.clone())
            patched_eot = (patched_01 - mean) / std

            for name, m in models.items():
                preds_per_model[name].append(m(patched_eot).argmax(1).item())

    print(f"\n── Surrogate predictions (random placement + EOT, {n_runs} runs) ──")
    print(f"  {'Model':<25}  {'Top pred':>9}  {'Conf':>6}  {'ASR@80':>7}")
    print("  " + "─" * 54)
    worst_asr = 1.0
    for name, preds in preds_per_model.items():
        counts  = Counter(preds)
        best    = counts.most_common(1)[0][0]
        conf    = counts[best] / n_runs
        asr80   = counts[target_pred] / n_runs
        hit     = " ✓" if best == target_pred else ""
        worst_asr = min(worst_asr, asr80)
        print(f"  {name:<25}  {ALL_SPEEDS[best]:>5} km/h  {conf:>5.1%}  {asr80:>6.1%}{hit}")
    print(f"  {'Ensemble worst-case ASR@80':<25}  {'':>9}  {'':>6}  {worst_asr:>6.1%}")
    print(f"\ntest_crop.png shows the clean sign the patch is applied onto each run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch",   required=True)
    p.add_argument("--out",     default="printable_sign.png")
    p.add_argument("--patch-x", type=float, default=0.0,
                   help="Patch X offset from sign centre in mm (+ = right). Default 0 = centred.")
    p.add_argument("--patch-y", type=float, default=0.0,
                   help="Patch Y offset from sign centre in mm (+ = down). Default 0 = centred.")
    p.add_argument("--test",    action="store_true",
                   help="Run surrogate ensemble over the rendered printable")
    p.add_argument("--n-runs",  type=int, default=20)
    args = p.parse_args()

    canvas = make_printable(args.patch, args.out, args.patch_x, args.patch_y)
    if args.test:
        test_surrogate(canvas, args.patch, n_runs=args.n_runs)
