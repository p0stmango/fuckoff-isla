"""
Synthetic Australian speed-limit sign generator.

Australian sign spec:
  - Circular, white face, red border ring
  - Black speed number using Transport / Highway Gothic style typeface
  - Mounted on a circular aluminium blank — no rectangular backing

Generates PNG images composited onto synthetic backgrounds representing
typical Australian environments: overcast sky, concrete/bitumen, vegetation,
night carpark.  This diversity is critical for physical transfer since
the EyeQ4 camera crops the sign from a wider scene.

Usage (standalone):
    python aus_signs.py --out-dir ./data/aus_synth --n 400
    python aus_signs.py --out-dir ./data/aus_synth --n 400 --preview
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Australian speed limits ──────────────────────────────────────────────────
AU_SPEEDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110]

# ── sign colours ─────────────────────────────────────────────────────────────
_WHITE  = (255, 255, 255)
_RED    = (206, 17,  38)    # AS/NZS red — slightly darker than pure red
_BLACK  = (10,  10,  10)    # near-black for ink variation


# ── font resolution ──────────────────────────────────────────────────────────
_FONT_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── perspective warp ──────────────────────────────────────────────────────────

def _perspective_warp(img: Image.Image, strength: float = 0.15) -> Image.Image:
    """
    Apply a random perspective warp simulating viewing angle.
    strength: max fractional shift of corners (0.15 = ±15% of image size).
    Uses PIL's transform with PERSPECTIVE coefficients.
    """
    w, h = img.size
    # Original corners: TL, TR, BR, BL
    orig = [0, 0,  w, 0,  w, h,  0, h]

    def jitter(v, axis_size):
        return v + random.uniform(-strength, strength) * axis_size

    # Randomise each corner independently but keep it a valid quadrilateral
    # by constraining the shift so corners don't cross
    s = strength * 0.7   # slightly tighter to keep sign readable
    new_corners = [
        jitter(0, w), jitter(0, h),   # TL
        jitter(w, w), jitter(0, h),   # TR
        jitter(w, w), jitter(h, h),   # BR
        jitter(0, w), jitter(h, h),   # BL
    ]

    # Compute perspective coefficients (8-point, PIL convention)
    # Solving: dst = M * src  →  find M
    def _find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array([p[i] for p in pb for i in range(2)], dtype=np.float64)
        res = np.linalg.solve(A, B)
        return np.array(res).flatten()

    src = [(orig[i], orig[i+1]) for i in range(0, 8, 2)]
    dst = [(new_corners[i], new_corners[i+1]) for i in range(0, 8, 2)]

    try:
        coeffs = _find_coeffs(dst, src)   # PIL warps destination→source
        return img.transform(
            (w, h), Image.PERSPECTIVE, coeffs,
            resample=Image.BILINEAR,
        )
    except np.linalg.LinAlgError:
        return img   # degenerate — skip warp


# ── sign renderer ─────────────────────────────────────────────────────────────

def render_au_sign(
    speed: int,
    canvas_size: int = 224,
    sign_fraction: float = 0.80,
    aging: float = 0.0,
) -> Image.Image:
    """
    Render one Australian speed-limit sign on a transparent background.
    Returns RGBA PIL image of size (canvas_size × canvas_size).
    """
    size = canvas_size
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = cy = size / 2
    r  = (size * sign_fraction) / 2
    ring_w = max(3, int(size * 0.07))

    # White disc
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_WHITE + (255,))
    # Red ring
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=_RED + (255,), width=ring_w)

    # Number
    label   = str(speed)
    n_chars = len(label)
    if n_chars == 1:
        num_frac = 0.52
    elif n_chars == 2:
        num_frac = 0.46
    else:
        num_frac = 0.34

    num_font = _load_font(int(size * num_frac))
    bbox = draw.textbbox((0, 0), label, font=num_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        (cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]),
        label, fill=_BLACK + (255,), font=num_font,
    )

    # Aging / weathering
    if aging > 0:
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + aging * 20, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - aging * 15, 0, 255)
        n_scratches = int(aging * 8)
        for _ in range(n_scratches):
            x0 = random.randint(0, size)
            y0 = random.randint(0, size)
            x1 = x0 + random.randint(-size // 4, size // 4)
            y1 = y0 + random.randint(-size // 4, size // 4)
            scratch_img = Image.fromarray(arr.astype(np.uint8), "RGBA")
            sd = ImageDraw.Draw(scratch_img)
            sd.line([x0, y0, x1, y1], fill=(80, 80, 80, 200), width=1)
            arr = np.array(scratch_img, dtype=np.float32)
        grey = arr[:, :, :3].mean(axis=2, keepdims=True)
        arr[:, :, :3] = arr[:, :, :3] * (1 - aging * 0.2) + grey * (aging * 0.2)
        img = Image.fromarray(arr.astype(np.uint8), "RGBA")

    return img


# ── background generators ─────────────────────────────────────────────────────

def _bg_sky(size: int) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        t = y / size
        r = int(150 + 60 * t + random.uniform(-5, 5))
        g = int(160 + 50 * t + random.uniform(-5, 5))
        b = int(185 + 40 * t + random.uniform(-5, 5))
        arr[y, :] = [r, g, b]
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise - 10, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_concrete(size: int) -> Image.Image:
    base  = random.randint(80, 170)   # wider range than before
    noise = np.random.normal(0, 15, (size, size, 3))
    arr   = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_vegetation(size: int) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        t = y / size
        arr[y, :] = [int(50 + 40*t), int(90 + 60*t), int(30 + 20*t)]
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    arr   = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_night(size: int) -> Image.Image:
    arr = np.random.randint(5, 30, (size, size, 3), dtype=np.uint8)
    cx = cy = size / 2
    sigma = size * 0.4
    for y in range(size):
        for x in range(size):
            d2 = ((x - cx)**2 + (y - cy)**2) / (sigma**2)
            glow = int(120 * math.exp(-d2))
            arr[y, x] = np.clip(
                arr[y, x] + np.array([glow, int(glow*0.85), int(glow*0.6)]),
                0, 255,
            )
    return Image.fromarray(arr, "RGB")


def _bg_white(size: int) -> Image.Image:
    val = random.randint(200, 255)
    arr = np.full((size, size, 3), val, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_wall(size: int) -> Image.Image:
    """Concrete/brick wall — common carpark background."""
    base  = random.randint(120, 200)
    tint  = random.choice([(1.0, 0.95, 0.90), (0.95, 0.95, 1.0), (1.0, 1.0, 1.0)])
    noise = np.random.normal(0, 18, (size, size, 3))
    arr   = np.clip(base + noise, 0, 255).astype(np.uint8)
    arr   = (arr * np.array(tint)).clip(0, 255).astype(np.uint8)
    # Horizontal mortar lines
    for y in range(0, size, random.randint(18, 32)):
        arr[max(0, y-1):y+1, :] = np.clip(arr[max(0, y-1):y+1, :].astype(int) - 30, 0, 255)
    return Image.fromarray(arr, "RGB")


_BG_GENERATORS = [_bg_sky, _bg_concrete, _bg_vegetation, _bg_night, _bg_white, _bg_wall]
_BG_WEIGHTS    = [0.15,    0.30,          0.10,           0.20,      0.10,      0.15]


def random_background(size: int) -> Image.Image:
    fn = random.choices(_BG_GENERATORS, weights=_BG_WEIGHTS, k=1)[0]
    return fn(size)


# ── compositing ───────────────────────────────────────────────────────────────

def composite_sign(
    speed:       int,
    canvas_size: int   = 224,
    jitter_pos:  bool  = True,
    aging:       float = None,
) -> Image.Image:
    """
    Return a single RGB image of a speed sign composited onto a random background.

    Randomises:
      - background type and texture
      - sign position (±12% of canvas)
      - sign scale (±20%)          ← wider than before
      - sign rotation (±10°)       ← wider than before
      - perspective warp (±12%)    ← NEW
      - aging level
      - motion blur                ← NEW
      - brightness/contrast        ← NEW wider range
      - JPEG-style compression     ← NEW
    """
    if aging is None:
        aging = random.uniform(0.0, 0.5)

    bg = random_background(canvas_size)

    # Scale: ±20% (was ±5%)
    scale  = random.uniform(0.80, 1.20)
    s_size = max(32, int(canvas_size * scale))
    sign_rgba = render_au_sign(speed, canvas_size=s_size, aging=aging)

    # Rotation: ±10° (was ±6°)
    angle = random.uniform(-10, 10)
    sign_rgba = sign_rgba.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Perspective warp — applied to sign before compositing
    if random.random() < 0.6:   # 60% of images get perspective warp
        warp_strength = random.uniform(0.05, 0.15)
        sign_rgba = _perspective_warp(sign_rgba, strength=warp_strength)

    # Position jitter: ±12% (was ±8%)
    if jitter_pos:
        max_off = int(canvas_size * 0.12)
        ox = random.randint(-max_off, max_off)
        oy = random.randint(-max_off, max_off)
    else:
        ox = oy = 0

    paste_x = (canvas_size - s_size) // 2 + ox
    paste_y = (canvas_size - s_size) // 2 + oy

    bg.paste(sign_rgba, (paste_x, paste_y), mask=sign_rgba.split()[3])

    # Defocus blur (wider range)
    if random.random() < 0.35:
        radius = random.uniform(0.3, 2.0)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))

    # Motion blur — horizontal smear simulating car movement
    if random.random() < 0.25:
        blur_px = random.randint(2, 6)
        kernel  = np.zeros((blur_px, blur_px))
        kernel[blur_px // 2, :] = 1.0 / blur_px
        from PIL import ImageFilter as _IF
        bg = bg.filter(_IF.Kernel(
            size=(blur_px, blur_px),
            kernel=kernel.flatten().tolist(),
            scale=1, offset=0,
        )) if blur_px <= 5 else bg.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Brightness / contrast variation (wider than default)
    arr = np.array(bg, dtype=np.float32)
    brightness = random.uniform(0.55, 1.45)   # was implicitly ~1.0
    contrast   = random.uniform(0.75, 1.35)
    mean       = arr.mean()
    arr        = (arr - mean) * contrast + mean * brightness
    arr        = np.clip(arr, 0, 255).astype(np.uint8)
    bg         = Image.fromarray(arr, "RGB")

    # JPEG compression artefacts (simulate camera encoding)
    if random.random() < 0.30:
        import io
        buf = io.BytesIO()
        bg.save(buf, format="JPEG", quality=random.randint(55, 85))
        buf.seek(0)
        bg = Image.open(buf).copy()

    return bg


# ── dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    out_dir: str,
    n_per_class: int = 400,
    canvas_size: int = 224,
    speeds: list = None,
):
    if speeds is None:
        speeds = AU_SPEEDS

    out = Path(out_dir)
    total = 0

    for speed in speeds:
        class_dir = out / str(speed)
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(class_dir.glob("*.png")))
        needed   = n_per_class - existing
        if needed <= 0:
            print(f"  {speed:3d} km/h — already have {existing} images, skipping")
            continue

        print(f"  {speed:3d} km/h — generating {needed} images", end="", flush=True)
        for i in range(needed):
            img = composite_sign(speed, canvas_size=canvas_size)
            img.save(class_dir / f"{existing + i:05d}.png")
            if (i + 1) % 50 == 0:
                print(".", end="", flush=True)
        print(f" done ({n_per_class} total)")
        total += needed

    print(f"\nGenerated {total} new images across {len(speeds)} classes → {out_dir}")


# ── preview helper ────────────────────────────────────────────────────────────

def save_preview(out_path: str = "aus_preview.png", n_cols: int = 7):
    speeds = AU_SPEEDS
    n_rows = math.ceil(len(speeds) / n_cols)
    cell   = 224
    pad    = 4

    canvas = Image.new("RGB", (n_cols * (cell + pad) + pad, n_rows * (cell + pad) + pad),
                       (40, 40, 40))

    for i, speed in enumerate(speeds):
        row = i // n_cols
        col = i % n_cols
        img = composite_sign(speed, canvas_size=cell)
        canvas.paste(img, (pad + col * (cell + pad), pad + row * (cell + pad)))

    canvas.save(out_path)
    print(f"Preview saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic Australian speed-limit sign images")
    p.add_argument("--out-dir",  default="./data/aus_synth", help="Output root directory")
    p.add_argument("--n",        type=int, default=400,      help="Images per class")
    p.add_argument("--size",     type=int, default=224,      help="Canvas size (px)")
    p.add_argument("--preview",  action="store_true",        help="Also save aus_preview.png")
    p.add_argument("--speeds",   nargs="+", type=int,        help="Override speed list")
    args = p.parse_args()

    speeds = args.speeds if args.speeds else AU_SPEEDS
    print(f"Generating {args.n} images × {len(speeds)} classes → {args.out_dir}")
    generate_dataset(args.out_dir, n_per_class=args.n, canvas_size=args.size, speeds=speeds)

    if args.preview:
        save_preview()
