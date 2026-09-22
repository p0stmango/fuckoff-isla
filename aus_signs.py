"""
Synthetic Australian speed-limit sign generator.

Australian sign spec:
  - Circular, white face, red border ring
  - Black speed number using Transport / Highway Gothic style typeface
  - Small "km/h" text below the number (standard since AS 1742.2)
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
# AU signs use Transport / Highway Gothic — a regular-weight condensed sans.
# Priority: regular-weight faces only, no bold variants.
_FONT_PATHS = [
    # macOS — Helvetica Neue Regular is the closest readily available match
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # Linux — regular weight
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


# ── sign renderer ─────────────────────────────────────────────────────────────

def render_au_sign(
    speed: int,
    canvas_size: int = 224,
    sign_fraction: float = 0.80,
    aging: float = 0.0,          # 0.0 = pristine, 1.0 = heavily weathered
) -> Image.Image:
    """
    Render one Australian speed-limit sign on a transparent background.
    Returns RGBA PIL image of size (canvas_size × canvas_size).

    AU signs: white circle, red border ring, regular-weight black number only.
    No "km/h" text — that is a European convention, not used in Australia.

    aging: adds yellowing, scratches, fading to simulate real-world wear.
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

    # ── number — centred, regular weight ──
    label   = str(speed)
    n_chars = len(label)

    if n_chars == 1:
        num_frac = 0.52
    elif n_chars == 2:
        num_frac = 0.46
    else:   # 3 digits (110)
        num_frac = 0.34

    num_font = _load_font(int(size * num_frac))
    bbox = draw.textbbox((0, 0), label, font=num_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        (cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]),
        label, fill=_BLACK + (255,), font=num_font,
    )

    # ── aging / weathering ──
    if aging > 0:
        arr = np.array(img, dtype=np.float32)
        # yellowing: push towards warm tint
        arr[:, :, 0] = np.clip(arr[:, :, 0] + aging * 20, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - aging * 15, 0, 255)
        # random scratches (thin dark lines)
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
        # fading: reduce saturation
        grey = arr[:, :, :3].mean(axis=2, keepdims=True)
        arr[:, :, :3] = arr[:, :, :3] * (1 - aging * 0.2) + grey * (aging * 0.2)
        img = Image.fromarray(arr.astype(np.uint8), "RGBA")

    return img


# ── background generators ─────────────────────────────────────────────────────

def _bg_sky(size: int) -> Image.Image:
    """Blue-grey overcast sky typical of Australian weather."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        t = y / size
        r = int(150 + 60 * t + random.uniform(-5, 5))
        g = int(160 + 50 * t + random.uniform(-5, 5))
        b = int(185 + 40 * t + random.uniform(-5, 5))
        arr[y, :] = [r, g, b]
    # cloud noise
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise - 10, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_concrete(size: int) -> Image.Image:
    """Grey concrete / bitumen — carpark floor colour."""
    base  = random.randint(100, 160)
    noise = np.random.normal(0, 12, (size, size, 3))
    arr   = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_vegetation(size: int) -> Image.Image:
    """Green-brown vegetation blob — typical roadside."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        t = y / size
        arr[y, :] = [
            int(50  + 40  * t),
            int(90  + 60  * t),
            int(30  + 20  * t),
        ]
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    arr   = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_night(size: int) -> Image.Image:
    """Dark night with carpark-style artificial lighting blob."""
    arr = np.random.randint(5, 30, (size, size, 3), dtype=np.uint8)
    # central warm light blob
    cx = cy = size / 2
    sigma = size * 0.4
    for y in range(size):
        for x in range(size):
            d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (sigma ** 2)
            glow = int(120 * math.exp(-d2))
            arr[y, x] = np.clip(
                arr[y, x] + np.array([glow, int(glow * 0.85), int(glow * 0.6)]),
                0, 255,
            )
    return Image.fromarray(arr, "RGB")


def _bg_white(size: int) -> Image.Image:
    """Clean white — for dataset variety / ablation."""
    val = random.randint(220, 255)
    arr = np.full((size, size, 3), val, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


_BG_GENERATORS = [_bg_sky, _bg_concrete, _bg_vegetation, _bg_night, _bg_white]

_BG_WEIGHTS = [0.20, 0.35, 0.15, 0.20, 0.10]   # carpark-heavy: more concrete/night


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
      - sign position (±8% of canvas)
      - sign scale (±5%)
      - sign rotation (±6°)
      - aging level
    """
    if aging is None:
        aging = random.uniform(0.0, 0.4)

    bg = random_background(canvas_size)

    scale  = random.uniform(0.90, 1.05)
    s_size = int(canvas_size * scale)
    sign_rgba = render_au_sign(
        speed, canvas_size=s_size, aging=aging,
    )

    # rotate
    angle = random.uniform(-6, 6)
    sign_rgba = sign_rgba.rotate(angle, resample=Image.BILINEAR, expand=False)

    # position jitter
    if jitter_pos:
        max_off = int(canvas_size * 0.08)
        ox = random.randint(-max_off, max_off)
        oy = random.randint(-max_off, max_off)
    else:
        ox = oy = 0

    paste_x = (canvas_size - s_size) // 2 + ox
    paste_y = (canvas_size - s_size) // 2 + oy

    bg.paste(sign_rgba, (paste_x, paste_y), mask=sign_rgba.split()[3])

    # slight overall blur (simulate camera defocus at distance)
    if random.random() < 0.3:
        radius = random.uniform(0.3, 1.2)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))

    return bg


# ── dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    out_dir: str,
    n_per_class: int = 400,
    canvas_size: int = 224,
    speeds: list[int] = None,
):
    """
    Write n_per_class images per speed to out_dir/<speed>/XXXXX.png.
    Directory structure mirrors torchvision ImageFolder convention.
    """
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
    """Save a grid showing all speed classes with various backgrounds."""
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
