"""
Evaluate the adversarial patch as it appears in printable_sign.png.

Crops the sign region from the printable (patch already composited),
resizes to 224×224 via val_transforms, applies EOT, and reports per-model
predictions. This is the closest digital proxy to "print → photograph → classify".

Usage:
    python eval_printable.py                           # printable_sign.png, 32 EOT runs
    python eval_printable.py --printable my_sign.png   # different file
    python eval_printable.py --n-eot 64               # more samples for stable vote
"""
import argparse

import numpy as np
import torch
from PIL import Image

# ── physical constants (must match make_printable.py) ────────────────────────
DPI          = 300
A4_W_MM      = 210
A4_H_MM      = 297
SIGN_DIAM_MM = 190
PLATE_W_MM   = SIGN_DIAM_MM * 1.25
PLATE_H_MM   = SIGN_DIAM_MM * 1.25
MM_TO_PX     = DPI / 25.4
A4_W_PX      = int(A4_W_MM  * MM_TO_PX)
A4_H_PX      = int(A4_H_MM  * MM_TO_PX)
SIGN_PX      = int(SIGN_DIAM_MM * MM_TO_PX)
PLATE_W_PX   = int(PLATE_W_MM * MM_TO_PX)
PLATE_H_PX   = int(PLATE_H_MM * MM_TO_PX)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SURROGATE_CONFIGS = {
    "resnet18":           ("surrogate.pt",                    "resnet18"),
    "mobilenet_v3_small": ("surrogate_mobilenet_v3_small.pt", "mobilenet_v3_small"),
    "efficientnet_b0":    ("surrogate_efficientnet_b0.pt",    "efficientnet_b0"),
}


def crop_sign_from_printable(img: Image.Image) -> Image.Image:
    """
    Extract the sign region from a printable_sign.png canvas.
    The plate is centred on the A4 page; sign circle is centred on the plate.
    We crop with a 12% margin around the sign circle — same region the model sees.
    """
    w, h = img.size
    # If the image isn't A4@300DPI, scale the crop accordingly
    scale_x = w / A4_W_PX
    scale_y = h / A4_H_PX

    # Sign centre: middle of the plate, which is middle of the page
    cx = w / 2
    cy = h / 2

    sign_px_scaled = SIGN_PX * scale_x   # assume uniform scaling
    margin = int(sign_px_scaled * 0.12)
    half   = int(sign_px_scaled / 2) + margin

    left   = int(cx - half)
    top    = int(cy - half)
    right  = int(cx + half)
    bottom = int(cy + half)

    return img.crop((left, top, right, bottom))


def main(args):
    from dataset import ALL_SPEEDS, KMH_TO_LABEL, val_transforms
    from model import load as load_model, get_device
    from eot import eot_batch

    device = get_device()
    mean   = _MEAN.to(device)
    std    = _STD.to(device)

    # ── Load surrogate ensemble ───────────────────────────────────────────────
    models = {}
    for name, (path, arch) in SURROGATE_CONFIGS.items():
        try:
            m = load_model(path, arch=arch).to(device)
            m.eval()
            models[name] = m
            print(f"Loaded  {name:<25}  ({path})")
        except Exception as e:
            print(f"Warning: could not load {name}: {e}")

    if not models:
        raise RuntimeError("No models loaded.")

    # ── Load printable and crop sign region ───────────────────────────────────
    printable = Image.open(args.printable).convert("RGB")
    print(f"\nPrintable : {args.printable}  ({printable.size[0]}×{printable.size[1]} px)")

    crop = crop_sign_from_printable(printable)
    crop.save("eval_crop.png")
    print(f"Sign crop : {crop.size[0]}×{crop.size[1]} px  → saved to eval_crop.png")

    # val_transforms: resize to 224×224, ToTensor, Normalize
    sign_tensor = val_transforms(crop).unsqueeze(0).to(device)

    # ── Baseline: no EOT — raw model input ───────────────────────────────────
    print("\n── Baseline (no EOT, patch as printed) ──")
    with torch.no_grad():
        for name, m in models.items():
            pred = m(sign_tensor).argmax(1).item()
            print(f"  {name:<25}  {ALL_SPEEDS[pred]:>5} km/h")

    # ── EOT vote — simulates geometric/lighting/sensor variation ─────────────
    target_pred = KMH_TO_LABEL[80]
    votes = {name: [0] * len(ALL_SPEEDS) for name in models}

    with torch.no_grad():
        for i in range(args.n_eot):
            # EOT transforms the composited sign+patch tensor (no re-application of patch)
            t01     = sign_tensor * std + mean          # back to [0,1]
            t01     = eot_batch(t01.clone())            # augment
            t_eot   = (t01 - mean) / std               # re-normalise

            for name, m in models.items():
                votes[name][m(t_eot).argmax(1).item()] += 1

    print(f"\n── EOT vote ({args.n_eot} samples) — simulates print → photo pipeline ──")
    print(f"  {'Model':<25}  {'Top pred':>9}  {'Conf':>6}  {'ASR@80':>7}")
    print("  " + "─" * 54)
    worst_asr = 1.0
    for name, v in votes.items():
        best  = max(range(len(v)), key=lambda i: v[i])
        conf  = v[best] / args.n_eot
        asr80 = v[target_pred] / args.n_eot
        hit   = " ✓" if best == target_pred else ""
        worst_asr = min(worst_asr, asr80)
        print(f"  {name:<25}  {ALL_SPEEDS[best]:>5} km/h  {conf:>5.1%}  {asr80:>6.1%}{hit}")

    print(f"  {'Ensemble worst-case ASR@80':<25}  {'':>9}  {'':>6}  {worst_asr:>6.1%}")
    print(f"\nCheck eval_crop.png to confirm the patch is visible and centred in the crop.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--printable", default="printable_sign.png",
                   help="Output of make_printable.py")
    p.add_argument("--n-eot",     type=int, default=32,
                   help="Number of EOT samples for the vote")
    main(p.parse_args())
