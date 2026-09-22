"""
Visualise the optimised patch applied to sample signs.
Saves a grid PNG showing clean vs patched images for each speed class.

Usage:
    python visualise.py --patch patch.pt --model surrogate.pt --data ./data
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from dataset import FilteredGTSRB, val_transforms, ALL_SPEEDS, KMH_TO_LABEL
from model import build_surrogate, get_device, load as load_model
from patch_attack import apply_patch, make_circular_mask

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def denorm(t):
    return (t.cpu() * _STD + _MEAN).clamp(0, 1)


def make_grid(args):
    device = get_device()

    model = load_model(args.model).to(device) if Path(args.model).exists() \
            else build_surrogate().to(device)
    model.eval()

    val_ds = FilteredGTSRB(args.data, split="test", transform=val_transforms, download=False)

    patch_01 = torch.load(args.patch, map_location="cpu", weights_only=True)
    patch_size = patch_01.shape[-1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_norm = ((patch_01.to(device) - mean) / std)
    mask = make_circular_mask(patch_size, device)

    # collect one sample per class
    class_samples = {}
    for img, label in val_ds:
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) == len(ALL_SPEEDS):
            break

    # build grid: rows=classes, cols=[clean, patched+label]
    cell = 224
    pad  = 4
    n    = len(class_samples)
    W    = (cell + pad) * 2 + pad
    H    = (cell + pad) * n + pad
    canvas = Image.new("RGB", (W, H), (30, 30, 30))

    for row, (label, img_t) in enumerate(sorted(class_samples.items())):
        img_batch = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            patched   = apply_patch(img_batch, patch_norm, mask)
            clean_pred   = model(img_batch).argmax(1).item()
            patched_pred = model(patched).argmax(1).item()

        clean_pil   = tensor_to_pil(denorm(img_batch))
        patched_pil = tensor_to_pil(denorm(patched))

        # annotate
        for pil, pred, col in [(clean_pil, clean_pred, (80, 200, 80)),
                                (patched_pil, patched_pred, (220, 80, 80))]:
            d = ImageDraw.Draw(pil)
            txt = f"{ALL_SPEEDS[pred]} km/h"
            d.rectangle([0, cell - 22, cell, cell], fill=(0, 0, 0, 180))
            d.text((4, cell - 18), txt, fill=(255, 255, 255))

        y = pad + row * (cell + pad)
        canvas.paste(clean_pil,   (pad,              y))
        canvas.paste(patched_pil, (pad + cell + pad, y))

    canvas.save(args.out)
    print(f"Grid saved: {args.out}")
    print("Left column = clean, right column = patched")
    print(f"Classes (top→bottom): {[ALL_SPEEDS[l] for l in sorted(class_samples)]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch", default="patch.pt")
    p.add_argument("--model", default="surrogate.pt")
    p.add_argument("--data",  default="./data")
    p.add_argument("--out",   default="patch_grid.png")
    make_grid(p.parse_args())
