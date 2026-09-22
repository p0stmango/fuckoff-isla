"""
Round-trip robustness test for patch.png before printing.

Simulates the full physical chain:
  patch.pt (75px optimised)
    → LANCZOS upsample → patch.png (944px, print-ready)
    → [print → mount → camera capture]
    → bilinear downsample → 75px in model input
    → EOT → predict

Tests multiple downscale methods and resolutions to check how brittle
the adversarial features are to the upscale/downsample cycle.

Usage:
    python round_trip_test.py                  # default: patch.png
    python round_trip_test.py --patch patch.png --n-eot 64
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import AUSynthDataset, KMH_TO_LABEL, ALL_SPEEDS, val_transforms
from eot import eot_batch
from model import get_device, load as load_model
from patch_attack import apply_patch

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_models(device):
    models = {
        "resnet18":           load_model("surrogate.pt",                    arch="resnet18").to(device),
        "mobilenet_v3_small": load_model("surrogate_mobilenet_v3_small.pt", arch="mobilenet_v3_small").to(device),
        "efficientnet_b0":    load_model("surrogate_efficientnet_b0.pt",    arch="efficientnet_b0").to(device),
    }
    for m in models.values():
        m.eval()
    return models


def vote(models, sign_img, patch_norm, target_patch_px, n_eot, device):
    mean = _MEAN.to(device)
    std  = _STD.to(device)
    target_pred = KMH_TO_LABEL[80]
    votes = {name: [0] * len(ALL_SPEEDS) for name in models}

    with torch.no_grad():
        for _ in range(n_eot):
            patched     = apply_patch(sign_img, patch_norm,
                                      randomise_placement=False,
                                      target_patch_px=target_patch_px)
            patched_01  = patched * std + mean
            patched_01  = eot_batch(patched_01.clone())
            patched_eot = (patched_01 - mean) / std
            for name, m in models.items():
                votes[name][m(patched_eot).argmax(1).item()] += 1

    worst = min(v[target_pred] / n_eot for v in votes.values())
    return votes, worst


def print_results(label, votes, n_eot):
    target_pred = KMH_TO_LABEL[80]
    print(f"\n── {label} ──")
    print(f"  {'Model':<25}  {'Predicted':>9}  {'Conf':>6}  {'ASR@80':>6}")
    print("  " + "─" * 52)
    for name, v in votes.items():
        best  = max(range(len(v)), key=lambda i: v[i])
        conf  = v[best] / n_eot
        asr80 = v[target_pred] / n_eot
        hit   = " ✓" if best == target_pred else ""
        print(f"  {name:<25}  {ALL_SPEEDS[best]:>5} km/h  {conf:>5.1%}  {asr80:>5.1%}{hit}")
    worst = min(v[target_pred] / n_eot for v in votes.values())
    print(f"  Worst-case ASR@80 : {worst:.1%}")
    return worst


def main(args):
    device = get_device()
    mean   = _MEAN.to(device)
    std    = _STD.to(device)

    models = load_models(device)

    # ── Load print-ready PNG ──────────────────────────────────────────────────
    arr      = np.array(Image.open(args.patch).convert("RGB")).astype(np.float32) / 255.0
    patch_hires = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"Loaded {args.patch}: {patch_hires.shape[-1]}px  (print resolution)")

    target_patch_px = args.footprint
    print(f"Target model footprint: {target_patch_px}px")

    # ── Get a sign image ──────────────────────────────────────────────────────
    ds = AUSynthDataset(args.aus_data, transform=val_transforms,
                        split="val", auto_generate=False)
    candidates = [img for img, lbl in ds if lbl == KMH_TO_LABEL[args.true_speed]]
    sign_img = random.choice(candidates).unsqueeze(0).to(device)

    # ── Baseline: optimised 75px patch (what the optimiser actually trained on) ─
    patch_75   = F.interpolate(patch_hires, size=(target_patch_px, target_patch_px),
                               mode="bilinear", align_corners=False)
    patch_norm = (patch_75 - mean) / std
    v, w = vote(models, sign_img, patch_norm, target_patch_px, args.n_eot, device)
    baseline = print_results(f"Baseline — optimised {target_patch_px}px (bilinear downsample from {patch_hires.shape[-1]}px)", v, args.n_eot)

    # ── Robustness: different downsample interpolations ───────────────────────
    for mode in ("nearest", "bicubic"):
        p = F.interpolate(patch_hires, size=(target_patch_px, target_patch_px),
                          mode=mode, align_corners=False if mode == "bicubic" else None)
        pn = (p - mean) / std
        v, w = vote(models, sign_img, pn, target_patch_px, args.n_eot, device)
        print_results(f"Downsample: {mode} → {target_patch_px}px", v, args.n_eot)

    # ── Robustness: simulate JPEG compression on the printed patch ────────────
    # Saves as JPEG at varying quality and reloads — models print→photo JPEG loss
    import io
    for quality in (90, 70, 50):
        buf = io.BytesIO()
        pil_img = Image.fromarray(
            (patch_hires.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_arr = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
        jpeg_t   = torch.from_numpy(jpeg_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        p  = F.interpolate(jpeg_t, size=(target_patch_px, target_patch_px),
                           mode="bilinear", align_corners=False)
        pn = (p - mean) / std
        v, w = vote(models, sign_img, pn, target_patch_px, args.n_eot, device)
        print_results(f"JPEG Q{quality} → bilinear {target_patch_px}px  (simulates lossy photo capture)", v, args.n_eot)

    # ── Robustness: slight gaussian blur before downsample (lens defocus) ─────
    import torchvision.transforms.functional as TF
    for blur_radius in (1, 2):
        pil_patch = Image.fromarray(
            (patch_hires.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        blurred = TF.gaussian_blur(pil_patch, kernel_size=blur_radius * 2 + 1, sigma=blur_radius)
        blur_t  = torch.from_numpy(np.array(blurred).astype(np.float32) / 255.0) \
                      .permute(2, 0, 1).unsqueeze(0).to(device)
        p  = F.interpolate(blur_t, size=(target_patch_px, target_patch_px),
                           mode="bilinear", align_corners=False)
        pn = (p - mean) / std
        v, w = vote(models, sign_img, pn, target_patch_px, args.n_eot, device)
        print_results(f"Gaussian blur σ={blur_radius} → {target_patch_px}px  (camera focus variation)", v, args.n_eot)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"Baseline (bilinear) ASR: {baseline:.1%}")
    print("If all variants stay above ~60%, the patch is robust to the")
    print("upscale/downsample cycle and safe to print.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch",       default="patch.png")
    p.add_argument("--footprint",   type=int,   default=75,
                   help="Patch footprint in model input (px) — must match optimisation")
    p.add_argument("--true-speed",  type=int,   default=5)
    p.add_argument("--aus-data",    default="./data/aus_synth")
    p.add_argument("--n-eot",       type=int,   default=32)
    main(p.parse_args())
