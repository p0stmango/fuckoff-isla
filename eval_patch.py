"""
Overlay the optimised patch on a 5 km/h sign, apply EOT, and report per-model
predictions across N EOT samples.

Usage:
    # Use a random 5 km/h sign from the AU synth val set
    python eval_patch.py

    # Use a specific sign image (e.g. a photo you took)
    python eval_patch.py --sign path/to/sign.jpg --true-speed 5

    # Use an intermediate patch checkpoint
    python eval_patch.py --patch patch_step_0500.png

    # More EOT samples for a stable vote
    python eval_patch.py --n-eot 64
"""
import argparse
import random

import numpy as np
import torch
from PIL import Image

from dataset import AUSynthDataset, KMH_TO_LABEL, ALL_SPEEDS, val_transforms
from eot import eot_batch
from model import get_device, load as load_model
from patch_attack import apply_patch

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main(args):
    device = get_device()
    mean = _MEAN.to(device)
    std  = _STD.to(device)

    # ── load models ──────────────────────────────────────────────────────────
    models = {
        "resnet18":           load_model("surrogate.pt",                    arch="resnet18").to(device),
        "mobilenet_v3_small": load_model("surrogate_mobilenet_v3_small.pt", arch="mobilenet_v3_small").to(device),
        "efficientnet_b0":    load_model("surrogate_efficientnet_b0.pt",    arch="efficientnet_b0").to(device),
    }
    for m in models.values():
        m.eval()

    # ── load patch ───────────────────────────────────────────────────────────
    if args.patch.endswith(".pt"):
        patch_01 = torch.load(args.patch, map_location="cpu", weights_only=True)
    else:
        arr = np.array(Image.open(args.patch).convert("RGB")).astype(np.float32) / 255.0
        patch_01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    patch_01 = patch_01.to(device)  # (1, 3, P, P)
    print(f"Patch loaded : {args.patch}  [{patch_01.shape[-1]}px]")

    # ── compute model-input footprint (same formula as optimise_patch) ───────
    target_patch_px = max(4, int(int(224 * 0.80) * (args.print_cm * 10) / args.sign_diam_mm))
    print(f"Patch footprint in model input : {target_patch_px}px")

    # ── get a sign image ─────────────────────────────────────────────────────
    true_label = KMH_TO_LABEL[args.true_speed]
    if args.sign:
        sign_img = val_transforms(Image.open(args.sign).convert("RGB")).unsqueeze(0).to(device)
        print(f"Sign image   : {args.sign}")
    else:
        # Sample from AU synth dataset
        for split in ("val", "train"):
            ds = AUSynthDataset(args.aus_data, transform=val_transforms,
                                split=split, auto_generate=False)
            candidates = [img for img, lbl in ds if lbl == true_label]
            if candidates:
                break
        if not candidates:
            raise RuntimeError(
                f"No {args.true_speed} km/h samples found in {args.aus_data}. "
                "Pass --sign path/to/sign.jpg instead."
            )
        sign_img = random.choice(candidates).unsqueeze(0).to(device)
        print(f"Sign image   : random {args.true_speed} km/h sample from AU synth ({split} split)")

    print(f"True class   : {ALL_SPEEDS[true_label]} km/h  (label {true_label})")

    # ── baseline (no patch) ──────────────────────────────────────────────────
    with torch.no_grad():
        print(f"\n── Baseline (no patch) ──")
        for name, m in models.items():
            pred = m(sign_img).argmax(1).item()
            print(f"  {name:<25}  {ALL_SPEEDS[pred]:>5} km/h")

    # ── patched + EOT vote ───────────────────────────────────────────────────
    patch_norm = (patch_01 - mean) / std
    votes = {name: [0] * len(ALL_SPEEDS) for name in models}

    with torch.no_grad():
        for _ in range(args.n_eot):
            patched    = apply_patch(sign_img, patch_norm,
                                     randomise_placement=False,
                                     target_patch_px=target_patch_px)
            patched_01 = patched * std + mean
            patched_01 = eot_batch(patched_01.clone())
            patched_eot = (patched_01 - mean) / std

            for name, m in models.items():
                votes[name][m(patched_eot).argmax(1).item()] += 1

    target_pred = KMH_TO_LABEL[80]
    print(f"\n── Patched + EOT ({args.n_eot} samples) ──")
    print(f"{'Model':<25}  {'Predicted':>9}  {'Conf':>6}  {'ASR@80':>6}")
    print("─" * 55)
    for name, v in votes.items():
        best  = max(range(len(v)), key=lambda i: v[i])
        conf  = v[best] / args.n_eot
        asr80 = v[target_pred] / args.n_eot
        hit   = " ✓" if best == target_pred else ""
        print(f"  {name:<25}  {ALL_SPEEDS[best]:>5} km/h  {conf:>5.1%}  {asr80:>5.1%}{hit}")

    worst_asr = min(v[target_pred] / args.n_eot for v in votes.values())
    print(f"\n  Ensemble worst-case ASR@80 : {worst_asr:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch",         default="patch.pt",
                   help="Patch file (.pt tensor or .png image)")
    p.add_argument("--sign",          default=None,
                   help="Path to sign image. Omit to sample from AU synth dataset.")
    p.add_argument("--true-speed",    type=int,   default=5,
                   help="True speed class of the sign (km/h)")
    p.add_argument("--aus-data",      default="./data/aus_synth")
    p.add_argument("--n-eot",         type=int,   default=32,
                   help="Number of EOT samples for the vote")
    p.add_argument("--print-cm",      type=float, default=8.0)
    p.add_argument("--sign-diam-mm",  type=float, default=190.0)
    main(p.parse_args())
