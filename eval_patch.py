"""
Honest patch evaluation — run this on any patch_step_XXXX.png or patch_945.pt.

Usage:
    python3 eval_patch.py patch_step_0400.png
    python3 eval_patch.py patch_945.pt
    python3 eval_patch.py patch_step_0400.png --target 80
"""
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pathlib import Path

from dataset import ALL_SPEEDS, val_transforms, KMH_TO_LABEL, FilteredGTSRB, AUSynthDataset
from model import load as load_model, get_device, ARCH_CHOICES
from patch_attack import apply_patch
from eot import eot_batch
from torch.utils.data import ConcatDataset, DataLoader


def _load_surrogate(arch: str, model_path: str, device) -> torch.nn.Module:
    stem   = Path(model_path).stem
    suffix = Path(model_path).suffix
    arch_path = str(Path(model_path).parent / f"{stem}_{arch}{suffix}") \
                if arch != ARCH_CHOICES[0] else model_path
    if not Path(arch_path).exists():
        arch_path = model_path   # fall back to default checkpoint
    m = load_model(arch_path, arch=arch).to(device)
    m._arch_name = arch
    return m


def evaluate(patch_path, target_kmh=80, n_eot=16, batch_size=64,
             arch="resnet18", model_path="surrogate.pt"):
    device = get_device()
    print(f"\nDevice       : {device}")
    print(f"Patch        : {patch_path}")
    print(f"Target       : {target_kmh} km/h")
    print(f"EOT samples  : {n_eot}")

    if patch_path.endswith(".pt"):
        patch_01 = torch.load(patch_path, map_location="cpu")
    else:
        arr = np.array(Image.open(patch_path)).astype(np.float32) / 255.
        patch_01 = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)

    target_label    = KMH_TO_LABEL[target_kmh]
    target_patch_px = max(4, int(int(224 * 0.80) * 80 / 190))
    print(f"Footprint    : {target_patch_px}px in 224px input")
    print(f"Patch size   : {patch_01.shape[-1]}px")

    gtsrb_val = FilteredGTSRB("./data", split="test", transform=val_transforms, download=False)
    aus_val   = AUSynthDataset("./data/synthetic_aus_signs", transform=val_transforms,
                               split="val", auto_generate=False)
    val_ds    = ConcatDataset([gtsrb_val, aus_val])
    loader    = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Val samples  : {len(val_ds)}  (GTSRB={len(gtsrb_val)}, AU={len(aus_val)})")

    arch_list = [a.strip() for a in arch.split(",")]
    models = [_load_surrogate(a, model_path, device) for a in arch_list]
    for m in models:
        m.eval()
    print(f"Surrogates   : {[m._arch_name for m in models]}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_norm = (patch_01.to(device) - mean) / std

    n_m = len(models)
    totals         = [0] * n_m
    clean_corrects = [0] * n_m
    target_corrects= [0] * n_m
    baselines      = [0] * n_m

    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        with torch.no_grad():
            for mi, m in enumerate(models):
                clean_preds = m(imgs).argmax(1)
                clean_corrects[mi] += (clean_preds == labels).sum().item()
                baselines[mi]      += (clean_preds == target_label).sum().item()

                eot_votes = torch.zeros(B, dtype=torch.long, device=device)
                for _ in range(n_eot):
                    patched     = apply_patch(imgs, patch_norm, randomise_placement=True,
                                              target_patch_px=target_patch_px)
                    patched_01  = patched * std + mean
                    patched_01  = eot_batch(patched_01)
                    patched_eot = (patched_01 - mean) / std
                    preds       = m(patched_eot).argmax(1)
                    eot_votes  += (preds == target_label).long()

                target_corrects[mi] += (eot_votes >= (n_eot // 2 + 1)).sum().item()
                totals[mi]          += B

        # Progress bar shows worst-case adjusted ASR across models
        worst = min(
            (target_corrects[mi] / totals[mi] - baselines[mi] / totals[mi]) /
            (1.0 - baselines[mi] / totals[mi] + 1e-9)
            for mi in range(n_m)
        )
        pbar.set_postfix(worst_adj_ASR=f"{worst:.1%}")

    print(f"\n{'─'*60}")
    for mi, m in enumerate(models):
        T    = totals[mi]
        raw  = target_corrects[mi] / T
        base = baselines[mi] / T
        adj  = (raw - base) / (1.0 - base + 1e-9)
        print(f"[{m._arch_name:22s}]  clean={clean_corrects[mi]/T:.2%}  "
              f"raw_ASR={raw:.2%}  baseline={base:.2%}  adj_ASR={adj:.2%}")

    worst_adj = min(
        (target_corrects[mi] / totals[mi] - baselines[mi] / totals[mi]) /
        (1.0 - baselines[mi] / totals[mi] + 1e-9)
        for mi in range(n_m)
    )
    print(f"{'─'*60}")
    print(f"Worst-case adj ASR : {worst_adj:.2%}  (majority vote, {n_eot} EOT)")
    if worst_adj < 0.20:
        print("⚠  Adjusted ASR < 20% — patch may not be strong enough yet.")
    elif worst_adj < 0.50:
        print("~  Adjusted ASR 20-50% — moderate. Worth printing for physical test.")
    else:
        print("✓  Adjusted ASR > 50% — looks strong. Print and test physically.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("patch", help="Path to patch_step_XXXX.png or patch_945.pt")
    p.add_argument("--target", type=int,   default=80,            help="Target km/h")
    p.add_argument("--eot",    type=int,   default=16,            help="EOT samples")
    p.add_argument("--batch",  type=int,   default=64,            help="Batch size")
    p.add_argument("--arch",   type=str,   default="resnet18",
                   help=f"Comma-separated arch list. Choices: {','.join(ARCH_CHOICES)}")
    p.add_argument("--model",  type=str,   default="surrogate.pt",help="Surrogate checkpoint path")
    args = p.parse_args()
    evaluate(args.patch, target_kmh=args.target, n_eot=args.eot, batch_size=args.batch,
             arch=args.arch, model_path=args.model)
