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

from dataset import ALL_SPEEDS, val_transforms, KMH_TO_LABEL, FilteredGTSRB, AUSynthDataset
from model import load as load_model, get_device
from patch_attack import apply_patch
from eot import eot_batch
from torch.utils.data import ConcatDataset, DataLoader

def evaluate(patch_path, target_kmh=80, n_eot=16, batch_size=64):
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

    model = load_model("surrogate.pt").to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_norm = (patch_01.to(device) - mean) / std

    total = clean_correct = target_correct = baseline_target = 0

    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        with torch.no_grad():
            clean_preds = model(imgs).argmax(1)
            clean_correct   += (clean_preds == labels).sum().item()
            baseline_target += (clean_preds == target_label).sum().item()

            eot_votes = torch.zeros(B, dtype=torch.long, device=device)
            for _ in range(n_eot):
                patched     = apply_patch(imgs, patch_norm, randomise_placement=True,
                                          target_patch_px=target_patch_px)
                patched_01  = patched * std + mean
                patched_01  = eot_batch(patched_01)
                patched_eot = (patched_01 - mean) / std
                preds       = model(patched_eot).argmax(1)
                eot_votes  += (preds == target_label).long()

            target_correct += (eot_votes >= (n_eot // 2 + 1)).sum().item()
            total          += B

        clean_acc = clean_correct / total
        raw_asr   = target_correct / total
        baseline  = baseline_target / total
        adj_asr   = (raw_asr - baseline) / (1.0 - baseline + 1e-9)
        pbar.set_postfix(
            clean=f"{clean_acc:.1%}",
            raw_ASR=f"{raw_asr:.1%}",
            baseline=f"{baseline:.1%}",
            adj_ASR=f"{adj_asr:.1%}",
        )

    print(f"\n{'─'*55}")
    print(f"Total samples      : {total}")
    print(f"Clean accuracy     : {clean_correct/total:.2%}")
    print(f"Baseline →{target_kmh:>4}     : {baseline_target/total:.2%}  (no patch)")
    print(f"Raw ASR  →{target_kmh:>4}     : {target_correct/total:.2%}  (majority vote, {n_eot} EOT)")
    adj = (target_correct/total - baseline_target/total) / (1.0 - baseline_target/total + 1e-9)
    print(f"Adjusted ASR       : {adj:.2%}  (corrected for baseline bias)")
    print(f"{'─'*55}")
    if adj < 0.20:
        print("⚠  Adjusted ASR < 20% — patch may not be strong enough yet.")
    elif adj < 0.50:
        print("~  Adjusted ASR 20-50% — moderate. Worth printing for physical test.")
    else:
        print("✓  Adjusted ASR > 50% — looks strong. Print and test physically.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("patch", help="Path to patch_step_XXXX.png or patch_945.pt")
    p.add_argument("--target", type=int, default=80, help="Target km/h")
    p.add_argument("--eot",    type=int, default=16, help="EOT samples")
    p.add_argument("--batch",  type=int, default=64, help="Batch size")
    args = p.parse_args()
    evaluate(args.patch, target_kmh=args.target, n_eot=args.eot, batch_size=args.batch)
