"""
Universal adversarial patch optimiser (targeted — misclassify any speed sign as 50 km/h).

Algorithm: PGD-style iterative update with EOT on a circular patch applied
to the sign image.  The patch is the only variable being optimised.

Patch placement: centred on the sign face, covering ~35% of sign diameter.
This corresponds to roughly 12–15 cm on a real sign, which fits on A4 paper.

Usage:
    python patch_attack.py \\
        --model surrogate.pt \\
        --data  ./data \\
        --out   patch.pt \\
        [--patch-size 80] \\
        [--steps 2000] \\
        [--lr 0.01] \\
        [--eot-samples 16] \\
        [--batch 32]
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
import numpy as np

from dataset import (
    AUSynthDataset, FilteredGTSRB, val_transforms, NORMALIZE,
    KMH_TO_LABEL, ALL_SPEEDS, IMG_SIZE, NUM_CLASSES,
)
from model import build_surrogate, get_device, load as load_model
from eot import eot_batch

# ── denormalise helper (for rendering) ──────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    return t.cpu() * _STD + _MEAN


# ── circular mask ────────────────────────────────────────────────────────────

def make_circular_mask(patch_size: int, device) -> torch.Tensor:
    """1 inside circle, 0 outside — shape (1, 1, P, P)."""
    y, x = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    cx = cy = patch_size / 2.0
    r  = patch_size / 2.0
    mask = ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).float()
    return mask.unsqueeze(0).unsqueeze(0)   # (1,1,P,P)


# ── patch application ────────────────────────────────────────────────────────

def apply_patch(
    images: torch.Tensor,          # (B, 3, H, W)  normalised
    patch_norm: torch.Tensor,      # (1, 3, P, P)  normalised patch
    mask: torch.Tensor,            # (1, 1, P, P)
    cx_frac: float = 0.5,          # horizontal centre as fraction of W
    cy_frac: float = 0.5,          # vertical centre as fraction of H
    randomise_placement: bool = False,  # jitter position within sign face
) -> torch.Tensor:
    """
    Place the patch centred at (cx_frac, cy_frac) of the image.

    randomise_placement=True jitters the position across each image in
    the batch independently, sampling uniformly within the inner 60% of
    the sign face.  Used during optimisation so the patch doesn't overfit
    to a single location.
    """
    B, C, H, W = images.shape
    P = patch_norm.shape[-1]

    patched     = images.clone()
    patch_tiled = patch_norm.expand(B, -1, -1, -1)
    mask_tiled  = mask.expand(B, C, -1, -1)

    if not randomise_placement:
        top  = max(0, min(int(cy_frac * H - P / 2), H - P))
        left = max(0, min(int(cx_frac * W - P / 2), W - P))
        patched[:, :, top:top + P, left:left + P] = (
            patch_tiled * mask_tiled
            + patched[:, :, top:top + P, left:left + P] * (1 - mask_tiled)
        )
    else:
        # Each image in the batch gets an independently jittered placement.
        # Sample within a central band — the patch must stay on the sign face
        # (inner 60% avoids the red ring and sign edges).
        margin = int(H * 0.20)
        for b in range(B):
            top  = random.randint(margin, max(margin, H - P - margin))
            left = random.randint(margin, max(margin, W - P - margin))
            patched[b:b+1, :, top:top + P, left:left + P] = (
                patch_tiled[b:b+1] * mask_tiled[b:b+1]
                + patched[b:b+1, :, top:top + P, left:left + P] * (1 - mask_tiled[b:b+1])
            )

    return patched


# ── optimisation loop ────────────────────────────────────────────────────────

def optimise_patch(
    model:        torch.nn.Module,
    dataset:      torch.utils.data.Dataset,
    target_label: int,
    patch_size:   int   = 80,
    steps:        int   = 2000,
    lr:           float = 0.01,
    eot_samples:  int   = 16,
    batch_size:   int   = 32,
    device:       torch.device = None,
    universal:    bool  = True,
) -> torch.Tensor:
    """
    Returns the optimised patch as a (1, 3, P, P) tensor in [0,1].

    universal=True  — draws random images from all classes each step, making the
                      patch fool any sign (desired for "any sign in carpark").
    universal=False — draws only images whose true label != target_label, which
                      is slightly more focused but only useful if source is known.
    """
    if device is None:
        device = get_device()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Initialise patch in [0,1] space, then convert to normalised space
    patch_01 = torch.rand(1, 3, patch_size, patch_size, device=device) * 0.5 + 0.25
    patch_01.requires_grad_(True)

    optimizer = torch.optim.Adam([patch_01], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # For universal attack use full dataset; otherwise exclude the target class
    # so we don't accidentally reinforce "already correct" predictions.
    if not universal:
        indices = [i for i, (_, l) in enumerate(dataset) if l != target_label]
        sub_ds  = torch.utils.data.Subset(dataset, indices)
    else:
        sub_ds = dataset

    loader = torch.utils.data.DataLoader(
        sub_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    data_iter = iter(loader)

    target_t = torch.tensor([target_label], device=device)
    mask     = make_circular_mask(patch_size, device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def to_normalised(p_01):
        return (p_01 - mean) / std

    asr_history = []

    pbar = tqdm(range(1, steps + 1), desc="Optimising patch")
    for step in pbar:
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)

        imgs = imgs.to(device)
        B    = imgs.size(0)

        # Clamp patch to valid [0,1] range before each step
        with torch.no_grad():
            patch_01.clamp_(0.0, 1.0)

        patch_norm = to_normalised(patch_01)

        total_loss = torch.tensor(0.0, device=device)

        # EOT: average loss over N random physical transforms
        for _ in range(eot_samples):
            # Randomise placement per sample — patch must work anywhere on sign
            patched = apply_patch(imgs, patch_norm, mask, randomise_placement=True)
            # EOT operates in [0,1] so denorm → EOT → renorm
            patched_01 = patched * std + mean
            patched_01 = eot_batch(patched_01.clone())
            patched_eot = (patched_01 - mean) / std

            logits = model(patched_eot)
            labels = target_t.expand(B)
            loss   = F.cross_entropy(logits, labels)
            total_loss = total_loss + loss

        total_loss = total_loss / eot_samples
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Project patch back into valid range after update (proper PGD step)
        with torch.no_grad():
            patch_01.clamp_(0.0, 1.0)

        # ── logging ──
        # ASR measured under the same EOT + random placement conditions as
        # training — centred/clean eval would give inflated numbers.
        if step % 50 == 0:
            with torch.no_grad():
                patch_norm_det  = to_normalised(patch_01.detach().clamp(0, 1))
                patched_log     = apply_patch(imgs, patch_norm_det, mask, randomise_placement=True)
                patched_01_log  = patched_log * std + mean
                patched_01_log  = eot_batch(patched_01_log)
                patched_eot_log = (patched_01_log - mean) / std
                preds = model(patched_eot_log).argmax(1)
                asr   = (preds == target_label).float().mean().item()
            asr_history.append(asr)
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", ASR=f"{asr:.2%}")

            # Save patch snapshot every 50 steps so you can watch it evolve
            arr = patch_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((arr * 255).astype(np.uint8)).save(f"patch_step_{step:04d}.png")

    return patch_01.detach().clamp(0, 1)


# ── export to printable PNG ───────────────────────────────────────────────────

def save_patch_png(patch_01: torch.Tensor, path: str, print_cm: float = 15.0):
    """
    Save patch as a high-res PNG ready for A4 printing.
    At 300 DPI, 15 cm ≈ 1772 px.  We upscale the patch to that resolution.
    """
    arr = patch_01.squeeze(0).permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    dpi = 300
    px  = int(print_cm / 2.54 * dpi)
    img_high = img.resize((px, px), resample=Image.LANCZOS)
    img_high.save(path, dpi=(dpi, dpi))
    print(f"Saved print-ready patch: {path}  ({px}×{px} px @ {dpi} DPI)")


# ── evaluate patch across full val set ───────────────────────────────────────

def evaluate_patch(
    model:        torch.nn.Module,
    dataset:      torch.utils.data.Dataset,
    patch_01:     torch.Tensor,
    target_label: int,
    patch_size:   int   = 80,
    device:       torch.device = None,
    batch_size:   int   = 64,
    n_eot:        int   = 16,
):
    """
    Evaluate patch ASR under EOT + random placement — same conditions as
    training.  Also reports clean accuracy to confirm the surrogate is sane.
    """
    if device is None:
        device = get_device()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_norm = (patch_01.to(device) - mean) / std
    mask       = make_circular_mask(patch_size, device)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model.eval()
    total = correct_target = originally_correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Clean accuracy — no patch
            clean_preds = model(imgs).argmax(1)
            originally_correct += (clean_preds == labels).sum().item()

            # Attack ASR — averaged over n_eot EOT samples per batch
            eot_votes = torch.zeros(imgs.size(0), dtype=torch.long, device=device)
            for _ in range(n_eot):
                patched     = apply_patch(imgs, patch_norm, mask, randomise_placement=True)
                patched_01  = patched * std + mean
                patched_01  = eot_batch(patched_01)
                patched_eot = (patched_01 - mean) / std
                preds       = model(patched_eot).argmax(1)
                eot_votes  += (preds == target_label).long()

            # Majority vote across EOT samples
            correct_target += (eot_votes >= (n_eot // 2 + 1)).sum().item()
            total          += imgs.size(0)

    print(f"\n── Patch Evaluation (EOT + random placement) ──")
    print(f"Total samples       : {total}")
    print(f"Clean accuracy      : {originally_correct/total:.2%}")
    print(f"Attack success rate : {correct_target/total:.2%}  "
          f"(target={ALL_SPEEDS[target_label]} km/h, majority vote over {n_eot} EOT samples)")


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device: {device}")

    if Path(args.model).exists():
        model = load_model(args.model, arch=args.arch).to(device)
        print(f"Loaded fine-tuned surrogate from {args.model}")
    else:
        print(f"No checkpoint at {args.model} — building surrogate from pretrain cache")
        model = build_surrogate(arch=args.arch, pretrain_path=args.pretrain_path).to(device)

    from torch.utils.data import ConcatDataset
    gtsrb_val = FilteredGTSRB(args.data, split="test", transform=val_transforms, download=False)
    aus_val   = AUSynthDataset(args.aus_data, transform=val_transforms, split="val", auto_generate=False)
    val_ds    = ConcatDataset([gtsrb_val, aus_val])
    print(f"Val samples: {len(val_ds)}  (GTSRB={len(gtsrb_val)}, AU synth={len(aus_val)})")

    target_label = KMH_TO_LABEL[args.target]
    print(f"Target class: {args.target} km/h  (label {target_label})")
    print(f"Universal attack: {args.universal}")

    patch_01 = optimise_patch(
        model        = model,
        dataset      = val_ds,
        target_label = target_label,
        patch_size   = args.patch_size,
        steps        = args.steps,
        lr           = args.lr,
        eot_samples  = args.eot_samples,
        batch_size   = args.batch,
        device       = device,
        universal    = args.universal,
    )

    torch.save(patch_01, args.out)
    print(f"Patch tensor saved: {args.out}")

    png_path = Path(args.out).with_suffix(".png")
    save_patch_png(patch_01.cpu(), str(png_path), print_cm=args.print_cm)

    evaluate_patch(model, val_ds, patch_01, target_label,
                   patch_size=args.patch_size, device=device, n_eot=args.eot_samples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="surrogate.pt",
                   help="Path to fine-tuned surrogate checkpoint")
    p.add_argument("--pretrain-path", default="./gtsrb_backbone.pt",
                   help="Fallback pretrain cache if --model not found")
    p.add_argument("--arch",          default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--data",          default="./data")
    p.add_argument("--aus-data",      default="./data/aus_synth")
    p.add_argument("--out",           default="patch.pt")
    p.add_argument("--universal",     action="store_true", default=True,
                   help="Optimise patch against all classes (any sign → target)")
    p.add_argument("--target",        type=int,   default=50, help="Target km/h class")
    p.add_argument("--patch-size",    type=int,   default=80, help="Patch pixel size in 224px input")
    p.add_argument("--steps",         type=int,   default=2000)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--eot-samples",   type=int,   default=16,  help="EOT transforms per step")
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--print-cm",      type=float, default=15.0, help="Printed patch diameter in cm")
    main(p.parse_args())