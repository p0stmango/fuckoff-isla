"""
Universal adversarial patch optimiser (targeted — misclassify any speed sign as 80 km/h).

Algorithm: PGD-style iterative update with EOT on a RECTANGULAR patch applied
to the sign image.  The patch is the only variable being optimised.

Usage:
    python patch_attack.py \
        --model surrogate.pt \
        --data  ./data \
        --out   patch.pt \
        [--patch-size 945] \
        [--steps 2000] \
        [--lr 0.01] \
        [--eot-samples 16] \
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

# ── denormalise helper ───────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    return t.cpu() * _STD + _MEAN


# ── patch application (rectangular — no mask) ────────────────────────────────

def apply_patch(
    images: torch.Tensor,          # (B, 3, H, W)  normalised
    patch_norm: torch.Tensor,      # (1, 3, P, P)  normalised patch — may be print-res
    cx_frac: float = 0.5,
    cy_frac: float = 0.5,
    randomise_placement: bool = False,
    target_patch_px: int = None,   # if set, downsample patch to this size first
) -> torch.Tensor:
    """
    Place the rectangular patch centred at (cx_frac, cy_frac).
    No circular mask — full rectangle is applied.

    target_patch_px: the patch footprint in the model's input space (px).
    When patch_norm is at print resolution (e.g. 945px) and the model sees
    224px images, pass target_patch_px=32 to downsample through a
    differentiable bilinear interpolation before compositing.  Gradients
    flow back through the interpolation to update the full-res patch tensor.
    """
    B, C, H, W = images.shape

    # Downsample print-res patch to model input footprint (differentiable)
    if target_patch_px is not None and patch_norm.shape[-1] != target_patch_px:
        patch_small = F.interpolate(
            patch_norm, size=(target_patch_px, target_patch_px),
            mode="bilinear", align_corners=False,
        )
    else:
        patch_small = patch_norm

    P = patch_small.shape[-1]

    patched     = images.clone()
    patch_tiled = patch_small.expand(B, -1, -1, -1)

    if not randomise_placement:
        top  = max(0, min(int(cy_frac * H - P / 2), H - P))
        left = max(0, min(int(cx_frac * W - P / 2), W - P))
        patched[:, :, top:top + P, left:left + P] = patch_tiled
    else:
        margin = int(H * 0.20)
        for b in range(B):
            top  = random.randint(margin, max(margin, H - P - margin))
            left = random.randint(margin, max(margin, W - P - margin))
            patched[b:b+1, :, top:top + P, left:left + P] = patch_tiled[b:b+1]

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
    print_cm:     float = 8.0,
    sign_diam_mm: float = 190.0,
    real_sign_mm: float = 450.0,
) -> torch.Tensor:
    """
    patch_size is the PRINT resolution of the patch (e.g. 945 = 8cm @ 300 DPI).
    During optimisation it is downsampled via bilinear interpolation to the
    correct footprint in the 224px model input, so gradients flow through the
    resize back to the full-res tensor.  What you optimise is literally what
    you print — no separate upscale step.
    """
    if device is None:
        device = get_device()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Compute patch footprint in model input space (224px)
    # Sign occupies sign_diam_mm/real_sign_mm of the real sign diameter.
    # In the model input (224px) the sign fills ~80% = 179px.
    patch_mm        = print_cm * 10.0
    patch_frac      = patch_mm / real_sign_mm          # fraction of real sign diam
    sign_input_px   = int(224 * 0.80)                  # ~179px sign in 224px input
    target_patch_px = max(4, int(sign_input_px * patch_frac * (sign_diam_mm / real_sign_mm)))
    # Simpler direct calculation: patch covers patch_mm of the 190mm rendered sign
    # which fills 179px → patch_px = 179 * patch_mm / 190
    target_patch_px = max(4, int(sign_input_px * patch_mm / sign_diam_mm))
    print(f"Print-res patch : {patch_size}px  ({print_cm}cm @ 300 DPI)")
    print(f"Model footprint : {target_patch_px}px  in 224px input")

    patch_01 = torch.rand(1, 3, patch_size, patch_size, device=device) * 0.5 + 0.25
    patch_01.requires_grad_(True)

    optimizer = torch.optim.Adam([patch_01], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

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
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def to_normalised(p_01):
        return (p_01 - mean) / std

    pbar = tqdm(range(1, steps + 1), desc="Optimising patch")
    for step in pbar:
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)

        imgs = imgs.to(device)
        B    = imgs.size(0)

        with torch.no_grad():
            patch_01.clamp_(0.0, 1.0)

        patch_norm = to_normalised(patch_01)
        total_loss = torch.tensor(0.0, device=device)

        for _ in range(eot_samples):
            patched = apply_patch(imgs, patch_norm, randomise_placement=True,
                                  target_patch_px=target_patch_px)
            patched_01  = patched * std + mean
            patched_01  = eot_batch(patched_01.clone())
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

        with torch.no_grad():
            patch_01.clamp_(0.0, 1.0)

        if step % 50 == 0:
            with torch.no_grad():
                patch_norm_det  = to_normalised(patch_01.detach().clamp(0, 1))
                patched_log     = apply_patch(imgs, patch_norm_det, randomise_placement=True,
                                              target_patch_px=target_patch_px)
                patched_01_log  = patched_log * std + mean
                patched_01_log  = eot_batch(patched_01_log)
                patched_eot_log = (patched_01_log - mean) / std
                preds = model(patched_eot_log).argmax(1)
                asr   = (preds == target_label).float().mean().item()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", ASR=f"{asr:.2%}")

            arr = patch_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((arr * 255).astype(np.uint8)).save(f"patch_step_{step:04d}.png")

    return patch_01.detach().clamp(0, 1), target_patch_px


# ── export ────────────────────────────────────────────────────────────────────

def save_patch_png(patch_01: torch.Tensor, path: str, print_cm: float = 8.0):
    """Patch tensor is already at print resolution — save directly, no upscale."""
    arr = patch_01.squeeze(0).permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    dpi = 300
    px  = int(print_cm / 2.54 * dpi)
    if img.size != (px, px):
        # Only resize if somehow wrong size (shouldn't happen with --patch-size 945)
        img = img.resize((px, px), resample=Image.LANCZOS)
    img.save(path, dpi=(dpi, dpi))
    print(f"Saved print-ready patch: {path}  ({img.size[0]}×{img.size[1]} px @ {dpi} DPI)")


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate_patch(
    model:           torch.nn.Module,
    dataset:         torch.utils.data.Dataset,
    patch_01:        torch.Tensor,
    target_label:    int,
    target_patch_px: int   = 32,
    device:          torch.device = None,
    batch_size:      int   = 64,
    n_eot:           int   = 16,
):
    if device is None:
        device = get_device()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_norm = (patch_01.to(device) - mean) / std

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model.eval()
    total = correct_target = originally_correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            clean_preds = model(imgs).argmax(1)
            originally_correct += (clean_preds == labels).sum().item()

            eot_votes = torch.zeros(imgs.size(0), dtype=torch.long, device=device)
            for _ in range(n_eot):
                patched     = apply_patch(imgs, patch_norm, randomise_placement=True,
                                          target_patch_px=target_patch_px)
                patched_01  = patched * std + mean
                patched_01  = eot_batch(patched_01)
                patched_eot = (patched_01 - mean) / std
                preds       = model(patched_eot).argmax(1)
                eot_votes  += (preds == target_label).long()

            correct_target += (eot_votes >= (n_eot // 2 + 1)).sum().item()
            total          += imgs.size(0)

    print(f"\n── Patch Evaluation (EOT + random placement) ──")
    print(f"Total samples       : {total}")
    print(f"Clean accuracy      : {originally_correct/total:.2%}")
    print(f"Attack success rate : {correct_target/total:.2%}  "
          f"(target={ALL_SPEEDS[target_label]} km/h, majority vote over {n_eot} EOT samples)")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device: {device}")

    if Path(args.model).exists():
        model = load_model(args.model, arch=args.arch).to(device)
        print(f"Loaded surrogate from {args.model}")
    else:
        print(f"No checkpoint at {args.model} — building from pretrain")
        model = build_surrogate(arch=args.arch, pretrain_path=args.pretrain_path).to(device)

    from torch.utils.data import ConcatDataset
    gtsrb_val = FilteredGTSRB(args.data, split="test", transform=val_transforms, download=False)
    aus_val   = AUSynthDataset(args.aus_data, transform=val_transforms, split="val", auto_generate=False)
    val_ds    = ConcatDataset([gtsrb_val, aus_val])
    print(f"Val samples: {len(val_ds)}  (GTSRB={len(gtsrb_val)}, AU synth={len(aus_val)})")

    target_label = KMH_TO_LABEL[args.target]
    print(f"Target: {args.target} km/h  (label {target_label})")

    patch_01, target_patch_px = optimise_patch(
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
        print_cm     = args.print_cm,
    )

    torch.save(patch_01, args.out)
    print(f"Patch tensor saved: {args.out}")

    png_path = Path(args.out).with_suffix(".png")
    save_patch_png(patch_01.cpu(), str(png_path), print_cm=args.print_cm)

    evaluate_patch(model, val_ds, patch_01, target_label,
                   target_patch_px=target_patch_px, device=device, n_eot=args.eot_samples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="surrogate.pt")
    p.add_argument("--pretrain-path", default="./gtsrb_backbone.pt")
    p.add_argument("--arch",          default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--data",          default="./data")
    p.add_argument("--aus-data",      default="./data/aus_synth")
    p.add_argument("--out",           default="patch.pt")
    p.add_argument("--universal",     action="store_true", default=True)
    p.add_argument("--target",        type=int,   default=80,   help="Target km/h class")
    p.add_argument("--patch-size",    type=int,   default=945,  help="Patch pixel size")
    p.add_argument("--steps",         type=int,   default=2000)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--eot-samples",   type=int,   default=16)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--print-cm",      type=float, default=8.0,  help="Printed patch size in cm")
    main(p.parse_args())
