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
from model import build_surrogate, get_device, load as load_model, ARCH_CHOICES
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


# ── losses ───────────────────────────────────────────────────────────────────

def tv_loss(patch_01: torch.Tensor) -> torch.Tensor:
    """
    Total Variation loss — penalises pixel-to-pixel differences at print resolution.
    Forces spatial coherence so the optimiser can't exploit high-frequency noise
    that averages away through the bilinear downsample.
    """
    dh = (patch_01[:, :, 1:, :] - patch_01[:, :, :-1, :]).abs().mean()
    dw = (patch_01[:, :, :, 1:] - patch_01[:, :, :, :-1]).abs().mean()
    return dh + dw


# Colours reproducible by a typical inkjet on matte paper (sRGB [0,1]).
# Source: Eykholt et al. 2018 supplementary + common inkjet characterisation.
_PRINTABLE_COLOURS = torch.tensor([
    [0.000, 0.000, 0.000],  # black
    [1.000, 1.000, 1.000],  # white
    [1.000, 0.000, 0.000],  # red
    [0.000, 1.000, 0.000],  # green
    [0.000, 0.000, 1.000],  # blue
    [1.000, 1.000, 0.000],  # yellow
    [0.000, 1.000, 1.000],  # cyan
    [1.000, 0.000, 1.000],  # magenta
    [0.800, 0.000, 0.000],  # dark red
    [0.000, 0.600, 0.000],  # dark green
    [0.000, 0.000, 0.800],  # dark blue
    [0.900, 0.500, 0.000],  # orange
    [0.600, 0.300, 0.000],  # brown
    [0.500, 0.500, 0.500],  # mid grey
    [0.750, 0.750, 0.750],  # light grey
    [0.250, 0.250, 0.250],  # dark grey
    [1.000, 0.600, 0.600],  # light red / pink
    [0.600, 0.800, 1.000],  # light blue
    [0.600, 1.000, 0.600],  # light green
    [1.000, 0.900, 0.600],  # cream / light yellow
], dtype=torch.float32)


def printability_loss(patch_01: torch.Tensor,
                      printable_colours: torch.Tensor = None) -> torch.Tensor:
    """
    Non-Printability Score (NPS) from Eykholt et al.
    For each pixel, find the minimum squared distance to any printable colour.
    Returns mean over all pixels — differentiable w.r.t. patch_01.
    """
    if printable_colours is None:
        printable_colours = _PRINTABLE_COLOURS
    pc = printable_colours.to(patch_01.device)

    pixels = patch_01.squeeze(0).permute(1, 2, 0).reshape(-1, 3)  # (N, 3)
    diff   = pixels.unsqueeze(1) - pc.unsqueeze(0)                # (N, P, 3)
    dist   = (diff ** 2).sum(-1)                                   # (N, P)
    return dist.min(dim=1).values.mean()


# ── optimisation loop ────────────────────────────────────────────────────────

def optimise_patch(
    models:       list,
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
    nps_weight:   float = 0.01,
    tv_weight:    float = 0.05,
) -> torch.Tensor:
    """
    patch_size is the PRINT resolution of the patch (e.g. 945 = 8cm @ 300 DPI).
    During optimisation it is downsampled via bilinear interpolation to the
    correct footprint in the 224px model input, so gradients flow through the
    resize back to the full-res tensor.  What you optimise is literally what
    you print — no separate upscale step.

    TV loss forces spatial coherence at print resolution, preventing the
    optimiser from exploiting high-frequency noise that averages away through
    the bilinear downsample.
    """
    if device is None:
        device = get_device()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    patch_mm        = print_cm * 10.0
    sign_input_px   = int(224 * 0.80)                  # ~179px sign in 224px input
    target_patch_px = max(4, int(sign_input_px * patch_mm / sign_diam_mm))
    print(f"Print-res patch : {patch_size}px  ({print_cm}cm @ 300 DPI)")
    print(f"Model footprint : {target_patch_px}px  in 224px input")
    print(f"Ensemble size   : {len(models)} model(s): "
          f"{[getattr(m, '_arch_name', '?') for m in models]}")

    for m in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    patch_01 = torch.rand(1, 3, patch_size, patch_size, device=device) * 0.5 + 0.25
    patch_01.requires_grad_(True)

    optimizer = torch.optim.Adam([patch_01], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    sub_ds = dataset if universal else torch.utils.data.Subset(
        dataset, [i for i, (_, l) in enumerate(dataset) if l != target_label]
    )

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
            # Sample one surrogate per EOT step — prevents the patch from
            # exploiting any single model's blind spots.
            surrogate = random.choice(models)
            patched     = apply_patch(imgs, patch_norm, randomise_placement=True,
                                      target_patch_px=target_patch_px)
            patched_01  = patched * std + mean
            patched_01  = eot_batch(patched_01.clone())
            patched_eot = (patched_01 - mean) / std

            logits = surrogate(patched_eot)
            loss   = F.cross_entropy(logits, target_t.expand(B))
            total_loss = total_loss + loss

        total_loss = total_loss / eot_samples

        # TV loss — force spatial coherence at print resolution
        if tv_weight > 0:
            total_loss = total_loss + tv_weight * tv_loss(patch_01.clamp(0, 1))

        # NPS — penalise colours inkjet can't reproduce
        if nps_weight > 0:
            total_loss = total_loss + nps_weight * printability_loss(patch_01.clamp(0, 1))

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
                # Report ASR as the worst-case (minimum) across all surrogates
                asr = min(
                    (m(patched_eot_log).argmax(1) == target_label).float().mean().item()
                    for m in models
                )
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
        img = img.resize((px, px), resample=Image.LANCZOS)
    img.save(path, dpi=(dpi, dpi))
    print(f"Saved print-ready patch: {path}  ({img.size[0]}×{img.size[1]} px @ {dpi} DPI)")


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate_patch(
    models:          list,
    dataset:         torch.utils.data.Dataset,
    patch_01:        torch.Tensor,
    target_label:    int,
    target_patch_px: int   = 32,
    device:          torch.device = None,
    batch_size:      int   = 64,
    n_eot:           int   = 16,
):
    """
    Evaluate ASR independently on each surrogate so you can see per-arch
    transfer, then report the ensemble (worst-case) figure.
    """
    if device is None:
        device = get_device()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_norm = (patch_01.to(device) - mean) / std

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    for m in models:
        m.eval()

    n_models = len(models)
    totals           = [0] * n_models
    correct_targets  = [0] * n_models
    orig_correct     = [0] * n_models

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            for mi, m in enumerate(models):
                orig_correct[mi] += (m(imgs).argmax(1) == labels).sum().item()

                eot_votes = torch.zeros(B, dtype=torch.long, device=device)
                for _ in range(n_eot):
                    patched     = apply_patch(imgs, patch_norm, randomise_placement=True,
                                              target_patch_px=target_patch_px)
                    patched_01  = patched * std + mean
                    patched_01  = eot_batch(patched_01)
                    patched_eot = (patched_01 - mean) / std
                    preds       = m(patched_eot).argmax(1)
                    eot_votes  += (preds == target_label).long()

                correct_targets[mi] += (eot_votes >= (n_eot // 2 + 1)).sum().item()
                totals[mi]          += B

    print(f"\n── Patch Evaluation (EOT + random placement) ──")
    arch_names = [getattr(m, '_arch_name', f'model_{i}') for i, m in enumerate(models)]
    for mi, name in enumerate(arch_names):
        T = totals[mi]
        print(f"  [{name:22s}]  clean={orig_correct[mi]/T:.2%}  "
              f"ASR={correct_targets[mi]/T:.2%}")

    worst_asr = min(correct_targets[mi] / totals[mi] for mi in range(n_models))
    print(f"  Worst-case (ensemble) ASR : {worst_asr:.2%}  "
          f"(target={ALL_SPEEDS[target_label]} km/h, majority vote {n_eot} EOT)")


# ── main ──────────────────────────────────────────────────────────────────────

def _load_or_build(model_path: str, arch: str, pretrain_path: str, device) -> torch.nn.Module:
    if Path(model_path).exists():
        m = load_model(model_path, arch=arch).to(device)
        print(f"  Loaded {arch} from {model_path}")
    else:
        print(f"  No checkpoint at {model_path} — building from pretrain")
        m = build_surrogate(arch=arch, pretrain_path=pretrain_path).to(device)
    m._arch_name = arch
    return m


def main(args):
    device = get_device()
    print(f"Device: {device}")

    # Build ensemble — one surrogate per requested arch.
    # --arch accepts a comma-separated list, e.g. "resnet18,mobilenet_v3_small"
    arch_list = [a.strip() for a in args.arch.split(",")]
    ensemble = []
    for arch in arch_list:
        # Each arch gets its own checkpoint file derived from --model, e.g.
        # surrogate_mobilenet_v3_small.pt alongside surrogate.pt.
        stem   = Path(args.model).stem
        suffix = Path(args.model).suffix
        arch_path = str(Path(args.model).parent / f"{stem}_{arch}{suffix}") \
                    if arch != arch_list[0] else args.model
        ensemble.append(_load_or_build(arch_path, arch, args.pretrain_path, device))

    print(f"Ensemble: {[m._arch_name for m in ensemble]}")

    from torch.utils.data import ConcatDataset
    gtsrb_val = FilteredGTSRB(args.data, split="test", transform=val_transforms, download=False)
    aus_val   = AUSynthDataset(args.aus_data, transform=val_transforms, split="val", auto_generate=False)
    val_ds    = ConcatDataset([gtsrb_val, aus_val])
    print(f"Val samples: {len(val_ds)}  (GTSRB={len(gtsrb_val)}, AU synth={len(aus_val)})")

    target_label = KMH_TO_LABEL[args.target]
    print(f"Target: {args.target} km/h  (label {target_label})")

    patch_01, target_patch_px = optimise_patch(
        models       = ensemble,
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
        nps_weight   = args.nps_weight,
        tv_weight    = args.tv_weight,
    )

    torch.save(patch_01, args.out)
    print(f"Patch tensor saved: {args.out}")

    png_path = Path(args.out).with_suffix(".png")
    save_patch_png(patch_01.cpu(), str(png_path), print_cm=args.print_cm)

    evaluate_patch(ensemble, val_ds, patch_01, target_label,
                   target_patch_px=target_patch_px, device=device, n_eot=args.eot_samples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="surrogate.pt")
    p.add_argument("--pretrain-path", default="./gtsrb_backbone.pt")
    p.add_argument("--arch",          default="resnet18",
                   help=f"Comma-separated list of architectures for ensemble. "
                        f"Choices: {','.join(ARCH_CHOICES)}")
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
    p.add_argument("--nps-weight",    type=float, default=0.01, help="Printability loss weight (0 to disable)")
    p.add_argument("--tv-weight",     type=float, default=0.05, help="Total variation loss weight (0 to disable)")
    main(p.parse_args())
