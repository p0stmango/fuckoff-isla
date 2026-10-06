"""
Universal adversarial patch optimiser (targeted or untargeted).
Targeted: misclassify any speed sign as a specific class (default 80 km/h).
Untargeted: misclassify as ANY wrong class — much easier to achieve physically
because the optimizer has N-1 classes to push toward instead of 1.

Algorithm: PGD-style iterative update with EOT on a RECTANGULAR patch applied
to the sign image.  The patch is the only variable being optimised.

Usage:
    python patch_attack.py \
        --model surrogate.pt \
        --arch  resnet18,resnet50,mobilenet_v3_small \
        --data  ./data \
        --out   patch.pt \
        [--patch-size 945] \
        [--steps 2000] \
        [--lr 0.01] \
        [--eot-samples 16] \
        [--batch 32] \
        [--tv-weight 0.05] \
        [--nps-weight 0.01]
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
from eot import (eot_batch, eot_print, eot_scene, install_fake_quant_hooks,
                 FakeQuantSchedule, _ste_clamp_01)


# ── INT8 eval via deterministic fake-quant ──────────────────────────────────
#
# Instead of PyTorch's PTQ infrastructure (which needs backend-specific
# quantised op kernels that don't exist for depthwise-separable convs on
# aarch64), we hook the float models with deterministic fake-quant at
# p=1.0 during eval.  Every Conv2d/Linear activation gets quantise→dequantise
# on every forward — same information bottleneck as true INT8, just computed
# in float.  Works on any architecture, any platform, any device.

def _build_quant_eval_hooks(model: torch.nn.Module,
                             bits: int = 8,
                             calibration_loader=None) -> list:
    """
    Install deterministic (p=1.0) fake-quant hooks on every Conv2d/Linear in
    `model` — weights use symmetric quant, activations use asymmetric (matching
    EyeQ4's INT8 NPU: symmetric weights, asymmetric post-ReLU activations).

    If calibration_loader is provided, run a calibration pass first to collect
    per-layer activation statistics for more accurate scale/zero-point values.
    Returns the hook handles so they can be removed later.
    """
    from eot import _sym_fake_quant, _asym_fake_quant

    # Optional calibration: collect per-layer min/max statistics
    _calibration_stats = {}
    if calibration_loader is not None:
        _cal_hooks = []
        def _make_cal_hook(name):
            def _cal(module, inp, out):
                if name not in _calibration_stats:
                    _calibration_stats[name] = {'min': float('inf'), 'max': float('-inf')}
                _calibration_stats[name]['min'] = min(
                    _calibration_stats[name]['min'], out.detach().min().item())
                _calibration_stats[name]['max'] = max(
                    _calibration_stats[name]['max'], out.detach().max().item())
            return _cal

        for name, m in model.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                _cal_hooks.append(m.register_forward_hook(_make_cal_hook(name)))

        model.eval()
        with torch.no_grad():
            for imgs, _ in calibration_loader:
                model(imgs.to(next(model.parameters()).device))
        for h in _cal_hooks:
            h.remove()

    # Cache quantised weights per module
    _w_cache = {}

    def _det_quant_hook(module, inp, out):
        # Weight quant — symmetric, cached (weights don't change during eval)
        mid = id(module)
        if mid not in _w_cache:
            _w_cache[mid], _ = _sym_fake_quant(module.weight, bits)
        w_dq = _w_cache[mid]

        if isinstance(module, torch.nn.Conv2d):
            out_wq = F.conv2d(inp[0], w_dq, module.bias,
                              module.stride, module.padding, module.dilation,
                              module.groups)
        elif isinstance(module, torch.nn.Linear):
            out_wq = F.linear(inp[0], w_dq, module.bias)
        else:
            out_wq = out

        # Activation quant — asymmetric for post-ReLU, symmetric otherwise
        if out_wq.min() >= -1e-6:
            act_dq, _, _ = _asym_fake_quant(out_wq, bits)
        else:
            act_dq, _ = _sym_fake_quant(out_wq, bits)
        return act_dq  # no STE needed — eval only, no backward

    handles = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(m.register_forward_hook(_det_quant_hook))
    return handles


# ── denormalise helper ───────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    return t.cpu() * _STD + _MEAN


# ── patch application (rectangular — no mask) ────────────────────────────────

def make_patch_mask(
    B: int, H: int, W: int, P: int,
    cx_frac: float, cy_frac: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a binary mask (B, 1, H, W) with 1.0 where the patch sits.
    Used to tell eot_scene which pixels are matte-printed patch vs
    retroreflective sign background — for differential retroreflection.
    """
    mask = torch.zeros(B, 1, H, W, device=device)
    top  = max(0, min(int(cy_frac * H - P / 2), H - P))
    left = max(0, min(int(cx_frac * W - P / 2), W - P))
    mask[:, :, top:top + P, left:left + P] = 1.0
    return mask


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

def semi_targeted_probe(models, imgs, labels, patch_01, to_normalised_fn,
                        eot_print_fn, eot_scene_fn, apply_patch_fn,
                        make_patch_mask_fn, target_patch_px, cx_min, cx_max,
                        cy_min, cy_max, mean, std, device, n_probes=8,
                        oblique_eot=False, night_mode=False,
                        _use_autocast=False, IMG_SIZE=224):
    """
    Probe which non-true class the ensemble prefers, for semi-targeted mode.
    Runs n_probes EOT draws across all models and tallies predictions.
    Returns the class index that appears most often (excluding true class).
    """
    import random as _random
    from collections import Counter
    votes = Counter()
    B = imgs.size(0)
    true_set = set(labels.cpu().tolist())

    with torch.no_grad():
        for _ in range(n_probes):
            cx = _random.uniform(cx_min, cx_max)
            cy = _random.uniform(cy_min, cy_max)
            patch_printed = eot_print_fn(patch_01.detach().clamp(0, 1))
            patch_norm = to_normalised_fn(patch_printed)
            patched = apply_patch_fn(imgs, patch_norm, cx_frac=cx, cy_frac=cy,
                                     randomise_placement=False,
                                     target_patch_px=target_patch_px)
            p_mask = make_patch_mask_fn(B, IMG_SIZE, IMG_SIZE,
                                        target_patch_px, cx, cy, device)
            patched_01 = patched * std + mean
            patched_01 = eot_scene_fn(patched_01.clone(), oblique=oblique_eot,
                                       patch_mask=p_mask, training_step=0,
                                       night_mode=night_mode)
            patched_eot = (patched_01 - mean) / std
            for m in models:
                with torch.autocast("mps", dtype=torch.float16, enabled=_use_autocast):
                    preds = m(patched_eot).argmax(1)
                for p in preds.cpu().tolist():
                    if p not in true_set:
                        votes[p] += 1

    if not votes:
        return None
    return votes.most_common(1)[0][0]


def untargeted_loss(logits: torch.Tensor, true_labels: torch.Tensor,
                    margin: float = 5.0) -> torch.Tensor:
    """
    Untargeted CW-style margin loss: max(0, z_true - max_other + margin).

    Pushes the true-class logit BELOW the best non-true class by at least
    `margin` logit units.  The optimizer is free to pick whichever wrong
    class is easiest to push above the true class — much easier than forcing
    a specific target, especially against temporal majority voting where
    per-frame success rate needs to be very high.
    """
    B, C = logits.shape
    one_hot  = F.one_hot(true_labels, num_classes=C).bool()
    z_true   = logits[one_hot].view(B)                              # (B,)
    z_other  = logits.masked_fill(one_hot, -65000.0).max(dim=1).values  # (B,)  # fits float16
    return F.relu(z_true - z_other + margin).mean()


def margin_loss(logits: torch.Tensor, target: torch.Tensor,
                margin: float = 10.0) -> torch.Tensor:
    """
    CW-style margin loss: max(0, max_other - z_target + margin).

    Pushes the target logit above the runner-up by at least `margin` logit
    units.  Unlike cross-entropy, the gradient doesn't vanish once the target
    class wins by a slim plurality — the optimiser keeps pushing until the
    gap is large enough that small input perturbations (frame-to-frame
    noise, viewing angle changes) can't flip the prediction.

    This is *only* the loss function from Carlini & Wagner — the L2
    minimisation / binary search machinery is not used because we have a
    fixed patch footprint and don't care about perturbation norm.
    """
    B, C = logits.shape
    # Mask out the target class to find the best non-target logit
    one_hot  = F.one_hot(target, num_classes=C).bool()
    z_target = logits[one_hot].view(B)                              # (B,)
    z_other  = logits.masked_fill(one_hot, -65000.0).max(dim=1).values  # (B,)  # fits float16
    return F.relu(z_other - z_target + margin).mean()


def optimise_patch(
    models:       list,
    dataset:      torch.utils.data.Dataset,
    target_label: int,
    patch_size:   int   = 945,
    steps:        int   = 2000,
    lr:           float = 0.01,
    eot_samples:  int   = 16,
    batch_size:   int   = 32,
    device:       torch.device = None,
    universal:    bool  = True,
    print_cm:     float = 8.0,
    sign_diam_mm: float = 450.0,
    nps_weight:   float = 0.01,
    tv_weight:    float = 0.05,
    cx_min:       float = 0.62,
    cx_max:       float = 0.84,
    cy_min:       float = 0.35,
    cy_max:       float = 0.58,
    fake_quant_schedule = None,   # FakeQuantSchedule | None — flipped on after warmup
    fake_quant_warmup: int = 200,
    loss_fn:      str   = "ce",    # "ce" | "margin"
    cw_margin:    float = 10.0,
    oblique_eot:  bool  = False,
    init_patch:   str   = None,   # path to a saved patch tensor to resume from
    ensemble_weights: list = None, # per-model sampling weights (default: uniform)
    quant_eval_models: list = None, # PTQ INT8 models for eval ASR measurement
    guided_init:  bool  = False,   # ZQBA guided backprop init (paper 2510.00769)
    noise_sigma:  float = 0.0,     # Gaussian noise σ for gradient smoothing (ZQ-Attack)
    sequential_ensemble: bool = False, # sequential ensemble optimisation (ZQ-Attack)
    eval_every:   int   = 100,     # held-out eval interval (steps)
    night_mode:   bool  = False,   # bias EOT toward night conditions
    semi_warmup:  int   = 150,     # steps of untargeted warmup for semi-targeted mode
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

    patch_mm        = print_cm * 10.0
    sign_input_px   = int(224 * 0.80)                  # ~179px sign in 224px input
    target_patch_px = max(4, int(sign_input_px * patch_mm / sign_diam_mm))
    print(f"Sign diameter   : {sign_diam_mm}mm")
    print(f"Print-res patch : {patch_size}px  ({print_cm}cm @ 300 DPI)")
    print(f"Model footprint : {target_patch_px}px  in 224px input  "
          f"({patch_mm:.0f}mm patch on {sign_diam_mm:.0f}mm sign = {patch_mm/sign_diam_mm:.1%} coverage)")
    print(f"Loss function   : {loss_fn}" + (f"  (margin={cw_margin})" if loss_fn == "margin" else ""))
    print(f"Ensemble size   : {len(models)} model(s): "
          f"{[getattr(m, '_arch_name', '?') for m in models]}")

    for m in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # ── MPS performance: float16 autocast ────────────────────────────────
    # Apple Silicon's ANE/GPU throughput roughly doubles in float16.
    # The patch itself stays float32 (we need the precision for small
    # gradient updates), but model forward passes and EOT transforms
    # run in float16 via autocast.
    _use_autocast = (device.type == "mps")
    if _use_autocast:
        print("MPS detected — enabling float16 autocast for forward passes")

    if init_patch is not None:
        print(f"Initialising patch from: {init_patch}")
        patch_01 = torch.load(init_patch, map_location=device)
        if patch_01.dim() == 3:
            patch_01 = patch_01.unsqueeze(0)
        if patch_01.shape[-1] != patch_size:
            patch_01 = F.interpolate(patch_01, size=(patch_size, patch_size),
                                     mode="bilinear", align_corners=False)
        patch_01 = patch_01.clamp(0, 1).to(device)
    else:
        patch_01 = torch.rand(1, 3, patch_size, patch_size, device=device) * 0.5 + 0.25

    target_t = torch.tensor([target_label], device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def to_normalised(p_01):
        return (p_01 - mean) / std

    # ── ZQBA guided backprop init (paper 2510.00769) ─────────────────
    # Single-step guided backpropagation: X_adv = α·(∇_X·f(X)) + X
    # Provides a much better starting point than random noise by using
    # the gradient of the target loss w.r.t. the input to initialise
    # the patch in the direction that most increases target confidence.
    if guided_init and init_patch is None:
        print("Running ZQBA guided backprop init (α=0.4)...")
        patch_01.requires_grad_(True)
        _init_loss = torch.tensor(0.0, device=device)
        _n_init = min(4, len(dataset))
        _init_loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(batch_size, 16), shuffle=True, num_workers=0,
        )
        _init_imgs, _init_labels = next(iter(_init_loader))
        _init_imgs = _init_imgs.to(device)
        _init_labels = _init_labels.to(device)
        _init_patch_norm = to_normalised(patch_01)
        _init_patched = apply_patch(_init_imgs, _init_patch_norm,
                                     target_patch_px=target_patch_px)
        # Average gradient across all surrogates
        for _m in models:
            _logits = _m(_init_patched)
            if loss_fn in ("untargeted", "semi-targeted"):
                _init_loss = _init_loss + untargeted_loss(_logits, _init_labels)
            else:
                _init_loss = _init_loss + F.cross_entropy(
                    _logits, target_t.expand(_init_imgs.size(0)))
        _init_loss = _init_loss / len(models)
        _init_loss.backward()
        # ZQBA formula: X_adv = α·(∇_X·f(X)) + X, α = 0.4
        _alpha = 0.4
        with torch.no_grad():
            _grad = patch_01.grad
            # Guided backprop: keep only positive gradients × positive activations
            _guided_grad = F.relu(_grad) * F.relu(patch_01)
            if _guided_grad.abs().max() > 1e-8:
                _guided_grad = _guided_grad / _guided_grad.abs().max()
            patch_01.data = (patch_01.data - _alpha * _guided_grad).clamp(0, 1)
            patch_01.grad = None
        print(f"  Init loss: {_init_loss.item():.4f}")

    patch_01.requires_grad_(True)

    optimizer = torch.optim.Adam([patch_01], lr=lr)
    # Warm restarts prevent late-run collapse: cosine decay to near-zero LR
    # while EOT variance stays fixed causes the patch to drift on noise once
    # the margin loss gradients shrink at high ASR.  Restarting every T_0 steps
    # spikes the LR back up so the optimiser can escape bad basins.
    restart_period = max(200, steps // 5)   # ~5 restarts per run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_period, T_mult=1,
    )

    sub_ds = dataset if universal else torch.utils.data.Subset(
        dataset, [i for i, (_, l) in enumerate(dataset) if l != target_label]
    )

    loader = torch.utils.data.DataLoader(
        sub_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    data_iter = iter(loader)

    # ── Held-out eval batch (fixed seed for comparability across steps) ──
    # Sample once, reuse every eval — same images, same EOT seed each time,
    # so ASR numbers are directly comparable step-to-step.
    _eval_rng = torch.Generator()
    _eval_rng.manual_seed(42)
    _eval_indices = torch.randperm(len(sub_ds), generator=_eval_rng)[:batch_size].tolist()
    _eval_subset = torch.utils.data.Subset(sub_ds, _eval_indices)
    _eval_loader = torch.utils.data.DataLoader(
        _eval_subset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    _eval_imgs, _eval_labels = next(iter(_eval_loader))
    _eval_imgs = _eval_imgs.to(device)
    _eval_labels = _eval_labels.to(device)
    _eval_B = _eval_imgs.size(0)
    print(f"Held-out eval : {_eval_B} images, evaluated every {eval_every} steps")

    best_asr   = -1.0
    best_patch = patch_01.detach().clone()

    # ── Semi-targeted state ──────────────────────────────────────────────
    # During warmup: act as untargeted.  After warmup: probe the ensemble
    # to find which non-true class dominates, then switch to targeted
    # margin loss for that class — concentrating votes for temporal voting.
    _semi_locked_target = None   # once set, an int class index
    _effective_loss_fn  = loss_fn
    if loss_fn == "semi-targeted":
        _effective_loss_fn = "untargeted"
        print(f"Semi-targeted : untargeted warmup for {semi_warmup} steps, then lock target")

    pbar = tqdm(range(1, steps + 1), desc="Optimising patch")
    for step in pbar:
        # ── Semi-targeted: switch phase after warmup ─────────────────
        if (loss_fn == "semi-targeted" and _semi_locked_target is None
                and step == semi_warmup + 1):
            # Probe which class the ensemble prefers
            _probe_class = semi_targeted_probe(
                models, _eval_imgs, _eval_labels, patch_01,
                to_normalised, eot_print, eot_scene, apply_patch,
                make_patch_mask, target_patch_px,
                cx_min, cx_max, cy_min, cy_max,
                mean, std, device, n_probes=16,
                oblique_eot=oblique_eot, night_mode=night_mode,
                _use_autocast=_use_autocast, IMG_SIZE=IMG_SIZE,
            )
            if _probe_class is not None:
                _semi_locked_target = _probe_class
                _effective_loss_fn  = "margin"
                target_label        = _probe_class
                target_t            = torch.tensor([_probe_class], device=device)
                pbar.write(
                    f"[step {step}] ★ Semi-targeted: locking onto "
                    f"{ALL_SPEEDS[_probe_class]} km/h (label {_probe_class}) "
                    f"— switching to margin loss")
            else:
                pbar.write(
                    f"[step {step}] Semi-targeted: no dominant wrong class found, "
                    f"continuing untargeted")

        if (fake_quant_schedule is not None and not fake_quant_schedule.enabled
                and step >= fake_quant_warmup):
            fake_quant_schedule.enabled = True
            pbar.write(f"[step {step}] fake-quant hooks enabled — "
                       f"p ramps 0.05 → {fake_quant_schedule.p_max:.2f} "
                       f"over {fake_quant_schedule.ramp_steps} steps")

        # Advance the fake-quant p_apply ramp each step
        if fake_quant_schedule is not None and fake_quant_schedule.enabled:
            fake_quant_schedule.step()

        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, labels = next(data_iter)

        imgs   = imgs.to(device)
        labels = labels.to(device)
        B      = imgs.size(0)

        with torch.no_grad():
            patch_01.clamp_(0.0, 1.0)

        patch_norm = to_normalised(patch_01)
        total_loss = torch.tensor(0.0, device=device)

        if sequential_ensemble:
            # ── Sequential ensemble optimisation (ZQ-Attack 2406.19311) ──
            # Each model incorporates predecessors' gradients:
            #   δ_j = δ_0 - α·(1/j)·Σ∇_δ' L(x, δ'+σ, t, f_j)
            # This ensures each surrogate refines rather than overrides the
            # previous models' perturbation direction.
            #
            # Perf: EOT transforms (print→mount→geometry→lighting→camera) are
            # shared across models within each EOT draw — only the model forward
            # varies.  This saves ~2/3 of the transform overhead.
            for eot_i in range(eot_samples):
                cx = random.uniform(cx_min, cx_max)
                cy = random.uniform(cy_min, cy_max)

                # Compute shared EOT scene ONCE per draw
                patch_01_printed = eot_print(patch_01.clamp(0, 1))
                if noise_sigma > 0:
                    noise = torch.randn_like(patch_01_printed) * noise_sigma
                    patch_01_printed = _ste_clamp_01(patch_01_printed + noise)
                patch_norm_eot = to_normalised(patch_01_printed)
                patched = apply_patch(imgs, patch_norm_eot, cx_frac=cx, cy_frac=cy,
                                      randomise_placement=False,
                                      target_patch_px=target_patch_px)
                p_mask = make_patch_mask(B, IMG_SIZE, IMG_SIZE, target_patch_px,
                                        cx, cy, device)
                patched_01 = patched * std + mean
                patched_01 = eot_scene(patched_01.clone(), oblique=oblique_eot,
                                        patch_mask=p_mask, training_step=step,
                                        night_mode=night_mode)
                patched_eot = (patched_01 - mean) / std

                # Run each model on the SAME transformed input
                for mi, surrogate in enumerate(models):
                    with torch.autocast("mps", dtype=torch.float16, enabled=_use_autocast):
                        logits = surrogate(patched_eot)
                    if _effective_loss_fn == "untargeted":
                        loss = untargeted_loss(logits, labels, margin=cw_margin)
                    elif _effective_loss_fn == "margin":
                        loss = margin_loss(logits, target_t.expand(B), margin=cw_margin)
                    else:
                        loss = F.cross_entropy(logits, target_t.expand(B))
                    # Weight: 1/(j+1) so later models refine rather than dominate
                    w = 1.0 / (mi + 1)
                    total_loss = total_loss + w * loss

            total_loss = total_loss / (eot_samples * sum(1.0 / (i + 1) for i in range(len(models))))
        else:
            # ── Original random surrogate sampling ──────────────────────
            for _ in range(eot_samples):
                # Sample one surrogate per EOT step — prevents the patch from
                # exploiting any single model's blind spots.
                surrogate = random.choices(models, weights=ensemble_weights, k=1)[0]
                cx = random.uniform(cx_min, cx_max)
                cy = random.uniform(cy_min, cy_max)

                patch_01_printed = eot_print(patch_01.clamp(0, 1))

                # Gaussian noise injection for gradient smoothing
                if noise_sigma > 0:
                    noise = torch.randn_like(patch_01_printed) * noise_sigma
                    patch_01_printed = _ste_clamp_01(patch_01_printed + noise)

                patch_norm_eot = to_normalised(patch_01_printed)
                patched = apply_patch(imgs, patch_norm_eot, cx_frac=cx, cy_frac=cy,
                                      randomise_placement=False,
                                      target_patch_px=target_patch_px)
                p_mask = make_patch_mask(B, IMG_SIZE, IMG_SIZE, target_patch_px,
                                        cx, cy, device)
                patched_01 = patched * std + mean
                patched_01 = eot_scene(patched_01.clone(), oblique=oblique_eot,
                                        patch_mask=p_mask, training_step=step,
                                        night_mode=night_mode)
                patched_eot = (patched_01 - mean) / std

                with torch.autocast("mps", dtype=torch.float16, enabled=_use_autocast):
                    logits = surrogate(patched_eot)
                if _effective_loss_fn == "untargeted":
                    loss = untargeted_loss(logits, labels, margin=cw_margin)
                elif _effective_loss_fn == "margin":
                    loss = margin_loss(logits, target_t.expand(B), margin=cw_margin)
                else:
                    loss = F.cross_entropy(logits, target_t.expand(B))
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
        torch.nn.utils.clip_grad_norm_([patch_01], max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # Log LR restart events so we can correlate with ASR recovery
        cur_lr = optimizer.param_groups[0]["lr"]
        if step > 1 and step % restart_period == 0:
            print(f"  [step {step}] LR warm restart → {cur_lr:.6f}")

        # Flush MPS allocator periodically to prevent memory fragmentation.
        # Every 50 steps is enough — every 5 was causing GPU pipeline stalls
        # from forced synchronisation (~2-3s overhead per flush).
        if device.type == "mps" and step % 50 == 0:
            torch.mps.empty_cache()

        with torch.no_grad():
            patch_01.clamp_(0.0, 1.0)

        if step % eval_every == 0:
            # Disable fake-quant during eval so ASR reflects actual patch
            # quality, not stochastic quant noise on this particular draw
            _fq_was_enabled = (fake_quant_schedule is not None
                               and fake_quant_schedule.enabled)
            if _fq_was_enabled:
                fake_quant_schedule.enabled = False

            with torch.no_grad():
                # Use the held-out eval batch with a fixed EOT seed so that
                # ASR numbers are directly comparable across steps.  The same
                # images + same random placements every eval = no noise from
                # batch sampling.
                _n_eval_eot = 4
                _per_model_hits = [0] * len(models)
                _eval_state = random.getstate()  # save RNG state
                random.seed(step * 31337)        # deterministic EOT per step but varied across steps
                for _ei in range(_n_eval_eot):
                    patch_printed   = eot_print(patch_01.detach().clamp(0, 1))
                    patch_norm_det  = to_normalised(patch_printed)
                    cx_log = random.uniform(cx_min, cx_max)
                    cy_log = random.uniform(cy_min, cy_max)
                    patched_log     = apply_patch(_eval_imgs, patch_norm_det,
                                                  cx_frac=cx_log, cy_frac=cy_log,
                                                  randomise_placement=False,
                                                  target_patch_px=target_patch_px)
                    p_mask_log      = make_patch_mask(_eval_B, IMG_SIZE, IMG_SIZE,
                                                      target_patch_px, cx_log, cy_log,
                                                      device)
                    patched_01_log  = patched_log * std + mean
                    patched_01_log  = eot_scene(patched_01_log, oblique=oblique_eot,
                                                patch_mask=p_mask_log,
                                                training_step=step,
                                                night_mode=night_mode)
                    patched_eot_log = (patched_01_log - mean) / std
                    for mi, m in enumerate(models):
                        preds = m(patched_eot_log).argmax(1)
                        if _effective_loss_fn == "untargeted":
                            _per_model_hits[mi] += (preds != _eval_labels).sum().item()
                        else:
                            _per_model_hits[mi] += (preds == target_label).sum().item()
                random.setstate(_eval_state)     # restore RNG so training isn't disturbed
                _per_model_total = _eval_B * _n_eval_eot
                # Use MEAN across models, not MIN — min-of-3 is too pessimistic
                # during training and makes the progress bar useless.  Final eval
                # still uses min (worst-case) for the real number.
                _per_model_asr = [h / _per_model_total for h in _per_model_hits]
                asr = sum(_per_model_asr) / len(_per_model_asr)

                # Per-model ASR string for tqdm — shows which model is the bottleneck
                arch_names = [getattr(m, '_arch_name', f'm{i}')[:6] for i, m in enumerate(models)]
                per_model_str = " ".join(f"{n}={a:.0%}" for n, a in zip(arch_names, _per_model_asr))

                # Also measure with deterministic INT8 fake-quant if enabled
                quant_asr_str = ""
                if quant_eval_models:
                    _qh = {id(m): _build_quant_eval_hooks(m) for m in quant_eval_models}
                    _q_hits = [0] * len(quant_eval_models)
                    random.seed(step * 31337 + 1)  # same placements as float eval
                    for _ei in range(_n_eval_eot):
                        patch_printed   = eot_print(patch_01.detach().clamp(0, 1))
                        patch_norm_det  = to_normalised(patch_printed)
                        cx_log = random.uniform(cx_min, cx_max)
                        cy_log = random.uniform(cy_min, cy_max)
                        patched_log     = apply_patch(_eval_imgs, patch_norm_det,
                                                      cx_frac=cx_log, cy_frac=cy_log,
                                                      randomise_placement=False,
                                                      target_patch_px=target_patch_px)
                        p_mask_log      = make_patch_mask(_eval_B, IMG_SIZE, IMG_SIZE,
                                                          target_patch_px, cx_log, cy_log,
                                                          device)
                        patched_01_log  = patched_log * std + mean
                        patched_01_log  = eot_scene(patched_01_log, oblique=oblique_eot,
                                                    patch_mask=p_mask_log,
                                                    training_step=step,
                                                    night_mode=night_mode)
                        patched_eot_log = (patched_01_log - mean) / std
                        for mi, m in enumerate(quant_eval_models):
                            preds_q = m(patched_eot_log).argmax(1)
                            if _effective_loss_fn == "untargeted":
                                _q_hits[mi] += (preds_q != _eval_labels).sum().item()
                            else:
                                _q_hits[mi] += (preds_q == target_label).sum().item()
                    random.setstate(_eval_state)
                    for m in quant_eval_models:
                        for h in _qh[id(m)]:
                            h.remove()
                    quant_asr = sum(h / _per_model_total for h in _q_hits) / len(_q_hits)
                    quant_asr_str = f"  qASR={quant_asr:.2%}"

            # ── save best patch by mean-ASR on held-out batch ─────────
            if asr > best_asr:
                best_asr   = asr
                best_patch = patch_01.detach().clamp(0, 1).clone()
                _best_step = step
                _best_detail = per_model_str
                pbar.write(f"[step {step}] ★ new best eval ASR={asr:.2%}  ({per_model_str})")
                # Save best patch checkpoint immediately
                _inter_dir = Path("intermediate_examples")
                _inter_dir.mkdir(exist_ok=True)
                _best_arr = best_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
                Image.fromarray((_best_arr * 255).astype(np.uint8)).save(
                    _inter_dir / "patch_best.png")
                torch.save(best_patch, _inter_dir / "patch_best.pt")

            if _fq_was_enabled:
                fake_quant_schedule.enabled = True
            pbar.set_postfix(loss=f"{total_loss.item():.4f}",
                             ASR=f"{asr:.2%}{quant_asr_str}",
                             best=f"{best_asr:.2%}@{_best_step}",
                             detail=per_model_str)

            _inter_dir = Path("intermediate_examples")
            _inter_dir.mkdir(exist_ok=True)
            arr = patch_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((arr * 255).astype(np.uint8)).save(_inter_dir / f"patch_step_{step:04d}.png")

    print(f"\nBest patch from step {_best_step}: ASR={best_asr:.2%}  ({_best_detail})")
    if _semi_locked_target is not None:
        print(f"Semi-targeted locked class: {ALL_SPEEDS[_semi_locked_target]} km/h "
              f"(label {_semi_locked_target})")
    return best_patch, target_patch_px, target_label


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
    cx_min:          float = 0.62,
    cx_max:          float = 0.84,
    cy_min:          float = 0.35,
    cy_max:          float = 0.58,
    oblique_eot:     bool  = False,
    loss_fn:         str   = "ce",
    night_mode:      bool  = False,
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
                    cx = random.uniform(cx_min, cx_max)
                    cy = random.uniform(cy_min, cy_max)
                    # Print EOT on patch, then composite, then scene EOT
                    patch_printed = eot_print(patch_01.to(device))
                    patch_norm_ev = (patch_printed - mean) / std
                    patched     = apply_patch(imgs, patch_norm_ev, cx_frac=cx, cy_frac=cy,
                                              randomise_placement=False,
                                              target_patch_px=target_patch_px)
                    p_mask_ev   = make_patch_mask(B, IMG_SIZE, IMG_SIZE,
                                                  target_patch_px, cx, cy, device)
                    patched_01  = patched * std + mean
                    patched_01  = eot_scene(patched_01, oblique=oblique_eot,
                                            patch_mask=p_mask_ev,
                                            night_mode=night_mode)
                    patched_eot = (patched_01 - mean) / std
                    preds       = m(patched_eot).argmax(1)
                    if loss_fn in ("untargeted",):
                        eot_votes += (preds != labels).long()
                    else:
                        eot_votes += (preds == target_label).long()

                correct_targets[mi] += (eot_votes >= (n_eot // 2 + 1)).sum().item()
                totals[mi]          += B

    if loss_fn == "untargeted":
        mode_str = "misclassify"
    elif loss_fn == "semi-targeted":
        mode_str = f"semi-targeted@{ALL_SPEEDS[target_label]} km/h"
    else:
        mode_str = f"target={ALL_SPEEDS[target_label]} km/h"
    print(f"\n── Patch Evaluation (EOT + random placement, {mode_str}) ──")
    arch_names = [getattr(m, '_arch_name', f'model_{i}') for i, m in enumerate(models)]
    for mi, name in enumerate(arch_names):
        T = totals[mi]
        print(f"  [{name:22s}]  clean={orig_correct[mi]/T:.2%}  "
              f"ASR={correct_targets[mi]/T:.2%}")

    worst_asr = min(correct_targets[mi] / totals[mi] for mi in range(n_models))
    print(f"  Worst-case (ensemble) ASR : {worst_asr:.2%}  "
          f"({mode_str}, majority vote {n_eot} EOT)")


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
        # First arch used to always get args.model directly, but that
        # assumes the first arch is resnet18 (the default checkpoint).
        # Now: resnet18 → args.model; everything else → derived path.
        if arch == "resnet18":
            arch_path = args.model
        else:
            arch_path = str(Path(args.model).parent / f"{stem}_{arch}{suffix}")
        ensemble.append(_load_or_build(arch_path, arch, args.pretrain_path, device))

    print(f"Ensemble: {[m._arch_name for m in ensemble]}")

    fake_quant_schedule = None
    if not args.no_fake_quant:
        bits_choices = tuple(int(b) for b in args.fake_quant_bits.split(","))
        fake_quant_schedule = FakeQuantSchedule(
            enabled=False,
            p_max=args.fake_quant_p,
            ramp_steps=args.fake_quant_ramp,
        )
        print(f"Fake-quant EOT : bits={bits_choices}  p_max={args.fake_quant_p}  "
              f"warmup={args.fake_quant_warmup} steps  "
              f"ramp=0.05→{args.fake_quant_p} over {args.fake_quant_ramp} steps  "
              f"(eval runs clean — no quant noise in ASR measurement)")
        for m in ensemble:
            handles, _ = install_fake_quant_hooks(
                m, bits_choices=bits_choices, p_apply=args.fake_quant_p,
                schedule=fake_quant_schedule,
            )
            print(f"  {m._arch_name}: hooked {len(handles)} Conv2d/Linear layers")

    from torch.utils.data import ConcatDataset
    gtsrb_val = FilteredGTSRB(args.data, split="test", transform=val_transforms, download=False)
    aus_val   = AUSynthDataset(args.aus_data, transform=val_transforms, split="val", auto_generate=False)
    val_ds    = ConcatDataset([gtsrb_val, aus_val])
    print(f"Val samples: {len(val_ds)}  (GTSRB={len(gtsrb_val)}, AU synth={len(aus_val)})")

    target_label = KMH_TO_LABEL[args.target]
    print(f"Target: {args.target} km/h  (label {target_label})")

    # ── Prepare deterministic INT8 fake-quant hooks for eval ────────────────
    # These are installed/removed around eval passes so the same float models
    # can be used for both training (stochastic fake-quant via schedule) and
    # eval (deterministic full-layer quant).  No separate model copies needed.
    quant_eval_models = None
    if not args.no_quant_eval:
        quant_eval_models = ensemble  # same models, hooks toggled at eval time
        print(f"Quant eval: deterministic INT8 fake-quant on all {len(ensemble)} "
              f"surrogates during eval passes")

    patch_01, target_patch_px, target_label = optimise_patch(
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
        sign_diam_mm = args.sign_diam_mm,
        nps_weight   = args.nps_weight,
        tv_weight    = args.tv_weight,
        cx_min       = args.cx_min,
        cx_max       = args.cx_max,
        cy_min       = args.cy_min,
        cy_max       = args.cy_max,
        fake_quant_schedule = fake_quant_schedule,
        fake_quant_warmup   = args.fake_quant_warmup,
        loss_fn             = args.loss,
        cw_margin           = args.margin,
        oblique_eot         = args.oblique_eot,
        init_patch          = args.init_patch,
        ensemble_weights    = [float(w) for w in args.ensemble_weights.split(",")]
                              if args.ensemble_weights else None,
        quant_eval_models   = quant_eval_models,
        guided_init         = args.guided_init,
        noise_sigma         = args.noise_sigma,
        sequential_ensemble = args.sequential_ensemble,
        eval_every          = args.eval_every,
        night_mode          = args.night_mode if hasattr(args, 'night_mode') else False,
        semi_warmup         = args.semi_warmup if hasattr(args, 'semi_warmup') else 150,
    )

    torch.save(patch_01, args.out)
    print(f"Patch tensor saved: {args.out}")

    png_path = Path(args.out).with_suffix(".png")
    save_patch_png(patch_01.cpu(), str(png_path), print_cm=args.print_cm)

    # Final eval on float surrogates (fake-quant disabled for clean measurement)
    if fake_quant_schedule is not None:
        fake_quant_schedule.enabled = False
    print("\n── Float32 surrogate eval ──")
    evaluate_patch(ensemble, val_ds, patch_01, target_label,
                   target_patch_px=target_patch_px, device=device, n_eot=args.eot_samples,
                   cx_min=args.cx_min, cx_max=args.cx_max,
                   cy_min=args.cy_min, cy_max=args.cy_max,
                   oblique_eot=args.oblique_eot,
                   loss_fn=args.loss,
                   night_mode=args.night_mode)

    # Final eval with deterministic INT8 fake-quant + sanity gate
    if quant_eval_models:
        print("\n── INT8 eval (deterministic per-tensor symmetric quant) ──")

        # Calibration loader — 100 clean samples for per-layer stats
        _cal_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=32, shuffle=False, num_workers=0,
        )
        _cal_subset = []
        _cal_count = 0
        for _cb, _ in _cal_loader:
            _cal_subset.append((_cb, _))
            _cal_count += _cb.size(0)
            if _cal_count >= 100:
                break

        # Sanity gate: check that INT8 clean accuracy doesn't collapse
        # compared to float32.  If it drops >15pp, the fake-quant is destroying
        # that surrogate's representations and the ASR number is meaningless.
        _valid_quant_models = []
        for m in quant_eval_models:
            arch_name = getattr(m, '_arch_name', '?')
            # Measure float32 clean accuracy
            _f32_correct = _f32_total = 0
            with torch.no_grad():
                for _cb, _cl in _cal_subset:
                    _cb, _cl = _cb.to(device), _cl.to(device)
                    _f32_correct += (m(_cb).argmax(1) == _cl).sum().item()
                    _f32_total += _cb.size(0)
            _f32_acc = _f32_correct / max(_f32_total, 1)

            # Install INT8 hooks with calibration
            _qh_test = _build_quant_eval_hooks(m, calibration_loader=_cal_loader)
            _q8_correct = _q8_total = 0
            with torch.no_grad():
                for _cb, _cl in _cal_subset:
                    _cb, _cl = _cb.to(device), _cl.to(device)
                    _q8_correct += (m(_cb).argmax(1) == _cl).sum().item()
                    _q8_total += _cb.size(0)
            _q8_acc = _q8_correct / max(_q8_total, 1)
            for h in _qh_test:
                h.remove()

            _drop = _f32_acc - _q8_acc
            if _drop > 0.15:
                print(f"  ⚠ INT8 eval INVALID for {arch_name}: "
                      f"clean accuracy collapsed {_f32_acc:.1%} → {_q8_acc:.1%} "
                      f"(drop={_drop:.1%} > 15% threshold). "
                      f"Per-tensor symmetric quant destroys this architecture's activations. "
                      f"Skipping from INT8 ASR measurement.")
            else:
                print(f"  ✓ {arch_name}: INT8 clean acc {_q8_acc:.1%} "
                      f"(float32: {_f32_acc:.1%}, drop={_drop:.1%})")
                _valid_quant_models.append(m)

        if _valid_quant_models:
            _qh = {id(m): _build_quant_eval_hooks(m, calibration_loader=_cal_loader)
                   for m in _valid_quant_models}
            evaluate_patch(_valid_quant_models, val_ds, patch_01, target_label,
                           target_patch_px=target_patch_px, device=device,
                           n_eot=args.eot_samples,
                           cx_min=args.cx_min, cx_max=args.cx_max,
                           cy_min=args.cy_min, cy_max=args.cy_max,
                           oblique_eot=args.oblique_eot,
                           loss_fn=args.loss,
                           night_mode=args.night_mode)
            for m in _valid_quant_models:
                for h in _qh[id(m)]:
                    h.remove()
        else:
            print("  ⚠ No surrogates passed INT8 sanity gate — skipping INT8 eval entirely.")


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
    p.add_argument("--patch-size",    type=int,   default=945,  help="Patch pixel size (print res)")
    p.add_argument("--steps",         type=int,   default=2000)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--eot-samples",   type=int,   default=8,
                   help="EOT samples per step (lower = less gradient variance, faster convergence)")

    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--eval-every",    type=int,   default=100,
                   help="Run held-out eval every N steps (fixed batch, comparable across steps). "
                        "Best patch by mean eval ASR is saved and returned.")
    p.add_argument("--print-cm",      type=float, default=8.0,  help="Printed patch size in cm")
    p.add_argument("--sign-diam-mm",  type=float, default=450.0,
                   help="Physical sign diameter in mm (AU=450, EU=190). Controls patch "
                        "footprint calculation in model input space.")
    p.add_argument("--nps-weight",    type=float, default=0.01, help="Printability loss weight (0 to disable)")
    p.add_argument("--tv-weight",     type=float, default=0.05, help="Total variation loss weight (0 to disable)")
    # Model-side quantisation EOT — hooks each surrogate's Conv2d/Linear layers
    # so the optimiser sees int-N activation rounding, approximating an
    # embedded NPU without needing the real target's weights/calibration.
    p.add_argument("--init-patch",    default=None,
                   help="Path to a saved patch tensor (.pt) to resume from instead of random init")
    p.add_argument("--loss",          default="ce", choices=["ce", "margin", "untargeted", "semi-targeted"],
                   help="Loss function: 'ce' (cross-entropy on target), 'margin' (CW-style margin "
                        "on target), 'untargeted' (push away from true class — any misclassification wins), "
                        "'semi-targeted' (untargeted warmup, then lock onto the most-predicted wrong class)")
    p.add_argument("--semi-warmup",   type=int, default=150,
                   help="Steps of untargeted warmup before locking target in semi-targeted mode")
    p.add_argument("--margin",        type=float, default=5.0,
                   help="Margin for CW-style loss (logit gap target must exceed runner-up by)")
    p.add_argument("--no-fake-quant", action="store_true", default=False,
                   help="Disable fake-quant activation hooks (enabled by default)")
    p.add_argument("--no-quant-eval", action="store_true", default=False,
                   help="Skip building PTQ INT8 eval models (saves ~30s startup)")
    p.add_argument("--fake-quant-bits", default="6,8",
                   help="Comma-separated bit-widths to sample per layer per forward")
    p.add_argument("--fake-quant-p",  type=float, default=0.15,
                   help="Max probability a given layer is fake-quantised on a given forward "
                        "(ramps from 0.05 over --fake-quant-ramp steps)")
    p.add_argument("--fake-quant-warmup", type=int, default=200,
                   help="Steps of clean (unquantised) optimisation before enabling the hooks")
    p.add_argument("--fake-quant-ramp", type=int, default=500,
                   help="Steps over which p_apply linearly ramps from 0.05 to --fake-quant-p after warmup")
    # Placement range — train only over the off-centre region where the patch
    # will physically appear.  Defaults cover +60mm right / ±10mm vertical
    # with ±25mm human placement error.  cx/cy are fractions of image width/height
    # (0.5 = sign centre; 0.75 ≈ +60mm right on a 190mm sign at 224px input).
    p.add_argument("--ensemble-weights", default=None,
                   help="Comma-separated sampling weights per arch, e.g. '1,2,1' "
                        "to sample the 2nd arch twice as often.  Default: uniform.")
    p.add_argument("--cx-min",        type=float, default=0.62,
                   help="Min patch centre X fraction (0=left edge, 0.5=sign centre)")
    p.add_argument("--cx-max",        type=float, default=0.84,
                   help="Max patch centre X fraction")
    p.add_argument("--oblique-eot",   action="store_true", default=False,
                   help="Enable oblique (off-axis 45-80°) perspective transforms in EOT")
    p.add_argument("--cy-min",        type=float, default=0.35,
                   help="Min patch centre Y fraction (0=top, 0.5=sign centre)")
    p.add_argument("--cy-max",        type=float, default=0.58,
                   help="Max patch centre Y fraction")
    # Paper-derived techniques
    p.add_argument("--guided-init",   action="store_true", default=False,
                   help="ZQBA guided backprop init (paper 2510.00769) — uses "
                        "single-step gradient to initialise patch instead of random noise")
    p.add_argument("--noise-sigma",   type=float, default=0.0,
                   help="Gaussian noise σ for gradient smoothing (ZQ-Attack paper). "
                        "0 = disabled, try 0.01-0.05 for smoother gradients.")
    p.add_argument("--sequential-ensemble", action="store_true", default=False,
                   help="Sequential ensemble optimisation (ZQ-Attack 2406.19311) — "
                        "each model incorporates predecessors' gradients with 1/j weighting")
    p.add_argument("--night-mode",    action="store_true", default=False,
                   help="Bias EOT toward night conditions: stronger retroreflection (0.3-0.6), "
                        "darker brightness, warm (halogen) or cool (LED) colour temperature. "
                        "Matches physical testing under headlights / carpark lighting.")
    main(p.parse_args())
