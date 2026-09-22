"""
Expectation over Transformation (EOT) — physical-world augmentations applied
during patch optimisation to improve real-world transfer.

Transforms are grouped and applied in physical order:
  1. PRINT  — what the inkjet/laser printer does to the patch design
  2. MOUNT  — paper curl from blu-tack mounting on a non-flat surface
  3. GEOMETRY — projective warp from different viewing angles while driving
  4. LIGHTING — ambient + point-source illumination variation
  5. CAMERA  — sensor noise, motion blur, compression artifacts
  6. SENSOR  — grayscale conversion + CLAHE (Mobileye S-Cam4 pipeline)

All ops are differentiable so gradients flow back into the patch.
"""
import math
import random

import torch
import torch.nn.functional as F


def _rand(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — PRINT ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def apply_gamut_compression(x: torch.Tensor) -> torch.Tensor:
    factors = torch.tensor([0.82, 0.90, 0.88], device=x.device).view(1, 3, 1, 1)
    offsets = torch.tensor([0.04, 0.02, 0.02], device=x.device).view(1, 3, 1, 1)
    factors = factors + torch.randn_like(factors) * 0.03
    x = x * factors + offsets
    return torch.clamp(x, 0.0, 1.0)


def apply_cmyk_roundtrip(x: torch.Tensor) -> torch.Tensor:
    c = 1.0 - x[:, 0:1]
    m = 1.0 - x[:, 1:2]
    y = 1.0 - x[:, 2:3]
    k = torch.min(torch.cat([c, m, y], dim=1), dim=1, keepdim=True).values
    k_amount = _rand(0.05, 0.20)
    k = k * k_amount
    r = torch.clamp(1.0 - c - k, 0, 1)
    g = torch.clamp(1.0 - m - k, 0, 1)
    b = torch.clamp(1.0 - y - k, 0, 1)
    return torch.cat([r, g, b], dim=1)


def apply_dot_gain(x: torch.Tensor) -> torch.Tensor:
    gain = _rand(0.08, 0.22)
    x = x - gain * 4.0 * x * (1.0 - x)
    return torch.clamp(x, 0.0, 1.0)


def apply_channel_misregistration(x: torch.Tensor) -> torch.Tensor:
    shift_r = random.choice([-2, -1, 0, 1, 2])
    shift_b = random.choice([-2, -1, 0, 1, 2])
    if shift_r == 0 and shift_b == 0:
        return x
    out = x.clone()
    if shift_r != 0:
        out[:, 0] = torch.roll(x[:, 0], shifts=shift_r, dims=-1)
    if shift_b != 0:
        out[:, 2] = torch.roll(x[:, 2], shifts=shift_b, dims=-1)
    return out


def apply_print_banding(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    band_period = random.randint(8, 24)
    band_amp    = _rand(0.00, 0.04)
    rows    = torch.arange(H, device=x.device, dtype=x.dtype)
    banding = 1.0 + band_amp * torch.sin(2 * math.pi * rows / band_period)
    banding = banding.view(1, 1, H, 1).expand(B, C, H, W)
    return torch.clamp(x * banding, 0.0, 1.0)


def apply_paper_texture(x: torch.Tensor) -> torch.Tensor:
    texture_std = _rand(0.0, 0.03)
    texture = 1.0 + torch.randn_like(x) * texture_std
    weight  = x.detach().mean(dim=1, keepdim=True).expand_as(x)
    combined = 1.0 + (texture - 1.0) * weight
    return torch.clamp(x * combined, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — MOUNTING ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def apply_paper_curl(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    curl = _rand(0.0, 0.08)
    if curl < 0.01:
        return x
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij",
    )
    r2     = gx ** 2 + gy ** 2
    factor = 1.0 + curl * r2
    gx_w   = (gx * factor).clamp(-1, 1)
    gy_w   = (gy * factor).clamp(-1, 1)
    grid   = torch.stack([gx_w, gy_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="border")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — GEOMETRIC TRANSFORMS
# ════════════════════════════════════════════════════════════════════════════

def apply_perspective_warp(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    scenario = random.choices(
        ["straight", "approach_left", "approach_right", "steep_angle", "tilt"],
        weights=[0.15, 0.30, 0.30, 0.15, 0.10],
    )[0]
    src = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
    ], dtype=torch.float32, device=x.device)

    if scenario == "straight":
        tilt = _rand(-0.05, 0.05)
        dst  = src + torch.tensor([
            [-tilt, 0], [tilt, 0], [tilt, 0], [-tilt, 0]
        ], device=x.device)
    elif scenario == "approach_left":
        yaw, pitch = _rand(0.10, 0.35), _rand(0.02, 0.12)
        dst = torch.tensor([
            [-1.0 + yaw,      -1.0 + pitch    ],
            [ 1.0 - yaw*0.3,  -1.0 + pitch*0.5],
            [ 1.0 - yaw*0.3,   1.0 - pitch*0.5],
            [-1.0 + yaw,       1.0 - pitch    ],
        ], dtype=torch.float32, device=x.device)
    elif scenario == "approach_right":
        yaw, pitch = _rand(0.10, 0.35), _rand(0.02, 0.12)
        dst = torch.tensor([
            [-1.0 + yaw*0.3, -1.0 + pitch*0.5],
            [ 1.0 - yaw,     -1.0 + pitch    ],
            [ 1.0 - yaw,      1.0 - pitch    ],
            [-1.0 + yaw*0.3,  1.0 - pitch*0.5],
        ], dtype=torch.float32, device=x.device)
    elif scenario == "steep_angle":
        yaw = _rand(0.30, 0.50)
        dst = torch.tensor([
            [-1.0 + yaw, -1.0],
            [ 1.0,       -1.0],
            [ 1.0,        1.0],
            [-1.0 + yaw,  1.0],
        ], dtype=torch.float32, device=x.device)
    else:  # tilt
        angle        = _rand(-12, 12)
        rad          = angle * math.pi / 180.0
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=x.device)
        dst = (src @ rot.T).clamp(-1, 1)

    dst  = dst.clamp(-1.0, 1.0)
    grid = _four_point_to_grid(src, dst, H, W, x.device)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="border",
                         mode="bilinear")


def _four_point_to_grid(src, dst, H, W, device):
    A = []
    for (xs, ys), (xd, yd) in zip(src.tolist(), dst.tolist()):
        A.append([-xs, -ys, -1,  0,   0,  0, xd*xs, xd*ys, xd])
        A.append([ 0,   0,  0, -xs, -ys, -1, yd*xs, yd*ys, yd])
    A_t  = torch.tensor(A, dtype=torch.float32, device=device)
    _, _, Vt = torch.linalg.svd(A_t)
    h    = Vt[-1].reshape(3, 3)
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    ones   = torch.ones_like(gx)
    pts    = torch.stack([gx, gy, ones], dim=-1).reshape(-1, 3)
    H_inv  = torch.linalg.inv(h)
    warped = (H_inv @ pts.T).T
    warped = warped / warped[:, 2:3].clamp(min=1e-8)
    return warped[:, :2].reshape(H, W, 2).clamp(-2, 2)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — LIGHTING
# ════════════════════════════════════════════════════════════════════════════

def apply_brightness_gamma(x: torch.Tensor) -> torch.Tensor:
    brightness = _rand(0.45, 1.55)
    gamma      = _rand(0.55, 1.65)
    x = torch.clamp(x * brightness, 0.0, 1.0)
    x = torch.pow(x + 1e-8, gamma)
    return torch.clamp(x, 0.0, 1.0)


def apply_spotlight(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    cx, cy = _rand(0.1, 0.9) * W, _rand(0.0, 0.6) * H
    sigma  = _rand(0.35, 1.1) * max(H, W)
    gy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
    gx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)
    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
    light = (0.25 + 1.3 * torch.exp(-dist2 / (2 * sigma ** 2))).view(1, 1, H, W).expand(B, C, H, W)
    return torch.clamp(x * light, 0.0, 1.0)


def apply_retroreflection(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    sigma    = _rand(0.25, 0.55) * min(H, W)
    strength = _rand(0.0, 0.5)
    gy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
    gx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)
    dist2 = (gx - W/2.0)**2 + (gy - H/2.0)**2
    retro = (0.1 + 2.2 * torch.exp(-dist2 / (2 * sigma**2))).view(1, 1, H, W).expand(B, C, H, W)
    light = 1.0 + strength * (retro - 1.0)
    return torch.clamp(x * light, 0.0, 1.0)


def apply_colour_temperature(x: torch.Tensor) -> torch.Tensor:
    shift = _rand(-0.06, 0.08)
    tint  = torch.tensor([shift, shift * 0.3, -shift * 0.8],
                         device=x.device).view(1, 3, 1, 1)
    return torch.clamp(x + tint, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — CAMERA ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def apply_motion_blur(x: torch.Tensor) -> torch.Tensor:
    length = random.randint(1, 7)
    if length <= 1:
        return x
    kernel = torch.zeros(1, 1, 1, length, device=x.device)
    kernel[0, 0, 0, :] = 1.0 / length
    kernel = kernel.expand(3, 1, 1, length)
    return F.conv2d(x, kernel, padding=(0, length // 2), groups=3)


def apply_defocus_blur(x: torch.Tensor) -> torch.Tensor:
    sigma = _rand(0.0, 1.8)
    if sigma < 0.3:
        return x
    ks  = 5
    ax  = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    k1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0).expand(3, 1, ks, ks)
    return F.conv2d(x, k2d, padding=ks // 2, groups=3)


def apply_sensor_noise(x: torch.Tensor) -> torch.Tensor:
    std = _rand(0.0, 0.05)
    return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)


def apply_scale_jitter(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    scale = _rand(0.65, 1.20)
    if abs(scale - 1.0) < 0.03:
        return x
    new_h = max(16, int(H * scale))
    new_w = max(16, int(W * scale))
    resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    if scale < 1.0:
        pad_top   = (H - new_h) // 2
        pad_left  = (W - new_w) // 2
        pad_bot   = H - new_h - pad_top
        pad_right = W - new_w - pad_left
        out = F.pad(resized, (pad_left, pad_right, pad_top, pad_bot), value=0.5)
    else:
        top  = (new_h - H) // 2
        left = (new_w - W) // 2
        out  = resized[:, :, top:top + H, left:left + W]
    return out.clamp(0.0, 1.0)


def apply_crop_jitter(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    max_shift = max(1, int(H * 0.06))
    dy = random.randint(-max_shift, max_shift)
    dx = random.randint(-max_shift, max_shift)
    if dy == 0 and dx == 0:
        return x
    x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:  x[:, :, :dy, :]  = 0.5
    elif dy < 0: x[:, :, dy:, :]  = 0.5
    if dx > 0:  x[:, :, :, :dx]  = 0.5
    elif dx < 0: x[:, :, :, dx:]  = 0.5
    return x


def apply_windscreen_tint(x: torch.Tensor) -> torch.Tensor:
    tint_strength = _rand(0.0, 0.06)
    tint = torch.tensor([
        -tint_strength * 0.3,
         tint_strength * 0.2,
        -tint_strength * 0.1,
    ], device=x.device).view(1, 3, 1, 1)
    x = torch.clamp(x + tint, 0.0, 1.0)
    if random.random() < 0.4:
        shift = random.choice([1, 2])
        out = x.clone()
        out[:, 0] = torch.roll(x[:, 0], shifts=shift,  dims=-1)
        out[:, 2] = torch.roll(x[:, 2], shifts=-shift, dims=-1)
        x = out
    return torch.clamp(x, 0.0, 1.0)


def apply_jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    block   = 8
    quality = _rand(0.55, 0.95)
    ph = (block - H % block) % block
    pw = (block - W % block) % block
    xp = F.pad(x, (0, pw, 0, ph))
    _, _, Hp, Wp = xp.shape
    blocks = xp.unfold(2, block, block).unfold(3, block, block)
    noise  = torch.randn_like(blocks) * (1.0 - quality) * 0.04
    blocks = blocks + noise
    out = blocks.contiguous().view(B, C, Hp // block, Wp // block, block, block)
    out = out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)
    return torch.clamp(out[:, :, :H, :W], 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 6 — SENSOR (always last — physical order: optics → sensor → ISP)
# ════════════════════════════════════════════════════════════════════════════

def apply_grayscale(x: torch.Tensor) -> torch.Tensor:
    """
    Mobileye S-Cam4 uses a monochrome CMOS sensor — outputs grayscale frames.
    Convert to luminance-weighted grayscale, broadcast back to 3 channels so
    tensor shapes stay consistent with the rest of the pipeline.
    """
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return gray.expand_as(x).contiguous()


def apply_clahe(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate Mobileye's ISP local contrast normalisation via CLAHE.

    Uses a straight-through estimator so gradients survive:
      Forward  — real cv2 CLAHE is applied, loss sees the CLAHE-processed image,
                 so the patch is penalised for features CLAHE erases.
      Backward — gradient flows through as if CLAHE were identity (d_out/d_x = 1),
                 keeping the optimiser alive without breaking autograd.

    clipLimit=2.0, tileGridSize=(4,4) — standard automotive ISP settings.
    """
    import cv2
    import numpy as np

    x_det = x.detach()                                         # same values, no grad
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    results = []
    for i in range(x_det.shape[0]):                            # iterate over batch
        arr  = (x_det[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out  = clahe.apply(arr[:, :, 0])                       # single grayscale channel
        out3 = np.stack([out] * 3, axis=2).astype(np.float32) / 255.0
        results.append(torch.from_numpy(out3).permute(2, 0, 1))

    result = torch.stack(results, dim=0).to(x.device)

    # Straight-through: forward = result, backward gradient passes through x unchanged.
    # x - x_det == 0 in the forward pass, but carries x's gradient in the backward pass.
    return result + (x - x_det)


# ════════════════════════════════════════════════════════════════════════════
# EOT APPLICATION
# ════════════════════════════════════════════════════════════════════════════

PRINT_TRANSFORMS = [
    apply_gamut_compression,
    apply_cmyk_roundtrip,
    apply_dot_gain,
    apply_channel_misregistration,
    apply_print_banding,
    apply_paper_texture,
]

MOUNT_TRANSFORMS = [
    apply_paper_curl,
]

GEOMETRY_TRANSFORMS = [
    apply_perspective_warp,
    apply_scale_jitter,
    apply_crop_jitter,
]

LIGHTING_TRANSFORMS = [
    apply_brightness_gamma,
    apply_spotlight,
    apply_retroreflection,
    apply_colour_temperature,
]

CAMERA_TRANSFORMS = [
    apply_motion_blur,
    apply_defocus_blur,
    apply_sensor_noise,
    apply_jpeg_compression,
    apply_windscreen_tint,
]


def eot_batch(x: torch.Tensor, n_transforms: int = 3) -> torch.Tensor:
    """
    Apply transforms sampled from each physical stage in order.
    Physical causal chain: print → mount → geometry → lighting → camera → sensor.
    """
    # ── print artifacts (always one) ──────────────────────────────────────
    x = random.choice(PRINT_TRANSFORMS)(x)

    # ── mount curl (60% chance) ───────────────────────────────────────────
    if random.random() < 0.6:
        x = apply_paper_curl(x)

    # ── geometry (always all three, in order) ─────────────────────────────
    x = apply_perspective_warp(x)
    x = apply_scale_jitter(x)
    x = apply_crop_jitter(x)

    # ── lighting (always one) ─────────────────────────────────────────────
    x = random.choice(LIGHTING_TRANSFORMS)(x)

    # ── camera (one or two) ───────────────────────────────────────────────
    n_cam = 1 if n_transforms <= 3 else 2
    for fn in random.sample(CAMERA_TRANSFORMS, k=min(n_cam, len(CAMERA_TRANSFORMS))):
        x = fn(x)

    # ── sensor: grayscale then CLAHE (always last) ────────────────────────
    x = apply_grayscale(x)
    x = apply_clahe(x)

    return torch.clamp(x, 0.0, 1.0)
