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
  7. QUANT   — model-side: fake-quantised activations in the surrogate itself
               (hook-based, not part of eot_batch — see install_fake_quant_hooks)

All ops are differentiable so gradients flow back into the patch.
"""
import math
import random

import torch
import torch.nn as nn
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

def apply_oblique_perspective(x: torch.Tensor) -> torch.Tensor:
    """
    Extreme off-axis viewing angles for signs perpendicular to the road —
    i.e. when the car is passing a roadside sign close to the left or right.

    Physically: the sign face is nearly edge-on, so the image is severely
    foreshortened horizontally while the near vertical edge appears taller
    than the far edge (perspective convergence).

    Parameterised by off-axis angle θ (45–80° from the sign's normal) and
    lateral distance d (1.5–4m).  For a 450mm sign at θ=70°, d=2m:
      • apparent width  = 450 × cos(70°) ≈ 154mm  → heavy horizontal squeeze
      • convergence     = d / (d + 450×sin(70°)) ≈ 0.83  → near edge ~20% taller
    """
    B, C, H, W = x.shape

    side  = random.choice(["left", "right"])
    theta = _rand(45.0, 80.0) * math.pi / 180.0   # off-axis angle in radians
    d     = _rand(1.5, 4.0)                         # lateral distance in metres
    sign_w = 0.45                                    # sign width in metres

    # Horizontal foreshortening: how much of the sign width the camera sees
    squeeze = math.cos(theta)                        # 0.17 – 0.71

    # Perspective convergence: near edge vs far edge scale ratio
    far_scale = d / (d + sign_w * math.sin(theta))   # 0.75 – 0.95
    # near edge stays at scale ≈ 1.0

    # Small vertical tilt — sign may be slightly above/below camera height
    v_offset = _rand(-0.08, 0.08)

    if side == "left":
        # Camera is to the LEFT of the sign → left edge is near (larger)
        dst = torch.tensor([
            [-1.0,                     -1.0 + v_offset          ],   # TL (near, top)
            [-1.0 + 2.0 * squeeze,     -1.0 * far_scale + v_offset],  # TR (far, top)
            [-1.0 + 2.0 * squeeze,      1.0 * far_scale + v_offset],  # BR (far, bot)
            [-1.0,                      1.0 + v_offset          ],   # BL (near, bot)
        ], dtype=torch.float32, device=x.device)
    else:
        # Camera is to the RIGHT → right edge is near (larger)
        dst = torch.tensor([
            [ 1.0 - 2.0 * squeeze,     -1.0 * far_scale + v_offset],  # TL (far, top)
            [ 1.0,                      -1.0 + v_offset          ],  # TR (near, top)
            [ 1.0,                       1.0 + v_offset          ],  # BR (near, bot)
            [ 1.0 - 2.0 * squeeze,       1.0 * far_scale + v_offset],  # BL (far, bot)
        ], dtype=torch.float32, device=x.device)

    dst = dst.clamp(-1.0, 1.0)
    src = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
    ], dtype=torch.float32, device=x.device)

    grid = _four_point_to_grid(src, dst, H, W, x.device)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="border",
                         mode="bilinear")


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
    """Uniform retroreflection — legacy path when no patch mask is available."""
    B, C, H, W = x.shape
    sigma    = _rand(0.25, 0.55) * min(H, W)
    strength = _rand(0.0, 0.5)
    gy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
    gx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)
    dist2 = (gx - W/2.0)**2 + (gy - H/2.0)**2
    retro = (0.1 + 2.2 * torch.exp(-dist2 / (2 * sigma**2))).view(1, 1, H, W).expand(B, C, H, W)
    light = 1.0 + strength * (retro - 1.0)
    return torch.clamp(x * light, 0.0, 1.0)


def apply_differential_retroreflection(x: torch.Tensor,
                                        patch_mask: torch.Tensor) -> torch.Tensor:
    """
    The sign's retroreflective aluminium sheeting glows under headlights;
    the printed patch (matte paper / vinyl sticker) does not.  This creates
    a stark brightness contrast the EyeQ4 camera sees at night or under
    carpark lighting — the sign background is bright, the patch rectangle
    stays dark.

    patch_mask: (B, 1, H, W) — 1.0 where the patch is, 0.0 on sign background.
    Must already be geometry-warped to match x.
    """
    B, C, H, W = x.shape
    sigma    = _rand(0.25, 0.55) * min(H, W)
    strength = _rand(0.15, 0.70)   # stronger range — real retro is dramatic
    gy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
    gx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)
    dist2 = (gx - W / 2.0) ** 2 + (gy - H / 2.0) ** 2
    retro = (0.1 + 2.2 * torch.exp(-dist2 / (2 * sigma ** 2)))
    retro = retro.view(1, 1, H, W).expand(B, C, H, W)
    boost = strength * (retro - 1.0)
    # Only the sign background (non-patch) gets the retroreflective boost
    sign_mask = 1.0 - patch_mask                       # (B, 1, H, W)
    x = x * (1.0 + boost * sign_mask)
    return torch.clamp(x, 0.0, 1.0)


def apply_auto_exposure(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate the S-Cam4 ISP's auto-exposure highlight compression.

    When headlights hit a retroreflective sign, the raw sensor image has
    extreme dynamic range.  The ISP's AE algorithm compresses highlights
    to keep the sign readable without blowing out the frame.  This is a
    non-linear tone curve applied BEFORE the classifier sees the crop.

    Modelled as a soft knee compressor: pixels above `knee` are compressed
    by `ratio`, smoothed with a tanh rolloff so it stays differentiable.
    """
    knee  = _rand(0.55, 0.80)   # AE knee point (bright pixels above here get compressed)
    ratio = _rand(0.3, 0.7)     # compression strength (lower = harder compression)
    # Soft knee: below knee → identity, above knee → compressed
    excess = F.relu(x - knee)
    compressed = knee + excess * ratio * torch.tanh(excess / (0.1 + excess * 0.5))
    below = torch.min(x, torch.tensor(knee, device=x.device))
    x = below + compressed
    return torch.clamp(x, 0.0, 1.0)


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
    scale = _rand(0.80, 1.15)
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
# GROUP 7 — MODEL-SIDE QUANTISATION NOISE
# ════════════════════════════════════════════════════════════════════════════
# Unlike groups 1-6, this doesn't touch the input image — it hooks the
# surrogate's own Conv2d/Linear layers so the optimiser sees int-N rounding
# in the *activations*, approximating an embedded NPU without needing the
# real target's weights or calibration stats. Bit-width, and whether a given
# layer is quantised on a given forward pass, are both sampled — quantisation
# scheme is treated as an unknown nuisance to marginalise over, same as the
# unknown printer ICC profile in group 1.

class FakeQuantSchedule:
    """
    Shared state for every installed fake-quant hook.  Controls both the
    on/off switch AND a linear ramp on p_apply so quantisation noise is
    introduced gradually instead of slamming on at full strength.

    Usage from the training loop:
        schedule = FakeQuantSchedule(enabled=False, p_max=0.3, ramp_steps=500)
        ...
        # at warmup completion:
        schedule.enabled = True
        ...
        # every step after that:
        schedule.step()          # advances the ramp
        current_p = schedule.p   # hooks read this
    """
    def __init__(self, enabled: bool = True, p_max: float = 0.3,
                 ramp_steps: int = 500):
        self.enabled    = enabled
        self.p_max      = p_max
        self.ramp_steps = max(ramp_steps, 1)
        self._ramp_pos  = 0        # how many steps since enabled

    @property
    def p(self) -> float:
        """Current effective p_apply — linearly ramps from 0.05 to p_max."""
        if not self.enabled:
            return 0.0
        t = min(self._ramp_pos / self.ramp_steps, 1.0)
        return 0.05 + (self.p_max - 0.05) * t

    def step(self):
        """Call once per training step after enabling."""
        if self.enabled:
            self._ramp_pos += 1


def _sym_fake_quant(x: torch.Tensor, bits: int, subsample: int = 4096):
    """
    Symmetric per-tensor fake-quantise a tensor to `bits` width.
    Returns (dequantised_detached, scale) — caller decides STE vs detach.
    """
    qmax = float(2 ** (bits - 1) - 1)              # 127 for 8-bit
    flat = x.detach().reshape(-1).float()
    if flat.numel() > subsample:
        flat = flat[torch.randint(0, flat.numel(), (subsample,), device=flat.device)]
    amax    = torch.quantile(flat.abs(), 0.995).clamp(min=1e-8)
    scale   = amax / qmax
    clipped = x.detach().clamp(-amax, amax)
    q       = torch.round(clipped / scale).clamp(-qmax, qmax)
    return q * scale, scale


def _fake_quant_hook(bits_choices, p_apply, schedule):
    def hook(module, inp, out):
        # Use schedule's ramped p if available, otherwise fall back to static p_apply
        effective_p = schedule.p if hasattr(schedule, 'p') else p_apply
        if not schedule.enabled or random.random() > effective_p:
            return out
        bits = random.choice(bits_choices)

        # ── Activation-only fake-quant (STE) ─────────────────────────
        # We quantise activations only during training.  Weight quantisation
        # during training had a broken STE: the gradient flowed through `out`
        # (computed with float weights) but the forward used `out_dq` (from
        # quantised weights), so the gradient didn't match the forward path.
        #
        # Activation-only quant is sufficient because:
        #   1. The surrogates' weights are frozen during patch optimisation —
        #      gradients flow to the patch tensor, not the weights.
        #   2. The activation bottleneck is what matters for transfer: the patch
        #      must produce activations that survive INT8 rounding.
        #   3. Weight quant still runs in the deterministic eval hooks (p=1.0,
        #      no backward needed) so qASR correctly reflects both W+A quant.
        #
        # EyeQ4 uses symmetric per-tensor INT8 — the simplest hardware scheme.
        out_dq, _ = _sym_fake_quant(out, bits)
        # Straight-through: forward = quantised, backward = identity w.r.t. out.
        return out + (out_dq - out.detach())
    return hook


def install_fake_quant_hooks(model: nn.Module,
                             bits_choices=(6, 8),
                             p_apply: float = 0.3,
                             schedule: "FakeQuantSchedule | None" = None):
    """
    Register forward hooks on every Conv2d/Linear in `model` that fake-quantise
    that layer's output to a randomly sampled bit-width from bits_choices,
    applied with probability p_apply per layer per forward call, and only while
    schedule.enabled is True.

    When the schedule has a ramp (FakeQuantSchedule with ramp_steps > 0), the
    effective probability starts at 0.05 and linearly increases to p_max over
    ramp_steps training steps after enabling — call schedule.step() each
    iteration. The static p_apply is used as fallback only for legacy schedules.

    Straight-through estimator keeps gradients flowing to the patch. Returns
    (handles, schedule) — pass handles to remove_fake_quant_hooks() to undo.
    If no schedule is given a fresh always-enabled one is created.
    """
    if schedule is None:
        schedule = FakeQuantSchedule(enabled=True)
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(_fake_quant_hook(bits_choices, p_apply, schedule)))
    return handles, schedule


def remove_fake_quant_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


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

# Lighting transforms that don't include retroreflection — used in eot_scene
# when a patch mask is available and differential retroreflection is applied
# as a separate step.
LIGHTING_TRANSFORMS_BASE = [
    apply_brightness_gamma,
    apply_spotlight,
    apply_colour_temperature,
    apply_auto_exposure,
]

CAMERA_TRANSFORMS = [
    apply_motion_blur,
    apply_defocus_blur,
    apply_sensor_noise,
    apply_jpeg_compression,
    apply_windscreen_tint,
]


def eot_print(x: torch.Tensor) -> torch.Tensor:
    """
    Print-only EOT transforms — gamut, CMYK, dot gain, banding, paper texture,
    and mounting curl.  Applied to the PATCH ONLY, before compositing onto the
    sign image.  The physical sign is retroreflective aluminium — it never
    passes through a printer.

    A real inkjet print suffers ALL these artefacts simultaneously (gamut
    compression AND dot gain AND banding etc.), not one at a time.  We sample
    2–3 and apply them in sequence so the patch must survive the stacked
    degradation, not just each artefact individually.
    """
    n = random.randint(2, min(3, len(PRINT_TRANSFORMS)))
    for fn in random.sample(PRINT_TRANSFORMS, k=n):
        x = fn(x)
    if random.random() < 0.6:
        x = apply_paper_curl(x)
    return torch.clamp(x, 0.0, 1.0)


def eot_scene(x: torch.Tensor, n_transforms: int = 3,
              oblique: bool = False,
              patch_mask: torch.Tensor = None,
              training_step: int = None) -> torch.Tensor:
    """
    Scene-level EOT transforms — applied to the composited image (sign + patch
    together) because these happen in the physical world / camera, not the
    printer.

    Physical causal chain: geometry → lighting → camera → sensor.

    oblique: if True, mix in extreme off-axis perspective transforms (~40% of
    the time) to simulate the camera passing a roadside sign close to the
    left or right.  The sign is perpendicular to the road so the face is
    nearly edge-on — heavy horizontal foreshortening + convergence that the
    standard approach_left/right scenarios don't cover.

    patch_mask: (B, 1, H, W) binary mask — 1.0 where the patch is, 0.0 on
    sign background.  When provided, the mask is warped through the same
    geometry transforms so differential retroreflection can distinguish
    the retroreflective sign background from the matte printed patch.

    training_step: current optimisation step (None = eval / always apply).
    When set and < 500, the grayscale+CLAHE sensor path is skipped to let
    the patch converge on colour features first before hardening against
    the monochrome ISP pipeline.
    """
    has_mask = patch_mask is not None

    # ── concat mask as 4th channel so geometry warps it identically ────
    if has_mask:
        x = torch.cat([x, patch_mask], dim=1)          # (B, 4, H, W)

    # ── geometry (always all three, in order) ─────────────────────────────
    if oblique and random.random() < 0.4:
        x = apply_oblique_perspective(x)
    else:
        x = apply_perspective_warp(x)
    x = apply_scale_jitter(x)
    x = apply_crop_jitter(x)

    # ── split mask back out after geometry ─────────────────────────────
    if has_mask:
        warped_mask = (x[:, 3:4] > 0.5).float()        # threshold after interpolation
        x = x[:, :3]

    # ── retroreflection (separate from other lighting) ────────────────
    # The sign's aluminium sheeting retroreflects; the printed patch doesn't.
    # When we have the mask, apply differential retroreflection so the patch
    # must survive the brightness contrast.  Without a mask, fall back to
    # uniform retroreflection via the LIGHTING_TRANSFORMS list.
    if has_mask and random.random() < 0.5:
        x = apply_differential_retroreflection(x, warped_mask)

    # ── lighting (always one) ─────────────────────────────────────────────
    if has_mask:
        # Retroreflection handled above — pick from base lighting only
        x = random.choice(LIGHTING_TRANSFORMS_BASE)(x)
    else:
        # Legacy path — retroreflection mixed in with other lighting
        x = random.choice(LIGHTING_TRANSFORMS)(x)

    # ── camera (one or two) ───────────────────────────────────────────────
    n_cam = 1 if n_transforms <= 3 else 2
    for fn in random.sample(CAMERA_TRANSFORMS, k=min(n_cam, len(CAMERA_TRANSFORMS))):
        x = fn(x)

    # ── sensor: grayscale + CLAHE (50% of EOT samples) ─────────────────
    # The target's monochrome sensor always sees grayscale+CLAHE, but the
    # surrogates were trained on ~50% colour / ~50% grayscale (dataset.py
    # RandomGrayscale p=0.5).  Gating to 50% keeps the gradient signal
    # alive through colour features the surrogates actually learned, while
    # still forcing the patch to survive the grayscale+CLAHE path half the
    # time.  Without gating, 100% of gradient updates come through the
    # CLAHE STE, which is a systematically wrong approximation — the
    # optimizer doesn't know CLAHE's actual local effect on each pixel.
    #
    # During early training (step < 500), skip grayscale+CLAHE entirely so
    # the patch converges using richer colour gradients first.  The CLAHE
    # STE is a coarse approximation that adds gradient noise — deferring it
    # lets the optimiser find a good basin before hardening.
    _apply_sensor = (training_step is None or training_step >= 500)
    if _apply_sensor and random.random() < 0.5:
        x = apply_grayscale(x)
        x = apply_clahe(x)

    # STE clamp: forward value is clamped to [0,1] (correct for the model),
    # but backward gradient passes through unchanged so pixels at saturation
    # boundaries still get useful gradients.  This fixes the gradient death
    # from the denormalise → eot_scene(clamp) → renormalise round-trip in
    # patch_attack.py — without this, any pixel pushed to 0 or 1 by the
    # lighting/camera transforms has zero gradient back to the patch.
    clamped = torch.clamp(x, 0.0, 1.0)
    return x + (clamped - x).detach()      # forward=clamped, backward=identity


def eot_batch(x: torch.Tensor, n_transforms: int = 3,
              oblique: bool = False) -> torch.Tensor:
    """
    Legacy combined EOT — applies ALL transforms (print + scene) to the
    full image.  Kept for backward compatibility with evaluation code.
    For training, use eot_print() on the patch and eot_scene() on the
    composited image separately.
    """
    x = eot_print(x)
    return eot_scene(x, n_transforms=n_transforms, oblique=oblique)
