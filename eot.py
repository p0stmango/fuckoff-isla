"""
Expectation over Transformation (EOT) — physical-world augmentations applied
during patch optimisation to improve real-world transfer.

Transforms are grouped and applied in physical order:
  1. PRINT  — what the inkjet/laser printer does to the patch design
  2. MOUNT  — paper curl from blu-tack mounting on a non-flat surface
  3. GEOMETRY — projective warp from different viewing angles while driving
  4. LIGHTING — ambient + point-source illumination variation
  5. CAMERA  — sensor noise, motion blur, compression artifacts

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
# Simulate what an inkjet or laser printer does between the patch tensor
# and the physical piece of paper.
# ════════════════════════════════════════════════════════════════════════════

def apply_gamut_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Inkjet/laser printers work in CMYK and cannot reproduce the full sRGB
    gamut.  Saturated reds (the dominant colour in adversarial patches and
    in the sign ring itself) are particularly affected.

    Model: soft-clip via a smooth sigmoid knee, with channel-specific
    compression factors derived from typical inkjet gamut boundaries.
    """
    # Per-channel compression factors (R compressed hardest — inkjet reds clip)
    factors = torch.tensor([0.82, 0.90, 0.88], device=x.device).view(1, 3, 1, 1)
    offsets = torch.tensor([0.04, 0.02, 0.02], device=x.device).view(1, 3, 1, 1)
    # Add random per-run variation to avoid patch overfitting to a single printer
    factors = factors + torch.randn_like(factors) * 0.03
    x = x * factors + offsets
    return torch.clamp(x, 0.0, 1.0)


def apply_cmyk_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate the colour shift from sRGB → CMYK → sRGB conversion.
    Causes slight desaturation and a warm-shift in neutral tones.
    """
    # Simplified CMY: C=1-R, M=1-G, Y=1-B
    c = 1.0 - x[:, 0:1]
    m = 1.0 - x[:, 1:2]
    y = 1.0 - x[:, 2:3]
    k = torch.min(torch.cat([c, m, y], dim=1), dim=1, keepdim=True).values
    # Clamp K to simulate under-colour removal
    k_amount = _rand(0.05, 0.20)
    k = k * k_amount
    # Back to RGB with slight K contamination (muddy neutrals)
    r = torch.clamp(1.0 - c - k, 0, 1)
    g = torch.clamp(1.0 - m - k, 0, 1)
    b = torch.clamp(1.0 - y - k, 0, 1)
    return torch.cat([r, g, b], dim=1)


def apply_dot_gain(x: torch.Tensor) -> torch.Tensor:
    """
    Ink spread on paper fibre: darkens midtones, especially in high-coverage
    areas.  Standard inkjet dot gain is 15–25%.
    """
    gain = _rand(0.08, 0.22)
    # Tent function peaks at 0.5 (maximum midtone darkening)
    x = x - gain * 4.0 * x * (1.0 - x)
    return torch.clamp(x, 0.0, 1.0)


def apply_channel_misregistration(x: torch.Tensor) -> torch.Tensor:
    """
    Colour fringing from slight CMYK layer misalignment during printing.
    Shifts R and B channels by ±1–2 pixels relative to G.
    """
    B, C, H, W = x.shape
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
    """
    Horizontal banding from inkjet print-head passes.
    Every N rows is slightly lighter or darker.
    """
    B, C, H, W = x.shape
    band_period = random.randint(8, 24)
    band_amp    = _rand(0.00, 0.04)

    rows = torch.arange(H, device=x.device, dtype=x.dtype)
    # Sinusoidal banding pattern
    banding = 1.0 + band_amp * torch.sin(2 * math.pi * rows / band_period)
    banding  = banding.view(1, 1, H, 1).expand(B, C, H, W)
    return torch.clamp(x * banding, 0.0, 1.0)


def apply_paper_texture(x: torch.Tensor) -> torch.Tensor:
    """
    Paper surface texture: slight high-frequency multiplicative noise that
    simulates paper grain under the ink.  More visible in lighter areas.
    """
    texture_std = _rand(0.0, 0.03)
    texture = 1.0 + torch.randn_like(x) * texture_std
    # Weight by local brightness — texture shows more on light areas
    weight = x.detach().mean(dim=1, keepdim=True).expand_as(x)
    combined = 1.0 + (texture - 1.0) * weight
    return torch.clamp(x * combined, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — MOUNTING ARTIFACTS
# A4 paper blu-tacked to a sign outdoors will curl at the edges and may
# bow slightly in the middle.
# ════════════════════════════════════════════════════════════════════════════

def apply_paper_curl(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate edge curl of A4 paper on a sign: the corners lift slightly,
    causing a perspective-like warp strongest at the paper edges.
    Implemented as a smooth barrel/pincushion distortion.
    """
    B, C, H, W = x.shape
    curl = _rand(0.0, 0.08)
    if curl < 0.01:
        return x

    # Normalised coordinate grid [-1, 1]
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij",
    )
    r2 = gx ** 2 + gy ** 2
    # Barrel distortion: points move outward near edges
    factor = 1.0 + curl * r2
    gx_w = (gx * factor).clamp(-1, 1)
    gy_w = (gy * factor).clamp(-1, 1)
    grid = torch.stack([gx_w, gy_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="border")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — GEOMETRIC TRANSFORMS
# Simulate viewing the sign from a moving vehicle at various angles.
# ════════════════════════════════════════════════════════════════════════════

def apply_perspective_warp(x: torch.Tensor) -> torch.Tensor:
    """
    True projective (homographic) warp via 4-point correspondence.

    Simulates the dominant viewing geometries when driving past a sign:
      - Horizontal yaw: sign appears keystoned as you approach/pass
      - Vertical pitch: camera is below the sign, looking up slightly
      - Combined approach angles

    Driving past at low carpark speed means yaw angles up to ~35°.
    """
    B, C, H, W = x.shape

    # Choose a scenario weighted toward realistic driving geometries
    scenario = random.choices(
        ["straight", "approach_left", "approach_right", "steep_angle", "tilt"],
        weights=[0.15, 0.30, 0.30, 0.15, 0.10],
    )[0]

    # Source corners in normalised coords: TL, TR, BR, BL
    src = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
    ], dtype=torch.float32, device=x.device)

    if scenario == "straight":
        # Slight random tilt only
        tilt = _rand(-0.05, 0.05)
        dst = src + torch.tensor([
            [-tilt, 0], [tilt, 0], [tilt, 0], [-tilt, 0]
        ], device=x.device)

    elif scenario == "approach_left":
        # Car approaching from the left — right side of sign appears closer
        yaw = _rand(0.10, 0.35)
        pitch = _rand(0.02, 0.12)
        dst = torch.tensor([
            [-1.0 + yaw,  -1.0 + pitch],
            [ 1.0 - yaw*0.3, -1.0 + pitch*0.5],
            [ 1.0 - yaw*0.3,  1.0 - pitch*0.5],
            [-1.0 + yaw,   1.0 - pitch],
        ], dtype=torch.float32, device=x.device)

    elif scenario == "approach_right":
        yaw = _rand(0.10, 0.35)
        pitch = _rand(0.02, 0.12)
        dst = torch.tensor([
            [-1.0 + yaw*0.3, -1.0 + pitch*0.5],
            [ 1.0 - yaw,    -1.0 + pitch],
            [ 1.0 - yaw,     1.0 - pitch],
            [-1.0 + yaw*0.3,  1.0 - pitch*0.5],
        ], dtype=torch.float32, device=x.device)

    elif scenario == "steep_angle":
        # Passing close to the sign — strong perspective
        yaw = _rand(0.30, 0.50)
        dst = torch.tensor([
            [-1.0 + yaw,  -1.0],
            [ 1.0,        -1.0],
            [ 1.0,         1.0],
            [-1.0 + yaw,   1.0],
        ], dtype=torch.float32, device=x.device)

    else:  # tilt — sign post not perfectly vertical
        angle = _rand(-12, 12)
        rad   = angle * math.pi / 180.0
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = torch.tensor([
            [cos_a, -sin_a],
            [sin_a,  cos_a],
        ], device=x.device)
        dst = (src @ rot.T).clamp(-1, 1)

    dst = dst.clamp(-1.0, 1.0)

    # Build sampling grid from the 4-point correspondence via homography
    grid = _four_point_to_grid(src, dst, H, W, x.device)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode="border",
                         mode="bilinear")


def _four_point_to_grid(
    src: torch.Tensor,   # (4,2) normalised source corners
    dst: torch.Tensor,   # (4,2) normalised destination corners
    H: int, W: int,
    device,
) -> torch.Tensor:
    """
    Compute a dense sampling grid for grid_sample given 4-point correspondence.
    Uses the DLT homography estimate (SVD solution).
    Returns (H, W, 2) grid in normalised coords.
    """
    # Build the 8×9 DLT matrix
    A = []
    for (xs, ys), (xd, yd) in zip(src.tolist(), dst.tolist()):
        A.append([-xs, -ys, -1,  0,   0,  0, xd*xs, xd*ys, xd])
        A.append([ 0,   0,  0, -xs, -ys, -1, yd*xs, yd*ys, yd])
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    _, _, Vt = torch.linalg.svd(A_t)
    h = Vt[-1].reshape(3, 3)   # last right singular vector

    # Create dense normalised grid and apply H⁻¹ (inverse warp)
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(gx)
    pts  = torch.stack([gx, gy, ones], dim=-1).reshape(-1, 3)   # (H*W, 3)

    H_inv = torch.linalg.inv(h)
    warped = (H_inv @ pts.T).T   # (H*W, 3)
    warped = warped / warped[:, 2:3].clamp(min=1e-8)
    grid   = warped[:, :2].reshape(H, W, 2).clamp(-2, 2)
    return grid


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — LIGHTING
# ════════════════════════════════════════════════════════════════════════════

def apply_brightness_gamma(x: torch.Tensor) -> torch.Tensor:
    """Uniform ambient lighting variation — overcast vs sunny day."""
    brightness = _rand(0.45, 1.55)
    gamma      = _rand(0.55, 1.65)
    x = torch.clamp(x * brightness, 0.0, 1.0)
    x = torch.pow(x + 1e-8, gamma)
    return torch.clamp(x, 0.0, 1.0)


def apply_spotlight(x: torch.Tensor) -> torch.Tensor:
    """
    Non-uniform point-source illumination — carpark overhead lights,
    streetlights, or harsh direct sun.  Creates a hot spot offset from
    centre so one side of the patch is brighter than the other.
    """
    B, C, H, W = x.shape
    cx    = _rand(0.1, 0.9) * W
    cy    = _rand(0.0, 0.6) * H   # biased upward — light comes from above
    sigma = _rand(0.35, 1.1) * max(H, W)

    gy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
    gx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)
    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
    light = 0.25 + 1.3 * torch.exp(-dist2 / (2 * sigma ** 2))
    light = light.view(1, 1, H, W).expand(B, C, H, W)
    return torch.clamp(x * light, 0.0, 1.0)


def apply_retroreflection(x: torch.Tensor) -> torch.Tensor:
    """
    Road signs use retroreflective sheeting — they return light strongly
    toward the headlight source.  At night this makes the sign appear very
    bright in the centre and dim at the edges.

    The A4 paper patch does NOT retroreflect — only the underlying sign
    material does.  But the camera still sees the whole sign face with
    this lighting profile, and the patch must survive it.
    """
    B, C, H, W = x.shape
    cx = W / 2.0
    cy = H / 2.0
    sigma = _rand(0.25, 0.55) * min(H, W)

    gy = torch.arange(H, device=x.device, dtype=x.dtype).view(H, 1).expand(H, W)
    gx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, W).expand(H, W)
    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
    # Night mode: strong centre hotspot, dark falloff
    retro = 0.1 + 2.2 * torch.exp(-dist2 / (2 * sigma ** 2))
    retro = retro.view(1, 1, H, W).expand(B, C, H, W)
    strength = _rand(0.0, 0.5)   # 0 = no effect (day), 0.5 = strong (night)
    light = 1.0 + strength * (retro - 1.0)
    return torch.clamp(x * light, 0.0, 1.0)


def apply_colour_temperature(x: torch.Tensor) -> torch.Tensor:
    """
    Shift white balance — cool (overcast sky, ~6500K) to warm (sodium
    streetlight, ~2700K).  Carpark LED lights are typically ~4000K cool-white.
    """
    # Negative = cool/blue shift, positive = warm/orange shift
    shift = _rand(-0.06, 0.08)
    tint  = torch.tensor([shift, shift * 0.3, -shift * 0.8],
                         device=x.device).view(1, 3, 1, 1)
    return torch.clamp(x + tint, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — CAMERA ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def apply_motion_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Directional motion blur from vehicle forward motion.
    The sign moves horizontally across the camera frame as the car passes,
    so blur is primarily in the horizontal axis.
    At low carpark speeds blur is minimal; at road speed it can be several px.
    """
    B, C, H, W = x.shape
    length = random.randint(1, 7)   # blur kernel length in px
    if length <= 1:
        return x

    # Horizontal motion blur kernel
    kernel = torch.zeros(1, 1, 1, length, device=x.device)
    kernel[0, 0, 0, :] = 1.0 / length
    kernel = kernel.expand(3, 1, 1, length)
    return F.conv2d(x, kernel, padding=(0, length // 2), groups=3)


def apply_defocus_blur(x: torch.Tensor) -> torch.Tensor:
    """Lens defocus — sign not perfectly in focus at close or far range."""
    sigma = _rand(0.0, 1.8)
    if sigma < 0.3:
        return x
    ks = 5
    ax = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    k1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0).expand(3, 1, ks, ks)
    return F.conv2d(x, k2d, padding=ks // 2, groups=3)


def apply_sensor_noise(x: torch.Tensor) -> torch.Tensor:
    """Shot noise + read noise. More prominent in low-light carpark scenes."""
    std = _rand(0.0, 0.05)
    return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)


def apply_scale_jitter(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate the sign occupying different fractions of the camera frame as
    the car approaches.  Resizes the content to a random scale then pads
    or crops back to the original size.  This means the patch may appear
    larger or smaller within the 224px input the model sees.
    """
    B, C, H, W = x.shape
    scale = _rand(0.65, 1.20)
    if abs(scale - 1.0) < 0.03:
        return x

    new_h = max(16, int(H * scale))
    new_w = max(16, int(W * scale))
    resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    if scale < 1.0:
        # Smaller — pad back to original size with grey border
        pad_top  = (H - new_h) // 2
        pad_left = (W - new_w) // 2
        pad_bot  = H - new_h - pad_top
        pad_right = W - new_w - pad_left
        out = F.pad(resized, (pad_left, pad_right, pad_top, pad_bot), value=0.5)
    else:
        # Larger — centre-crop back to original size
        top  = (new_h - H) // 2
        left = (new_w - W) // 2
        out  = resized[:, :, top:top + H, left:left + W]

    return out.clamp(0.0, 1.0)


def apply_crop_jitter(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate imperfect sign detection crop from the EyeQ4 sign detector.
    The classifier receives a crop that may be slightly mistranslated
    relative to the true sign centre — a few pixels in any direction.
    """
    B, C, H, W = x.shape
    max_shift = max(1, int(H * 0.06))   # up to 6% of image dimension
    dy = random.randint(-max_shift, max_shift)
    dx = random.randint(-max_shift, max_shift)
    if dy == 0 and dx == 0:
        return x

    # Roll then zero-pad the exposed edge (cleaner than grid_sample here)
    x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        x[:, :, :dy, :]  = 0.5
    elif dy < 0:
        x[:, :, dy:, :]  = 0.5
    if dx > 0:
        x[:, :, :, :dx]  = 0.5
    elif dx < 0:
        x[:, :, :, dx:]  = 0.5
    return x


def apply_windscreen_tint(x: torch.Tensor) -> torch.Tensor:
    """
    The S-Cam4 sits behind the windscreen.  Modern windscreens have a slight
    green/grey tint and introduce a very mild prismatic blur on high-contrast
    edges.  Minor effect but free to include.
    """
    # Slight green-grey tint (typical automotive glass)
    tint_strength = _rand(0.0, 0.06)
    tint = torch.tensor([
        -tint_strength * 0.3,   # reduce red slightly
         tint_strength * 0.2,   # boost green slightly
        -tint_strength * 0.1,   # reduce blue slightly
    ], device=x.device).view(1, 3, 1, 1)
    x = torch.clamp(x + tint, 0.0, 1.0)

    # Very mild chromatic aberration on edges (prismatic effect)
    if random.random() < 0.4:
        shift = random.choice([1, 2])
        out = x.clone()
        out[:, 0] = torch.roll(x[:, 0], shifts=shift,  dims=-1)   # red shifts right
        out[:, 2] = torch.roll(x[:, 2], shifts=-shift, dims=-1)   # blue shifts left
        x = out

    return torch.clamp(x, 0.0, 1.0)


def apply_jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate JPEG block compression artifacts via 8×8 DCT coefficient
    quantisation.  The EyeQ4 camera pipeline likely applies in-ISP compression
    before passing frames to the CNN.
    """
    B, C, H, W = x.shape
    block = 8
    quality = _rand(0.55, 0.95)   # lower = more compression = more artifacts

    # Pad to block boundary
    ph = (block - H % block) % block
    pw = (block - W % block) % block
    xp = F.pad(x, (0, pw, 0, ph))
    _, _, Hp, Wp = xp.shape

    # Unfold into blocks, add quantisation noise, fold back
    blocks = xp.unfold(2, block, block).unfold(3, block, block)
    noise  = torch.randn_like(blocks) * (1.0 - quality) * 0.04
    blocks = blocks + noise
    # Fold back (average overlapping — here non-overlapping so it's exact)
    out = blocks.contiguous().view(B, C, Hp // block, Wp // block, block, block)
    out = out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)
    return torch.clamp(out[:, :, :H, :W], 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# EOT APPLICATION
# ════════════════════════════════════════════════════════════════════════════

# Each group lists its transforms.  eot_batch samples from each group
# independently so every call exercises at least one transform per physical
# stage — important because print artifacts always precede camera artifacts.

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

    Physical causal chain: print → mount → geometry → lighting → camera.

    Geometry (perspective + scale + crop) is always applied — it's the
    dominant real-world effect.  Other groups sample one transform each,
    with camera sampling more at higher n_transforms.
    """
    # ── print artifacts (always one) ──────────────────────────────────────
    x = random.choice(PRINT_TRANSFORMS)(x)

    # ── mount curl (60% chance — not always present) ──────────────────────
    if random.random() < 0.6:
        x = apply_paper_curl(x)

    # ── geometry (always all three, in order) ─────────────────────────────
    x = apply_perspective_warp(x)
    x = apply_scale_jitter(x)
    x = apply_crop_jitter(x)

    # ── lighting (always one) ─────────────────────────────────────────────
    x = random.choice(LIGHTING_TRANSFORMS)(x)

    # ── camera (one or two depending on n_transforms budget) ─────────────
    n_cam = 1 if n_transforms <= 3 else 2
    for fn in random.sample(CAMERA_TRANSFORMS, k=min(n_cam, len(CAMERA_TRANSFORMS))):
        x = fn(x)

    return torch.clamp(x, 0.0, 1.0)
