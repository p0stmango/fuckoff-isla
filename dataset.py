"""
Dataset loader merging GTSRB speed-limit classes with synthetic Australian signs.

GTSRB provides real-photo diversity for the overlapping speed classes.
Australian synthetic data (aus_signs.py) adds AU-only classes and the
correct "km/h" sub-label style for transfer to the EyeQ4 AU firmware.

AU speed set:  5 10 15 20 25 30 40 50 60 70 80 90 100 110
GTSRB speeds: 20    30    50 60 70 80   100 120     (no 5/10/15/25/40/90/110)
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.datasets as tvd

from aus_signs import AU_SPEEDS, generate_dataset


class RandomCLAHE:
    """
    Apply CLAHE with a given probability during training augmentation.

    The target's ISP runs CLAHE on every frame.  Training the surrogates
    with CLAHE augmentation ensures they learn features that survive local
    contrast normalisation, closing the domain gap between training
    (colour PIL images) and the EOT pipeline's CLAHE path.

    clipLimit is randomised (1.5–3.0) to avoid overfitting to one setting.
    """

    def __init__(self, p: float = 0.4, clip_lo: float = 1.5, clip_hi: float = 3.0):
        self.p = p
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img)
        clip = random.uniform(self.clip_lo, self.clip_hi)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        if arr.ndim == 2:
            arr = clahe.apply(arr)
        else:
            # Apply to L channel in LAB space to preserve colour
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)

# ── label space (union of AU and GTSRB speeds we care about) ─────────────────
# 120 km/h is GTSRB-only (no AU 120 limit) — include for model robustness.
ALL_SPEEDS   = sorted(set(AU_SPEEDS))
KMH_TO_LABEL = {kmh: i for i, kmh in enumerate(ALL_SPEEDS)}
NUM_CLASSES  = len(ALL_SPEEDS)

# Source-5 km/h is our known carpark sign; we want ANY sign → fool classifier
# Default attack target — override in patch_attack.py CLI
DEFAULT_TARGET_KMH = 50

# GTSRB class IDs that are speed-limit signs (skip class 6 = end-of-80)
GTSRB_SPEED_CLASSES = {0, 1, 2, 3, 4, 5, 7}
GTSRB_ID_TO_KMH     = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100}

IMG_SIZE  = 224
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.30, hue=0.08),
    T.RandomRotation(18),
    T.RandomAffine(degrees=0, shear=15),
    T.RandomPerspective(distortion_scale=0.40, p=0.6),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.2)),
    T.RandomGrayscale(p=0.5),     # ↑ from 0.3 — match eot.py 50% grayscale gate
    RandomCLAHE(p=0.4),           # ISP CLAHE aug — surrogates must learn CLAHE-robust features
    T.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
    T.ToTensor(),
    NORMALIZE,
])

val_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    NORMALIZE,
])


# ── GTSRB subset ─────────────────────────────────────────────────────────────

class FilteredGTSRB(Dataset):
    """GTSRB speed-limit subset, labels remapped to the AU+GTSRB union space."""

    def __init__(self, root: str, split: str = "train", transform=None, download: bool = False):
        self.base      = tvd.GTSRB(root=root, split=split, download=download)
        self.transform = transform
        self.indices   = [
            i for i, (_, cls) in enumerate(self.base._samples)  # noqa: SLF001
            if cls in GTSRB_SPEED_CLASSES
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, gtsrb_cls = self.base[self.indices[idx]]
        label = KMH_TO_LABEL[GTSRB_ID_TO_KMH[gtsrb_cls]]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Australian synthetic subset ───────────────────────────────────────────────

class AUSynthDataset(Dataset):
    """
    ImageFolder-style loader over ./data/aus_synth/<speed>/*.png.
    Call generate_dataset() first (or pass auto_generate=True).
    """

    def __init__(
        self,
        root:          str   = "./data/aus_synth",
        transform             = None,
        val_fraction:  float = 0.15,
        split:         str   = "train",
        auto_generate: bool  = False,
        n_per_class:   int   = 400,
    ):
        self.transform = transform
        root_p = Path(root)

        if auto_generate:
            generate_dataset(root, n_per_class=n_per_class)

        self.samples: list[tuple[str, int]] = []
        for speed in AU_SPEEDS:
            class_dir = root_p / str(speed)
            if not class_dir.exists():
                continue
            label = KMH_TO_LABEL[speed]
            files = sorted(class_dir.glob("*.png"))
            n_val = max(1, int(len(files) * val_fraction))
            if split == "train":
                files = files[n_val:]
            else:
                files = files[:n_val]
            for f in files:
                self.samples.append((str(f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── combined dataloader ───────────────────────────────────────────────────────

def make_dataloaders(
    gtsrb_root:    str   = "./data",
    aus_root:      str   = "./data/aus_synth",
    batch_size:    int   = 64,
    num_workers:   int   = 4,
    download:      bool  = True,
    auto_generate: bool  = True,
    n_aus_per_cls: int   = 400,
):
    """
    Returns (train_loader, val_loader) over the merged GTSRB + AU dataset.

    GTSRB covers: 20 30 50 60 70 80 100 120
    AU synth covers: 5 10 15 20 25 30 40 50 60 70 80 90 100 110
    Overlap classes get both real and synthetic samples — improves robustness.
    """
    train_gtsrb = FilteredGTSRB(gtsrb_root, split="train", transform=train_transforms, download=download)
    val_gtsrb   = FilteredGTSRB(gtsrb_root, split="test",  transform=val_transforms,   download=download)

    train_aus = AUSynthDataset(
        aus_root, transform=train_transforms, split="train",
        auto_generate=auto_generate, n_per_class=n_aus_per_cls,
    )
    val_aus = AUSynthDataset(
        aus_root, transform=val_transforms, split="val",
        auto_generate=False,
    )

    train_ds = ConcatDataset([train_gtsrb, train_aus])
    val_ds   = ConcatDataset([val_gtsrb,   val_aus])

    # build balanced sampler over the merged training set
    gtsrb_labels = [
        KMH_TO_LABEL[GTSRB_ID_TO_KMH[train_gtsrb.base._samples[i][1]]]  # noqa: SLF001
        for i in train_gtsrb.indices
    ]
    aus_labels = [label for _, label in train_aus.samples]
    all_labels = gtsrb_labels + aus_labels

    class_counts   = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts    = np.where(class_counts == 0, 1, class_counts)  # avoid /0
    sample_weights = [1.0 / class_counts[l] for l in all_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
