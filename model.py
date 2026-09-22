"""
Surrogate model — no HuggingFace dependency.

Bootstrap path (first run, ~15 min on M5 Pro):
    python train.py --pretrain-gtsrb

Subsequent runs load surrogate.pt directly — no internet required.

Architecture:
    GTSRBSurrogate = backbone (43-class GTSRB head)
                   + linear remapping head (43 → NUM_CLASSES AU speeds)

Supported backbones (ARCH_CHOICES):
    resnet18, resnet50               — strong baseline, GTSRB pretrain supported
    mobilenet_v3_small/large         — closest proxy for embedded ADAS (EyeQ4-like)
    efficientnet_b0                  — different feature hierarchy, good ensemble diversity

The remapping head is identity-initialised for overlapping GTSRB/AU classes
and zero-initialised for AU-only classes (5/10/15/25/40/90/110 km/h).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.datasets as tvd
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ALL_SPEEDS, NUM_CLASSES, KMH_TO_LABEL

# ── GTSRB constants ───────────────────────────────────────────────────────────
GTSRB_N_CLASSES = 43

GTSRB_43_SPEED_MAP: dict[int, int] = {
    0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100,
}

GTSRB_TO_AU_LABEL: dict[int, int] = {
    g: KMH_TO_LABEL[kmh]
    for g, kmh in GTSRB_43_SPEED_MAP.items()
}

ARCH_CHOICES = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
]


# ── Model ─────────────────────────────────────────────────────────────────────

class GTSRBSurrogate(nn.Module):
    """
    Backbone (43-class GTSRB output) with a learnable linear head that maps
    43 GTSRB logits → NUM_CLASSES AU speed logits.  Works for any backbone
    in ARCH_CHOICES.
    """

    def __init__(self, backbone: nn.Module, gtsrb_out: int = GTSRB_N_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(gtsrb_out, NUM_CLASSES, bias=False)
        self._init_head()

    def _init_head(self):
        with torch.no_grad():
            self.head.weight.zero_()
            for gtsrb_cls, au_label in GTSRB_TO_AU_LABEL.items():
                self.head.weight[au_label, gtsrb_cls] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── Backbone factory ──────────────────────────────────────────────────────────

def _build_backbone(arch: str, pretrained: bool = True) -> nn.Module:
    """
    Build a backbone with GTSRB_N_CLASSES output head.
    pretrained=True loads ImageNet weights; False gives a bare skeleton for
    loading a saved checkpoint (the final layer is still replaced so the
    state dict shapes match).
    """
    if arch == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, GTSRB_N_CLASSES)
    elif arch == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, GTSRB_N_CLASSES)
    elif arch == "mobilenet_v3_small":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v3_small(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, GTSRB_N_CLASSES)
    elif arch == "mobilenet_v3_large":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, GTSRB_N_CLASSES)
    elif arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_b0(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, GTSRB_N_CLASSES)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {ARCH_CHOICES}")
    return m


# ── Local GTSRB pretrain ──────────────────────────────────────────────────────

def _pretrain_on_gtsrb(
    arch:    str = "resnet18",
    epochs:  int = 5,
    batch:   int = 64,
    lr:      float = 1e-3,
    data_root: str = "./data",
    save_path: str = "./gtsrb_backbone.pt",
) -> nn.Module:
    """
    Train a backbone on torchvision GTSRB from ImageNet weights.
    Saves the backbone state dict to save_path and returns the model.
    ~15 min on M5 Pro, one-time cost.  ResNets recommended; other archs work
    but are not tuned for the 64×64 training resolution used here.
    """
    device = get_device()
    print(f"Pretraining {arch} on GTSRB ({epochs} epochs) → {save_path}")

    tfm = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = tvd.GTSRB(root=data_root, split="train", transform=tfm,  download=True)
    val_ds   = tvd.GTSRB(root=data_root, split="test",  transform=val_tfm, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    model = _build_backbone(arch, pretrained=True).to(device)

    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"GTSRB pretrain {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss   = crit(logits, labels)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += (model(imgs).argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        sched.step()
        print(f"  Epoch {epoch}  train={correct/total:.3f}  val={v_correct/v_total:.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"  Saved GTSRB backbone → {save_path}")
    return model


# ── build_surrogate ───────────────────────────────────────────────────────────

def build_surrogate(
    arch:          str  = "resnet18",
    pretrain:      bool = False,
    pretrain_path: str  = "./gtsrb_backbone.pt",
    pretrain_epochs: int = 5,
    data_root:     str  = "./data",
    **_kwargs,
) -> GTSRBSurrogate:
    """
    Build a GTSRBSurrogate for any arch in ARCH_CHOICES.

    Priority:
      1. Load cached backbone from pretrain_path (fast, no internet)
      2. Run local GTSRB pretrain if pretrain=True (one-time, ~15 min)
      3. Fall back to ImageNet weights (will still provide useful features)
    """
    import os

    if os.path.exists(pretrain_path):
        print(f"Loading GTSRB backbone from cache: {pretrain_path}")
        backbone = _build_backbone(arch, pretrained=False)
        state = torch.load(pretrain_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
    elif pretrain:
        backbone = _pretrain_on_gtsrb(
            arch=arch, epochs=pretrain_epochs,
            data_root=data_root, save_path=pretrain_path,
        )
    else:
        print(f"No pretrain cache — using ImageNet init for {arch}.")
        backbone = _build_backbone(arch, pretrained=True)

    model = GTSRBSurrogate(backbone, gtsrb_out=GTSRB_N_CLASSES)
    print(f"Surrogate ready [{arch}]: {NUM_CLASSES} AU classes → {ALL_SPEEDS}")
    return model


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load(path: str, arch: str = "resnet18") -> GTSRBSurrogate:
    """
    Load a saved surrogate checkpoint (full GTSRBSurrogate state dict).
    Works for any arch in ARCH_CHOICES — pass the same arch used when saving.
    """
    backbone = _build_backbone(arch, pretrained=False)
    model = GTSRBSurrogate(backbone, gtsrb_out=GTSRB_N_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def load_ensemble(
    r18_path:  str = "surrogate.pt",
    r50_path:  str = "surrogate_resnet50.pt",
    mbv3_path: str = "surrogate_mobilenet_v3_small.pt",
) -> list:
    """
    Load R18 + R50 + MBV3 surrogates and return as [r18, r50, mbv3] (all on CPU).

    Checkpoint names follow the patch_attack.py convention:
      surrogate.pt                       ← first arch (resnet18), uses --model directly
      surrogate_resnet50.pt              ← {stem}_{arch}{suffix}
      surrogate_mobilenet_v3_small.pt    ← {stem}_{arch}{suffix}

    Raises FileNotFoundError immediately if any checkpoint is missing — do not
    degrade silently to a smaller ensemble, as that gives false confidence in
    the loss signal and inflates transfer ASR.

    Move to device in the caller:
        models = load_ensemble(...)
        models = [m.to(device) for m in models]
        for m in models:
            m._arch_name = ["resnet18", "resnet50", "mobilenet_v3_small"][models.index(m)]
    """
    import os
    missing = [
        label
        for label, p in [("R18", r18_path), ("R50", r50_path), ("MBV3", mbv3_path)]
        if not os.path.exists(p)
    ]
    if missing:
        paths = dict(R18=r18_path, R50=r50_path, MBV3=mbv3_path)
        detail = "  ".join(f"{k}='{paths[k]}'" for k in missing)
        raise FileNotFoundError(
            f"Ensemble incomplete — missing checkpoint(s): {detail}\n"
            "Train missing surrogates before running ensemble optimisation:\n"
            "  python train.py --arch resnet50 --out surrogate_resnet50.pt\n"
            "  python train.py --arch mobilenet_v3_small --out surrogate_mobilenet_v3_small.pt"
        )

    print(f"Loading R18  surrogate : {r18_path}")
    r18  = load(r18_path,  arch="resnet18")
    r18._arch_name = "resnet18"

    print(f"Loading R50  surrogate : {r50_path}")
    r50  = load(r50_path,  arch="resnet50")
    r50._arch_name = "resnet50"

    print(f"Loading MBV3 surrogate : {mbv3_path}")
    mbv3 = load(mbv3_path, arch="mobilenet_v3_small")
    mbv3._arch_name = "mobilenet_v3_small"

    print(f"Ensemble ready: 3 surrogates (R18 + R50 + MBV3), {NUM_CLASSES} AU classes")
    return [r18, r50, mbv3]
