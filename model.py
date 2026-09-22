"""
Surrogate model — no HuggingFace dependency.

Bootstrap path (first run, ~15 min on M5 Pro):
    python train.py --pretrain-gtsrb

Subsequent runs load surrogate.pt directly — no internet required.

Architecture:
    GTSRBSurrogate = ResNet-18 backbone (43-class GTSRB head)
                   + linear remapping head (43 → NUM_CLASSES AU speeds)

The remapping head is identity-initialised for overlapping GTSRB/AU classes
and zero-initialised for AU-only classes (5/10/15/25/40/90/110 km/h).
Those learn during fine-tuning on AU synthetic data.
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


# ── Model ─────────────────────────────────────────────────────────────────────

class GTSRBSurrogate(nn.Module):
    """
    ResNet-18/50 backbone (43-class GTSRB output) with a learnable linear head
    that maps 43 GTSRB logits → NUM_CLASSES AU speed logits.
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
    Train a ResNet on torchvision GTSRB from ImageNet weights.
    Saves the backbone state dict to save_path and returns the model.
    ~15 min on M5 Pro, one-time cost.
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

    if arch == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, GTSRB_N_CLASSES)
    model = model.to(device)

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
    # legacy kwargs silently accepted so old callers don't crash
    **_kwargs,
) -> GTSRBSurrogate:
    """
    Build a GTSRBSurrogate.

    Priority:
      1. Load cached backbone from pretrain_path (fast, no internet)
      2. Run local GTSRB pretrain if pretrain=True (one-time, ~15 min)
      3. Fall back to random ImageNet weights (will still fine-tune OK)
    """
    import os

    if arch == "resnet18":
        backbone = tvm.resnet18(weights=None)
    else:
        backbone = tvm.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, GTSRB_N_CLASSES)

    if os.path.exists(pretrain_path):
        print(f"Loading GTSRB backbone from cache: {pretrain_path}")
        state = torch.load(pretrain_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
    elif pretrain:
        backbone = _pretrain_on_gtsrb(
            arch=arch, epochs=pretrain_epochs,
            data_root=data_root, save_path=pretrain_path,
        )
    else:
        print("No pretrain cache found and --pretrain-gtsrb not set — using ImageNet init.")
        if arch == "resnet18":
            backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, GTSRB_N_CLASSES)

    model = GTSRBSurrogate(backbone, gtsrb_out=GTSRB_N_CLASSES)
    print(f"Surrogate ready: {NUM_CLASSES} AU classes → {ALL_SPEEDS}")
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
    Load a saved surrogate checkpoint.
    Builds a bare skeleton (no pretrained weights needed) and loads the state dict.
    surrogate.pt contains everything — no internet access required.
    """
    if arch == "resnet18":
        backbone = tvm.resnet18(weights=None)
    else:
        backbone = tvm.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, GTSRB_N_CLASSES)
    model = GTSRBSurrogate(backbone, gtsrb_out=GTSRB_N_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model