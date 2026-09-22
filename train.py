"""
Fine-tune the pretrained GTSRB surrogate on Australian synthetic data.

This is OPTIONAL — the pretrained backbone already handles the GTSRB speed
classes well.  Fine-tuning is only needed to teach the model AU-only classes
(5, 10, 15, 25, 40, 90, 110 km/h) that GTSRB never saw.

Strategy:
  1. Load pretrained GTSRB backbone (frozen)
  2. Train only the remapping head for a few epochs on AU synth data
  3. Optionally unfreeze the last block for a few more epochs

Usage:
    # ResNet-18 — load existing GTSRB pretrain cache
    python train.py

    # ResNet-50 — pretrain on GTSRB then fine-tune (~15 min one-time cost)
    python train.py --arch resnet50 --pretrain-gtsrb --out surrogate_resnet50.pt

    # MobileNetV3-Small — pretrain on GTSRB then fine-tune (~15 min)
    python train.py --arch mobilenet_v3_small --pretrain-gtsrb \
        --out surrogate_mobilenet_v3_small.pt

    # EfficientNet-B0
    python train.py --arch efficientnet_b0 --pretrain-gtsrb \
        --out surrogate_efficientnet_b0.pt

    # Skip GTSRB pretrain (faster but lower accuracy)
    python train.py --arch mobilenet_v3_small --epochs 20 --unfreeze-epoch 3 \
        --out surrogate_mobilenet_v3_small.pt
"""
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import make_dataloaders, ALL_SPEEDS, NUM_CLASSES
from model import build_surrogate, get_device, save, ARCH_CHOICES


# Per-architecture parameter patterns to unfreeze in stage 2.
# Keep earlier features frozen — they transfer well from ImageNet/GTSRB.
_UNFREEZE_PATTERNS = {
    "resnet18":            ["layer4", "fc"],
    "resnet50":            ["layer4", "fc"],
    "mobilenet_v3_small":  ["features.12", "classifier"],
    "mobilenet_v3_large":  ["features.16", "classifier"],
    "efficientnet_b0":     ["features.8",  "classifier"],
}


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Derive a per-arch pretrain path so each arch gets its own GTSRB backbone.
    # Avoids trying to load an R18 state dict into MBV3 (shape mismatch crash).
    if args.pretrain_path is None:
        pretrain_path = f"./gtsrb_backbone_{args.arch}.pt"
    else:
        pretrain_path = args.pretrain_path

    train_loader, val_loader = make_dataloaders(
        gtsrb_root=args.data,
        aus_root=args.aus_data,
        batch_size=args.batch,
        num_workers=args.workers,
        download=True,
        auto_generate=True,
        n_aus_per_cls=args.n_aus,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"Classes ({NUM_CLASSES}): {ALL_SPEEDS}")

    model = build_surrogate(
        arch=args.arch,
        pretrain=args.pretrain_gtsrb,
        pretrain_path=pretrain_path,
    ).to(device)
    model.freeze_backbone()

    # Only the remapping head is trained initially
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    unfreeze_patterns = _UNFREEZE_PATTERNS.get(args.arch, ["fc"])
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            print(f"Epoch {epoch}: unfreezing {unfreeze_patterns} for {args.arch}")
            for name, p in model.backbone.named_parameters():
                if any(pat in name for pat in unfreeze_patterns):
                    p.requires_grad = True
            optimizer.add_param_group({
                "params": [
                    p for name, p in model.backbone.named_parameters()
                    if any(pat in name for pat in unfreeze_patterns) and p.requires_grad
                ],
                "lr": args.lr * 0.05,
            })

        # ── train ──
        model.train()
        running_loss = correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(1) == labels).sum().item()
            total        += imgs.size(0)
        train_acc = correct / total

        # ── validate ──
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val  ", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += (model(imgs).argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)
        val_acc = v_correct / v_total

        scheduler.step()

        print(
            f"Epoch {epoch:3d}  loss={running_loss/total:.4f}  "
            f"train={train_acc:.3f}  val={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save(model, args.out)
            print(f"  ↳ saved ({val_acc:.3f}) → {args.out}")

    print(f"\nFine-tune complete. Best val accuracy: {best_acc:.3f}")
    print("If val_acc is low on AU-only classes, increase --n-aus or --epochs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch",           default="resnet18", choices=ARCH_CHOICES)
    p.add_argument("--pretrain-gtsrb", action="store_true", default=False,
                   help="Run GTSRB pretrain before AU fine-tune (~15 min one-time cost). "
                        "Strongly recommended for non-ResNet archs.")
    p.add_argument("--pretrain-path",  default=None,
                   help="Path to GTSRB backbone cache. "
                        "Default: gtsrb_backbone_{arch}.pt (per-arch, avoids shape mismatches)")
    p.add_argument("--epochs",         type=int,   default=8)
    p.add_argument("--unfreeze-epoch", type=int,   default=5,
                   help="Epoch at which to unfreeze the last backbone block")
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3e-3)
    p.add_argument("--data",           type=str,   default="./data")
    p.add_argument("--aus-data",       type=str,   default="./data/aus_synth")
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--n-aus",          type=int,   default=400)
    p.add_argument("--out",            type=str,   default="surrogate.pt")
    train(p.parse_args())
