"""
Convert patch.pt → print-ready PNG at a specified physical size.

Prints at 100% scale on any printer = exact size patch.

Usage:
    python3 pt_to_printable.py                              # 10cm square @ 300 DPI
    python3 pt_to_printable.py --patch best_patch.pt --print-cm 8.0
    python3 pt_to_printable.py --dpi 600                    # higher res print
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main(args):
    path = Path(args.patch)
    dpi = args.dpi
    target_px = int(args.print_cm / 2.54 * dpi)

    if path.suffix == ".pt":
        t = torch.load(str(path), map_location="cpu", weights_only=True)
        if t.dim() == 4:
            t = t.squeeze(0)
        arr = t.permute(1, 2, 0).clamp(0, 1).numpy()
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        print(f"Loaded tensor: {path}  ({img.size[0]}×{img.size[1]} px)")
    else:
        img = Image.open(str(path)).convert("RGB")
        print(f"Loaded image: {path}  ({img.size[0]}×{img.size[1]} px)")

    # Resize to exact physical size at target DPI
    img = img.resize((target_px, target_px), Image.LANCZOS)

    out = Path(args.out)
    img.save(str(out), dpi=(dpi, dpi))
    print(f"Saved: {out}  ({target_px}×{target_px} px, {args.print_cm} cm @ {dpi} DPI)")
    print(f"Print at 100% scale — do not fit to page.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch",    default="patch.pt")
    p.add_argument("--print-cm", type=float, default=10.0,
                   help="Physical side length in cm (default 10)")
    p.add_argument("--dpi",      type=int, default=300)
    p.add_argument("--out",      default="patch_printable.png")
    main(p.parse_args())
