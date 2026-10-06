"""
Overlay the optimised patch on a 5 km/h sign, apply EOT, and report per-model
predictions across N EOT samples.  Includes multi-scale evaluation,
temporal majority voting simulation, and class distribution histograms.

Supports both targeted (ASR@80) and untargeted (any misclassification) modes.

Usage:
    python eval_patch.py --loss untargeted
    python eval_patch.py --loss untargeted --oblique-eot --temporal-frames 11
    python eval_patch.py --patch intermediate_examples/patch_best.pt --n-eot 64
    python eval_patch.py --multi-scale --loss untargeted
"""
import argparse
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image

from dataset import AUSynthDataset, KMH_TO_LABEL, ALL_SPEEDS, val_transforms
from eot import eot_batch
from model import get_device, load as load_model
from patch_attack import apply_patch

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SURROGATE_CONFIGS = {
    "mobilenet_v3_large":  ("surrogate_mobilenet_v3_large.pt",  "mobilenet_v3_large"),
    "efficientnet_b0":     ("surrogate_efficientnet_b0.pt",     "efficientnet_b0"),
    "shufflenet_v2_x1_0":  ("surrogate_shufflenet_v2_x1_0.pt", "shufflenet_v2_x1_0"),
    "resnet18":            ("surrogate.pt",                     "resnet18"),
}


def print_histogram(votes: dict, n_eot: int, true_label: int, title: str):
    """Print a class-distribution bar chart from aggregated votes."""
    total_votes = [0] * len(ALL_SPEEDS)
    for v in votes.values():
        for i, c in enumerate(v):
            total_votes[i] += c
    total = sum(total_votes)
    ranked = sorted(enumerate(total_votes), key=lambda x: -x[1])
    peak = max(total_votes) if max(total_votes) > 0 else 1
    bar_max = 40

    print(f"\n── {title} (all models, {n_eot} EOT × {len(votes)} models = {total} votes) ──")
    for idx, count in ranked:
        if count == 0:
            continue
        pct = count / total
        bar = "█" * max(1, int(bar_max * count / peak))
        true_marker = " ◄ true" if idx == true_label else ""
        print(f"  {ALL_SPEEDS[idx]:>3} km/h  {bar}  {count:>4} ({pct:>5.1%}){true_marker}")


def main(args):
    device = get_device()
    mean = _MEAN.to(device)
    std  = _STD.to(device)

    untargeted = (args.loss == "untargeted")

    # ── load models ──────────────────────────────────────────────────────────
    models = {}
    for name, (path, arch) in SURROGATE_CONFIGS.items():
        try:
            m = load_model(path, arch=arch).to(device)
            m.eval()
            models[name] = m
            print(f"Loaded  {name:<25}  ({path})")
        except Exception as e:
            print(f"Warning: could not load {name}: {e}")

    if not models:
        raise RuntimeError("No models loaded.")

    # ── load patch ───────────────────────────────────────────────────────────
    if args.patch.endswith(".pt"):
        patch_01 = torch.load(args.patch, map_location="cpu", weights_only=True)
    else:
        arr = np.array(Image.open(args.patch).convert("RGB")).astype(np.float32) / 255.0
        patch_01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    patch_01 = patch_01.to(device)  # (1, 3, P, P)
    print(f"Patch loaded : {args.patch}  [{patch_01.shape[-1]}px]")

    # ── compute model-input footprint (same formula as optimise_patch) ───────
    target_patch_px = max(4, int(int(224 * 0.80) * (args.print_cm * 10) / args.sign_diam_mm))
    print(f"Patch footprint in model input : {target_patch_px}px")

    # ── multi-scale footprints (simulate different viewing distances) ─────────
    if args.multi_scale:
        scale_factors = [0.67, 0.80, 1.0, 1.25, 1.5]
        multi_patch_px = [max(4, int(target_patch_px / s)) for s in scale_factors]
        print(f"Multi-scale footprints: {list(zip(scale_factors, multi_patch_px))}")
    else:
        multi_patch_px = [target_patch_px]

    # ── get a sign image ─────────────────────────────────────────────────────
    true_label = KMH_TO_LABEL[args.true_speed]
    target_pred = KMH_TO_LABEL[80]

    if args.sign:
        sign_img = val_transforms(Image.open(args.sign).convert("RGB")).unsqueeze(0).to(device)
        print(f"Sign image   : {args.sign}")
    else:
        for split in ("val", "train"):
            ds = AUSynthDataset(args.aus_data, transform=val_transforms,
                                split=split, auto_generate=False)
            candidates = [img for img, lbl in ds if lbl == true_label]
            if candidates:
                break
        if not candidates:
            raise RuntimeError(
                f"No {args.true_speed} km/h samples found in {args.aus_data}. "
                "Pass --sign path/to/sign.jpg instead."
            )
        sign_img = random.choice(candidates).unsqueeze(0).to(device)
        print(f"Sign image   : random {args.true_speed} km/h sample from AU synth ({split} split)")

    mode_str = "untargeted, misclassify" if untargeted else "targeted@80"
    print(f"True class   : {ALL_SPEEDS[true_label]} km/h  (label {true_label})")
    print(f"Loss mode    : {mode_str}")

    # ── baseline (no patch) ──────────────────────────────────────────────────
    with torch.no_grad():
        print(f"\n── Baseline (no patch) ──")
        for name, m in models.items():
            pred = m(sign_img).argmax(1).item()
            print(f"  {name:<25}  {ALL_SPEEDS[pred]:>5} km/h")

    # ── patched + EOT vote ───────────────────────────────────────────────────
    patch_norm = (patch_01 - mean) / std
    votes = {name: [0] * len(ALL_SPEEDS) for name in models}

    with torch.no_grad():
        for _ in range(args.n_eot):
            tpp = random.choice(multi_patch_px)
            patched    = apply_patch(sign_img, patch_norm,
                                     randomise_placement=False,
                                     target_patch_px=tpp)
            patched_01 = patched * std + mean
            patched_01 = eot_batch(patched_01.clone(), oblique=args.oblique_eot)
            patched_eot = (patched_01 - mean) / std

            for name, m in models.items():
                votes[name][m(patched_eot).argmax(1).item()] += 1

    # ── per-model table ──────────────────────────────────────────────────────
    print(f"\n── Patched + EOT ({args.n_eot} samples, {mode_str}) ──")
    if untargeted:
        print(f"  {'Model':<25}  {'Top pred':>9}  {'Conf':>6}  {'ASR(≠5)':>8}")
    else:
        print(f"  {'Model':<25}  {'Top pred':>9}  {'Conf':>6}  {'ASR@80':>7}")
    print("  " + "─" * 56)

    worst_asr = 1.0
    for name, v in votes.items():
        best = max(range(len(v)), key=lambda i: v[i])
        conf = v[best] / args.n_eot
        if untargeted:
            asr = 1.0 - (v[true_label] / args.n_eot)
        else:
            asr = v[target_pred] / args.n_eot
        hit = " ✓" if (untargeted and best != true_label) or (not untargeted and best == target_pred) else ""
        worst_asr = min(worst_asr, asr)
        print(f"  {name:<25}  {ALL_SPEEDS[best]:>5} km/h  {conf:>5.1%}  {asr:>6.1%}{hit}")

    label = "Ensemble worst-case ASR(≠5)" if untargeted else "Ensemble worst-case ASR@80"
    print(f"  {label:<42}  {worst_asr:>6.1%}")

    # ── class distribution histogram ─────────────────────────────────────────
    print_histogram(votes, args.n_eot, true_label, "Class distribution")

    # ── INT8 quantised evaluation ────────────────────────────────────────────
    if not args.no_quant:
        from patch_attack import _build_quant_eval_hooks

        q_votes = {name: [0] * len(ALL_SPEEDS) for name in models}
        for name, m in models.items():
            hooks = _build_quant_eval_hooks(m)
            try:
                with torch.no_grad():
                    for _ in range(args.n_eot):
                        tpp = random.choice(multi_patch_px)
                        patched = apply_patch(sign_img, patch_norm,
                                              randomise_placement=False,
                                              target_patch_px=tpp)
                        patched_01 = patched * std + mean
                        patched_01 = eot_batch(patched_01.clone(), oblique=args.oblique_eot)
                        patched_eot = (patched_01 - mean) / std
                        q_votes[name][m(patched_eot).argmax(1).item()] += 1
            finally:
                for h in hooks:
                    h.remove()

        print(f"\n── INT8 quantised EOT vote ({args.n_eot} samples, {mode_str}) ──")
        if untargeted:
            print(f"  {'Model':<25}  {'Top pred':>9}  {'Conf':>6}  {'qASR(≠5)':>9}")
        else:
            print(f"  {'Model':<25}  {'Top pred':>9}  {'Conf':>6}  {'qASR@80':>8}")
        print("  " + "─" * 56)

        worst_qasr = 1.0
        for name, v in q_votes.items():
            best = max(range(len(v)), key=lambda i: v[i])
            conf = v[best] / args.n_eot
            if untargeted:
                qasr = 1.0 - (v[true_label] / args.n_eot)
            else:
                qasr = v[target_pred] / args.n_eot
            hit = " ✓" if (untargeted and best != true_label) or (not untargeted and best == target_pred) else ""
            worst_qasr = min(worst_qasr, qasr)
            print(f"  {name:<25}  {ALL_SPEEDS[best]:>5} km/h  {conf:>5.1%}  {qasr:>6.1%}{hit}")

        qlabel = "Ensemble worst-case qASR(≠5)" if untargeted else "Ensemble worst-case qASR@80"
        print(f"  {qlabel:<42}  {worst_qasr:>6.1%}")

        print_histogram(q_votes, args.n_eot, true_label, "INT8 class distribution")

    # ── temporal majority voting simulation ──────────────────────────────────
    if args.temporal_frames > 1:
        n_frames = args.temporal_frames
        n_trials = args.temporal_trials

        def _run_temporal(model_dict, quant=False, label_prefix=""):
            from patch_attack import _build_quant_eval_hooks

            prefix = "INT8 " if quant else ""
            asr_col = f"{prefix}Temporal ASR(≠5)" if untargeted else f"{prefix}Temporal ASR@80"
            qasr_tag = "q" if quant else ""

            print(f"\n── {prefix}Temporal Majority Voting ({n_frames} frames × {n_trials} trials, {mode_str}) ──")
            print(f"  Simulates frame-level voting: attack succeeds if majority")
            print(f"  vote result ≠ true class (untargeted) or = target (targeted).\n")

            temporal_wins = {name: 0 for name in model_dict}
            temporal_class_counts = {name: Counter() for name in model_dict}

            for name, m in model_dict.items():
                hooks = _build_quant_eval_hooks(m) if quant else []
                try:
                    with torch.no_grad():
                        for trial in range(n_trials):
                            frame_preds = []
                            for _ in range(n_frames):
                                tpp = random.choice(multi_patch_px)
                                patched = apply_patch(sign_img, patch_norm,
                                                      randomise_placement=False,
                                                      target_patch_px=tpp)
                                patched_01 = patched * std + mean
                                patched_01 = eot_batch(patched_01.clone(), oblique=args.oblique_eot)
                                patched_eot = (patched_01 - mean) / std
                                pred = m(patched_eot).argmax(1).item()
                                frame_preds.append(pred)

                            vote_result = Counter(frame_preds).most_common(1)[0][0]
                            temporal_class_counts[name][vote_result] += 1

                            if untargeted:
                                if vote_result != true_label:
                                    temporal_wins[name] += 1
                            else:
                                if vote_result == target_pred:
                                    temporal_wins[name] += 1
                finally:
                    for h in hooks:
                        h.remove()

            print(f"  {'Model':<25}  {asr_col:>24}")
            print("  " + "─" * 52)
            for name, wins in temporal_wins.items():
                t_asr = wins / n_trials
                print(f"  {name:<25}  {t_asr:>23.1%}")
            worst_temporal = min(w / n_trials for w in temporal_wins.values())
            wlabel = f"worst-case {prefix}temporal {qasr_tag}ASR(≠5)" if untargeted else f"worst-case {prefix}temporal {qasr_tag}ASR@80"
            print(f"\n  Ensemble {wlabel} : {worst_temporal:.1%}")

            # Temporal vote class histogram
            total_temporal = Counter()
            for c in temporal_class_counts.values():
                total_temporal.update(c)
            total_t = sum(total_temporal.values())
            peak_t = max(total_temporal.values()) if total_temporal else 1
            bar_max = 40
            print(f"\n── {prefix}Temporal vote class distribution ({n_trials} trials × {len(model_dict)} models = {total_t} votes) ──")
            for idx, count in total_temporal.most_common():
                pct = count / total_t
                bar = "█" * max(1, int(bar_max * count / peak_t))
                true_marker = " ◄ true" if idx == true_label else ""
                print(f"  {ALL_SPEEDS[idx]:>3} km/h  {bar}  {count:>4} ({pct:>5.1%}){true_marker}")

        # Float temporal voting
        _run_temporal(models, quant=False)

        # INT8 temporal voting
        if not args.no_quant:
            _run_temporal(models, quant=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch",         default="patch.pt",
                   help="Patch file (.pt tensor or .png image)")
    p.add_argument("--sign",          default=None,
                   help="Path to sign image. Omit to sample from AU synth dataset.")
    p.add_argument("--true-speed",    type=int,   default=5,
                   help="True speed class of the sign (km/h)")
    p.add_argument("--aus-data",      default="./data/aus_synth")
    p.add_argument("--n-eot",         type=int,   default=32,
                   help="Number of EOT samples for the vote")
    p.add_argument("--print-cm",      type=float, default=10.0,
                   help="Physical patch side length in cm (default 10)")
    p.add_argument("--sign-diam-mm",  type=float, default=450.0,
                   help="Physical sign diameter in mm (AU=450)")
    p.add_argument("--loss",          default="untargeted",
                   choices=["targeted", "untargeted"],
                   help="'untargeted': any misclassification = success; "
                        "'targeted': only 80 km/h counts")
    p.add_argument("--oblique-eot",   action="store_true", default=False,
                   help="Mix in extreme oblique viewing angles")
    p.add_argument("--multi-scale",   action="store_true", default=False,
                   help="Evaluate at multiple viewing distances (0.67x to 1.5x)")
    p.add_argument("--no-quant",      action="store_true", default=False,
                   help="Skip INT8 quantised evaluation")
    p.add_argument("--temporal-frames", type=int, default=1,
                   help="Frames in temporal voting window (1 = disabled)")
    p.add_argument("--temporal-trials", type=int, default=50,
                   help="Number of independent voting windows to simulate")
    main(p.parse_args())