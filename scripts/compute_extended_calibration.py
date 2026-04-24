"""Day-14 extended calibration + label-shift baselines on cached preds.

Recomputes the Day-13A calibration table plus three additions the 2024-2026
review literature expects alongside ECE: (1) temperature scaling (TS) as a
one-parameter post-hoc baseline, (2) Brier score, (3) NLL. All metrics are
reported pre- and post-TS so the reader can see TS's effect explicitly.

Emits:
  results/calibration_extended_table.md   paper-ready markdown
  results/calibration_extended_table.csv  machine-readable
  results/calibration_extended.json       structured per-config detail

Usage:
  uv run python scripts/compute_extended_calibration.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from src.calibration import chain_rule_probs
from src.calibration_extended import temperature_scale_eval


EXPERIMENTS = [
    ("RETFound x Ordinal CapsNet",       "APTOS",       ["results/ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",     "APTOS",       ["results/mlp_ordinal/seed*/preds_fold*.npz"]),
    ("DINOv2 x Ordinal CapsNet",         "APTOS",       ["results/ordinal_capsnet_dinov2/seed*/preds_fold*.npz"]),
    ("DINOv2 x MLP + K-1 sigmoid",       "APTOS",       ["results/mlp_ordinal_dinov2/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet + LoRA","APTOS",       ["results/lora_ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet",       "Messidor-2",  ["results/messidor2/ordinal_capsnet/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",     "Messidor-2",  ["results/messidor2/mlp_k1_sigmoid/preds_fold*.npz"]),
    ("DINOv2 x Ordinal CapsNet",         "Messidor-2",  ["results/messidor2_dinov2/ordinal_capsnet/preds_fold*.npz"]),
    ("DINOv2 x MLP + K-1 sigmoid",       "Messidor-2",  ["results/messidor2_dinov2/mlp_k1_sigmoid/preds_fold*.npz"]),
]


def pool(globs: list[str]) -> dict | None:
    paths: list[Path] = []
    for g in globs:
        paths.extend(sorted(Path(".").glob(g)))
    if not paths:
        return None
    yts, yps, hls = [], [], []
    for p in paths:
        d = np.load(p)
        yts.append(d["y_true"])
        yps.append(d["y_pred"])
        hls.append(d["head_lengths"])
    return {
        "y_true": np.concatenate(yts).astype(np.int64),
        "y_pred": np.concatenate(yps).astype(np.int64),
        "head_lengths": np.concatenate(hls, axis=0).astype(np.float32),
    }


def main() -> None:
    out_root = Path("results")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, dict]] = []
    for name, dataset, globs in EXPERIMENTS:
        p = pool(globs)
        if p is None:
            print(f"  SKIP {dataset:<12s} {name}")
            continue
        probs, _ = chain_rule_probs(p["head_lengths"][:, :, 1])
        r = temperature_scale_eval(probs, p["y_true"], p["y_pred"])
        rows.append((dataset, name, r))
        print(f"  {dataset:<12s} {name:<40s}  T={r['T']:.3f}  "
              f"ECE {r['ece_pre']:.3f}->{r['ece_post']:.3f}  "
              f"Brier {r['brier_pre']:.3f}->{r['brier_post']:.3f}  "
              f"NLL {r['nll_pre']:.3f}->{r['nll_post']:.3f}")

    # Markdown
    md: list[str] = []
    md.append("# Extended calibration: Temperature Scaling + Brier + NLL")
    md.append("")
    md.append("All rows pool per-sample predictions across every fold / seed for that "
              "config. Temperature T is fitted on a 50% random calibration split "
              "(minimising NLL via golden-section search on log T); ECE / Brier / NLL "
              "are reported pre- and post-TS on the held-out 50%. ECE uses 10 "
              "equal-mass bins on P(y = predicted_grade). Lower is better for every "
              "column. T $>$ 1 means the head was over-confident on the cal split; T "
              "$<$ 1 means under-confident.")
    md.append("")
    md.append("| Dataset | Model | T | ECE pre | ECE post | Brier pre | Brier post | NLL pre | NLL post |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    csv_rows = [["dataset", "model", "T",
                 "ece_pre", "ece_post", "brier_pre", "brier_post", "nll_pre", "nll_post"]]
    for ds, name, r in rows:
        md.append(f"| {ds} | {name} | "
                  f"{r['T']:.3f} | "
                  f"{r['ece_pre']:.3f} | {r['ece_post']:.3f} | "
                  f"{r['brier_pre']:.3f} | {r['brier_post']:.3f} | "
                  f"{r['nll_pre']:.3f} | {r['nll_post']:.3f} |")
        csv_rows.append([ds, name, f"{r['T']:.4f}",
                         f"{r['ece_pre']:.4f}", f"{r['ece_post']:.4f}",
                         f"{r['brier_pre']:.4f}", f"{r['brier_post']:.4f}",
                         f"{r['nll_pre']:.4f}", f"{r['nll_post']:.4f}"])
    (out_root / "calibration_extended_table.md").write_text("\n".join(md) + "\n")
    with (out_root / "calibration_extended_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    (out_root / "calibration_extended.json").write_text(json.dumps(
        [{"dataset": d, "model": n, **r} for d, n, r in rows], indent=2))
    print(f"\nWrote {out_root/'calibration_extended_table.md'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
