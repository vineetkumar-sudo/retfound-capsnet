"""Day 8 — IDRiD ablation table + cross-dataset row + per-class breakdown.

Reads each model's summary.json under results/idrid/<model>/ (written by
scripts/run_idrid.py) plus the cross-dataset summary at
results/cross_dataset/aptos_to_idrid/summary.json. Emits:

  results/idrid_final_results.csv      canonical schema (matches Day 7 APTOS one)
  results/idrid_final_table.md         paper-ready markdown with best-in-column bold
  results/idrid_per_class_breakdown.csv  vanilla vs ordinal per-grade (test set)

Usage:
    uv run python scripts/aggregate_day8.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np


# Matches Day 7 canonical ordering; first 4 are within-IDRiD 5-fold CV rows,
# last row is the APTOS→IDRiD cross-dataset ensemble.
ROW_SPECS = [
    ("MLP + CE",                "mlp_ce",               Path("results/idrid/mlp_ce")),
    ("MLP + MSE",               "mlp_mse",              Path("results/idrid/mlp_mse")),
    ("Vanilla CapsNet",         "capsnet_vanilla",      Path("results/idrid/capsnet_vanilla")),
    ("Ordinal CapsNet",         "ordinal_capsnet",      Path("results/idrid/ordinal_capsnet")),
]

CROSS_SPEC = ("APTOS → IDRiD  (Ordinal, ensemble)", "cross_aptos_to_idrid",
              Path("results/cross_dataset/aptos_to_idrid"))


def _load(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def _fmt(m: float, s: float, width: int = 4) -> str:
    if np.isnan(m):
        return "—"
    return f"{m:.{width}f} ± {s:.{width}f}"


def build_rows() -> list[dict]:
    rows = []
    for display, cid, d in ROW_SPECS:
        s = _load(d)
        rows.append({
            "model": cid,
            "display": display,
            "regime": "5-fold CV → IDRiD test (n=103)",
            "qwk_mean":  s["test_qwk_mean"],  "qwk_std":  s["test_qwk_std"],
            "acc_mean":  s["test_accuracy_mean"],  "acc_std":  s["test_accuracy_std"],
            "f1_mean":   s["test_macro_f1_mean"],  "f1_std":   s["test_macro_f1_std"],
            "mae_mean":  s["test_mae_mean"],  "mae_std":  s["test_mae_std"],
            "val_qwk_mean": s["val_qwk_mean"], "val_qwk_std": s["val_qwk_std"],
            "cm_pooled_test": s.get("test_confusion_matrix_pooled"),
        })

    display, cid, d = CROSS_SPEC
    if (d / "summary.json").exists():
        s = _load(d)
        ens = s["ensemble_test"]
        rows.append({
            "model": cid,
            "display": display,
            "regime": "APTOS train → IDRiD test (no IDRiD training)",
            "qwk_mean": ens["qwk"], "qwk_std": s["per_fold_test_qwk_std"],
            "acc_mean": ens["accuracy"], "acc_std": float("nan"),
            "f1_mean":  ens["macro_f1"],  "f1_std": float("nan"),
            "mae_mean": ens["mae"], "mae_std": float("nan"),
            "val_qwk_mean": float("nan"), "val_qwk_std": float("nan"),
            "cm_pooled_test": None,
        })
    return rows


def write_canonical_csv(rows: list[dict], path: Path) -> None:
    cols = ["model", "qwk_mean", "qwk_std", "acc_mean", "acc_std",
            "f1_mean", "f1_std", "mae_mean", "mae_std", "regime"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_md(rows: list[dict], path: Path) -> None:
    # Best-in-column across the 4 within-IDRiD rows only (cross-dataset is an
    # apples-to-oranges row and shouldn't compete for 'best').
    within = rows[:4]
    def _best(key: str, mode: str) -> int:
        vals = [r[key] for r in within]
        idx = int(np.argmax(vals) if mode == "max" else np.argmin(vals))
        return idx
    best_qwk = _best("qwk_mean", "max")
    best_acc = _best("acc_mean", "max")
    best_f1  = _best("f1_mean",  "max")
    best_mae = _best("mae_mean", "min")

    lines = [
        "# IDRiD Disease Grading — Ablation Table",
        "",
        "## Within-IDRiD (5-fold CV on 413 train → official 103 test set)",
        "",
        "| # | Model | Val QWK | Test QWK | Test Accuracy | Test Macro F1 | Test MAE |",
        "|---|-------|---------|----------|---------------|---------------|----------|",
    ]
    for i, r in enumerate(within):
        cells = [
            _fmt(r["qwk_mean"], r["qwk_std"]),
            _fmt(r["acc_mean"], r["acc_std"]),
            _fmt(r["f1_mean"],  r["f1_std"]),
            _fmt(r["mae_mean"], r["mae_std"], width=3),
        ]
        if i == best_qwk: cells[0] = f"**{cells[0]}**"
        if i == best_acc: cells[1] = f"**{cells[1]}**"
        if i == best_f1:  cells[2] = f"**{cells[2]}**"
        if i == best_mae: cells[3] = f"**{cells[3]}**"
        lines.append(f"| {i+1} | {r['display']} | "
                     f"{_fmt(r['val_qwk_mean'], r['val_qwk_std'])} | "
                     f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")
    lines.append("## Cross-dataset generalization")
    lines.append("")
    lines.append("| Evaluation | Test QWK | Test Accuracy | Test Macro F1 | Test MAE |")
    lines.append("|------------|----------|---------------|---------------|----------|")
    if len(rows) > 4:
        r = rows[4]
        lines.append(
            f"| {r['display']} | "
            f"{_fmt(r['qwk_mean'], r['qwk_std'])} | "
            f"{r['acc_mean']:.4f} | {r['f1_mean']:.4f} | {r['mae_mean']:.3f} |"
        )
    lines.append("")
    lines.append("Best value per column (within-IDRiD rows only) in **bold**. "
                 "Cross-dataset row uses std across APTOS folds.")
    path.write_text("\n".join(lines))


def per_class_breakdown_csv(rows: list[dict], path: Path) -> None:
    """Per-grade accuracy from pooled test confusion matrices."""
    vanilla = next((r for r in rows if r["model"] == "capsnet_vanilla"), None)
    ordinal = next((r for r in rows if r["model"] == "ordinal_capsnet"), None)
    if not (vanilla and ordinal) or not (vanilla["cm_pooled_test"] and ordinal["cm_pooled_test"]):
        print("[per_class_breakdown] skipped (missing pooled CMs)")
        return
    cm_v = np.asarray(vanilla["cm_pooled_test"], dtype=float)
    cm_o = np.asarray(ordinal["cm_pooled_test"], dtype=float)
    classes = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Grade", "Vanilla_%", "Ordinal_%", "Delta_pp", "Note"])
        w.writeheader()
        print("\nIDRiD per-class test accuracy:")
        for g in range(5):
            v_total = cm_v[g].sum()
            o_total = cm_o[g].sum()
            v = (cm_v[g, g] / v_total * 100) if v_total else float("nan")
            o = (cm_o[g, g] / o_total * 100) if o_total else float("nan")
            d = o - v
            if np.isnan(d):
                note = "No samples"
            elif d > 5:   note = "Major improvement"
            elif d > 1:   note = "Improvement"
            elif d > -1:  note = "Stable"
            elif d > -5:  note = "Minor drop"
            else:         note = "Regression"
            row = {
                "Grade": f"{g} ({classes[g]})",
                "Vanilla_%": f"{v:.1f}", "Ordinal_%": f"{o:.1f}",
                "Delta_pp": f"{d:+.1f}" if not np.isnan(d) else "—",
                "Note": note,
            }
            w.writerow(row)
            print(f"  {row['Grade']:<14s}  Vanilla={row['Vanilla_%']}%  Ordinal={row['Ordinal_%']}%  "
                  f"delta={row['Delta_pp']}  ({row['Note']})")


def main() -> None:
    out = Path("results")
    rows = build_rows()
    write_canonical_csv(rows, out / "idrid_final_results.csv")
    write_md(rows, out / "idrid_final_table.md")
    per_class_breakdown_csv(rows, out / "idrid_per_class_breakdown.csv")

    # Console summary
    print("\nIDRiD results table:")
    print(f"{'#':<3}{'Model':<36s}{'Val QWK':<20s}{'Test QWK':<20s}{'Test Acc':<16s}{'MAE':<10s}")
    for i, r in enumerate(rows):
        print(f"{i+1:<3}{r['display']:<36s}"
              f"{_fmt(r['val_qwk_mean'], r['val_qwk_std']):<20s}"
              f"{_fmt(r['qwk_mean'], r['qwk_std']):<20s}"
              f"{_fmt(r['acc_mean'], r['acc_std']):<16s}"
              f"{r['mae_mean']:.3f}")

    print(f"\nWrote:\n  {out}/idrid_final_results.csv\n  {out}/idrid_final_table.md\n  "
          f"{out}/idrid_per_class_breakdown.csv")


if __name__ == "__main__":
    main()
