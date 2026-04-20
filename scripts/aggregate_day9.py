"""Day 9 — Messidor-2 aggregation: within-dataset + APTOS→Messidor-2 cross-dataset.

Emits:
  results/messidor2_final_results.csv        canonical schema (matches Day 7)
  results/messidor2_final_table.md           paper-ready markdown, two sections
  results/messidor2_binary_results.csv       Sens/Spec/AUC/F1 cross-dataset binary
  results/messidor2_per_class_breakdown.csv  Vanilla vs Ordinal per-grade (within)

Usage:
    uv run python scripts/aggregate_day9.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from scipy.stats import mannwhitneyu
import torch

from src.uncertainty import prediction_margin


WITHIN_MODELS = [
    ("MLP + CE",               "mlp_ce"),
    ("MLP + MSE",              "mlp_mse"),
    ("Vanilla CapsNet",        "capsnet_vanilla"),
    ("Ordinal CapsNet",        "ordinal_capsnet"),
]
CROSS_MODELS = [
    ("APTOS → Messidor-2  MLP + CE",          "cross_mlp_ce",           "mlp_ce"),
    ("APTOS → Messidor-2  MLP + MSE",         "cross_mlp_mse",          "mlp_mse"),
    ("APTOS → Messidor-2  Vanilla CapsNet",   "cross_capsnet_vanilla",  "capsnet_vanilla"),
    ("APTOS → Messidor-2  Ordinal CapsNet",   "cross_ordinal_capsnet",  "ordinal_capsnet"),
]

WITHIN_ROOT = Path("results/messidor2")
CROSS_ROOT = Path("results/cross_dataset/aptos_to_messidor2")
OUT = Path("results")


def _fmt(m: float, s: float, width: int = 4) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "—"
    if s is None or np.isnan(s):
        return f"{m:.{width}f}"
    return f"{m:.{width}f} ± {s:.{width}f}"


def _load(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


# ---------------------------------------------------------------------------
# Within-Messidor-2 rows
# ---------------------------------------------------------------------------

def within_rows() -> list[dict]:
    rows = []
    for display, cid in WITHIN_MODELS:
        s = _load(WITHIN_ROOT / cid)
        rows.append({
            "model": cid, "display": display,
            "regime": "5-fold CV within Messidor-2 (n=1744)",
            "qwk_mean":  s["val_qwk_mean"],  "qwk_std":  s["val_qwk_std"],
            "acc_mean":  s["val_accuracy_mean"], "acc_std":  s["val_accuracy_std"],
            "f1_mean":   s["val_macro_f1_mean"],  "f1_std":   s["val_macro_f1_std"],
            "mae_mean":  s["val_mae_mean"],       "mae_std":  s["val_mae_std"],
            "cm_pooled": s.get("val_confusion_matrix_pooled"),
        })
    return rows


# ---------------------------------------------------------------------------
# Cross-dataset rows (APTOS → Messidor-2)
# ---------------------------------------------------------------------------

def cross_rows() -> list[dict]:
    rows = []
    for display, cid, subdir in CROSS_MODELS:
        s = _load(CROSS_ROOT / subdir)
        # 5-class metrics reported as ENSEMBLE (single value); std across folds kept for reference
        e5 = s["ensemble_5class"]
        eb = s["ensemble_binary"]
        per_fold_qwk = s["per_fold_5class_qwk"]
        per_fold_auc = s["per_fold_binary_auc"]
        rows.append({
            "model": cid, "display": display,
            "regime": "APTOS train → Messidor-2 test (ensemble of 5 folds)",
            "qwk_mean": e5["qwk"], "qwk_std": float(np.std(per_fold_qwk)),
            "acc_mean": e5["accuracy"], "acc_std": float("nan"),
            "f1_mean":  e5["macro_f1"],  "f1_std":  float("nan"),
            "mae_mean": e5["mae"],       "mae_std": float("nan"),
            "binary": eb, "per_fold_auc": per_fold_auc,
        })
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_canonical_csv(rows: list[dict], path: Path) -> None:
    cols = ["model", "qwk_mean", "qwk_std", "acc_mean", "acc_std",
            "f1_mean", "f1_std", "mae_mean", "mae_std", "regime"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_md(within: list[dict], cross: list[dict], path: Path) -> None:
    def _best(rows, key, mode):
        vals = [r[key] for r in rows]
        return int(np.argmax(vals) if mode == "max" else np.argmin(vals))

    bw = {"qwk": _best(within, "qwk_mean", "max"),
          "acc": _best(within, "acc_mean", "max"),
          "f1":  _best(within, "f1_mean",  "max"),
          "mae": _best(within, "mae_mean", "min")}
    bc = {"qwk": _best(cross, "qwk_mean", "max"),
          "acc": _best(cross, "acc_mean", "max"),
          "f1":  _best(cross, "f1_mean",  "max"),
          "mae": _best(cross, "mae_mean", "min"),
          "auc": int(np.argmax([r["binary"]["auc"] for r in cross]))}

    lines = [
        "# Messidor-2 Disease Grading — Results",
        "",
        "## Within-Messidor-2 (5-fold CV on 1,744 gradable images)",
        "",
        "| # | Model | QWK | Accuracy | Macro F1 | MAE |",
        "|---|-------|-----|----------|----------|-----|",
    ]
    for i, r in enumerate(within):
        cells = [
            _fmt(r["qwk_mean"], r["qwk_std"]),
            _fmt(r["acc_mean"], r["acc_std"]),
            _fmt(r["f1_mean"],  r["f1_std"]),
            _fmt(r["mae_mean"], r["mae_std"], width=3),
        ]
        if i == bw["qwk"]: cells[0] = f"**{cells[0]}**"
        if i == bw["acc"]: cells[1] = f"**{cells[1]}**"
        if i == bw["f1"]:  cells[2] = f"**{cells[2]}**"
        if i == bw["mae"]: cells[3] = f"**{cells[3]}**"
        lines.append(f"| {i+1} | {r['display']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    lines += ["", "## APTOS → Messidor-2 (cross-dataset, no Messidor-2 training)", ""]
    lines += [
        "| # | Model | QWK | Accuracy | Macro F1 | MAE | Binary AUC |",
        "|---|-------|-----|----------|----------|-----|------------|",
    ]
    for i, r in enumerate(cross):
        cells = [
            _fmt(r["qwk_mean"], r["qwk_std"]),
            f"{r['acc_mean']:.4f}",
            f"{r['f1_mean']:.4f}",
            f"{r['mae_mean']:.3f}",
            f"{r['binary']['auc']:.4f}",
        ]
        if i == bc["qwk"]: cells[0] = f"**{cells[0]}**"
        if i == bc["acc"]: cells[1] = f"**{cells[1]}**"
        if i == bc["f1"]:  cells[2] = f"**{cells[2]}**"
        if i == bc["mae"]: cells[3] = f"**{cells[3]}**"
        if i == bc["auc"]: cells[4] = f"**{cells[4]}**"
        display_short = r["display"].replace("APTOS → Messidor-2  ", "")
        lines.append(f"| {i+1} | {display_short} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |")

    lines += [
        "",
        "Best value per column in **bold** (within each section independently). "
        "QWK / Accuracy / Macro F1 / AUC: higher is better; MAE: lower is better. "
        "Cross-dataset std columns are omitted because a single ensemble prediction is reported; "
        "per-fold stds are tracked in the canonical CSV.",
    ]
    path.write_text("\n".join(lines))


def write_binary_csv(cross: list[dict], path: Path) -> None:
    cols = ["model", "sensitivity", "specificity", "auc", "f1",
            "threshold", "youden_threshold", "youden_sensitivity", "youden_specificity"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in cross:
            b = r["binary"]
            w.writerow({
                "model": r["model"],
                "sensitivity": b["sensitivity"], "specificity": b["specificity"],
                "auc": b["auc"], "f1": b["f1"],
                "threshold": b["threshold"],
                "youden_threshold": b["youden_threshold"],
                "youden_sensitivity": b["youden_sensitivity"],
                "youden_specificity": b["youden_specificity"],
            })


def per_class_csv(within: list[dict], path: Path) -> None:
    vanilla = next(r for r in within if r["model"] == "capsnet_vanilla")
    ordinal = next(r for r in within if r["model"] == "ordinal_capsnet")
    if not (vanilla["cm_pooled"] and ordinal["cm_pooled"]):
        print("[per_class_breakdown] skipped (pooled CMs missing)")
        return
    cm_v = np.asarray(vanilla["cm_pooled"], dtype=float)
    cm_o = np.asarray(ordinal["cm_pooled"], dtype=float)
    classes = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    print("\nMessidor-2 within-dataset per-class accuracy:")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Grade", "Vanilla_%", "Ordinal_%", "Delta_pp", "Note"])
        w.writeheader()
        for g in range(5):
            vt, ot = cm_v[g].sum(), cm_o[g].sum()
            v = (cm_v[g, g] / vt * 100) if vt else float("nan")
            o = (cm_o[g, g] / ot * 100) if ot else float("nan")
            d = o - v
            if np.isnan(d):       note = "No samples"
            elif d > 5:           note = "Major improvement"
            elif d > 1:           note = "Improvement"
            elif d > -1:          note = "Stable"
            elif d > -5:          note = "Minor drop"
            else:                 note = "Regression"
            row = {
                "Grade": f"{g} ({classes[g]})",
                "Vanilla_%": f"{v:.1f}", "Ordinal_%": f"{o:.1f}",
                "Delta_pp": f"{d:+.1f}", "Note": note,
            }
            w.writerow(row)
            print(f"  {row['Grade']:<14s}  Vanilla={v:.1f}%  Ordinal={o:.1f}%  delta={d:+.1f}  ({note})")


# ---------------------------------------------------------------------------
# UQ Mann-Whitney (within + cross)
# ---------------------------------------------------------------------------

def uq_mann_whitney() -> dict:
    """Compute prediction-margin MW p-value on both within and cross-dataset ordinal preds."""
    out: dict[str, dict] = {}

    # Within
    files = sorted((WITHIN_ROOT / "ordinal_capsnet").glob("preds_fold*.npz"))
    if files:
        yts, yps, hls = [], [], []
        for f in files:
            d = np.load(f)
            yts.append(d["y_true"]); yps.append(d["y_pred"])
            hls.append(d["head_lengths"])
        y_true = np.concatenate(yts); y_pred = np.concatenate(yps)
        head_lengths = torch.from_numpy(np.concatenate(hls, axis=0))
        pm = prediction_margin(head_lengths).numpy()
        correct = y_true == y_pred
        if correct.sum() and (~correct).sum():
            _, p = mannwhitneyu(pm[~correct], pm[correct], alternative="greater")
            out["within"] = {"n_correct": int(correct.sum()),
                             "n_misclassified": int((~correct).sum()),
                             "p_value": float(p)}
    # Cross
    files = sorted((CROSS_ROOT / "ordinal_capsnet").glob("preds_fold*.npz"))
    if files:
        yts, yps, hls = [], [], []
        for f in files:
            d = np.load(f)
            yts.append(d["y_true_5class"]); yps.append(d["y_pred_5class"])
            hls.append(d["head_lengths"])
        y_true = np.concatenate(yts); y_pred = np.concatenate(yps)
        head_lengths = torch.from_numpy(np.concatenate(hls, axis=0))
        pm = prediction_margin(head_lengths).numpy()
        correct = y_true == y_pred
        if correct.sum() and (~correct).sum():
            _, p = mannwhitneyu(pm[~correct], pm[correct], alternative="greater")
            out["cross"] = {"n_correct": int(correct.sum()),
                            "n_misclassified": int((~correct).sum()),
                            "p_value": float(p)}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    within = within_rows()
    cross = cross_rows()

    # Canonical CSV: within + cross rows (same column schema; regime distinguishes)
    canonical = within + cross
    # For CSV we emit plain-row dicts only (drop Python objects like cm_pooled / binary / per_fold_auc)
    clean = [{k: v for k, v in r.items() if not isinstance(v, (list, dict)) or k == "regime"}
             for r in canonical]
    write_canonical_csv(clean, OUT / "messidor2_final_results.csv")
    write_md(within, cross, OUT / "messidor2_final_table.md")
    write_binary_csv(cross, OUT / "messidor2_binary_results.csv")
    per_class_csv(within, OUT / "messidor2_per_class_breakdown.csv")

    uq = uq_mann_whitney()
    print("\nUQ Mann-Whitney U (alternative: misclassified > correct on prediction margin):")
    for scope, d in uq.items():
        p = d["p_value"]
        sig = ("*** (p < 1e-300)" if p < 1e-300 else
               f"*** (p = {p:.2e})" if p < 0.001 else
               f"** (p = {p:.3f})" if p < 0.01 else
               f"* (p = {p:.3f})" if p < 0.05 else
               f"n.s. (p = {p:.3f})")
        print(f"  {scope}:  {sig}  n_correct={d['n_correct']}  n_mis={d['n_misclassified']}")

    # Console pretty-print
    print("\nMessidor-2 within-dataset:")
    print(f"{'#':<3}{'Model':<22}{'QWK':<22}{'Accuracy':<22}{'Macro F1':<22}{'MAE':<10}")
    for i, r in enumerate(within):
        print(f"{i+1:<3}{r['display']:<22}{_fmt(r['qwk_mean'], r['qwk_std']):<22}"
              f"{_fmt(r['acc_mean'], r['acc_std']):<22}"
              f"{_fmt(r['f1_mean'], r['f1_std']):<22}"
              f"{r['mae_mean']:.3f}")

    print("\nAPTOS → Messidor-2 (cross-dataset):")
    print(f"{'#':<3}{'Model':<22}{'QWK':<22}{'Acc':<12}{'F1':<12}{'MAE':<10}{'AUC':<10}")
    for i, r in enumerate(cross):
        print(f"{i+1:<3}{r['display'].replace('APTOS → Messidor-2  ', ''):<22}"
              f"{_fmt(r['qwk_mean'], r['qwk_std']):<22}"
              f"{r['acc_mean']:<12.4f}{r['f1_mean']:<12.4f}{r['mae_mean']:<10.3f}"
              f"{r['binary']['auc']:<10.4f}")

    print(f"\nWrote:\n  {OUT}/messidor2_final_results.csv"
          f"\n  {OUT}/messidor2_final_table.md"
          f"\n  {OUT}/messidor2_binary_results.csv"
          f"\n  {OUT}/messidor2_per_class_breakdown.csv")


if __name__ == "__main__":
    main()
