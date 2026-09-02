"""Clinical screening metrics for referable and vision-threatening DR.

Reviewer comment 7: "The assessment does not include critical clinical
screening metrics like sensitivity and specificity for referable DR at
Grade >= 2, and actionable DR triage metrics, as only QWK and ECE are
reported."

QWK and ECE describe agreement and probability quality, but a screening
deployment is judged on whether it refers the patients who need referral. The
standard operating points on the 5-point ICDR scale are:

    any DR   y >= 1   (mild NPDR or worse)
    rDR      y >= 2   (moderate NPDR or worse)  -- the referral threshold
    vtDR     y >= 3   (severe NPDR or PDR)      -- vision-threatening
    PDR      y >= 4

A convenient property of the K-1 ordinal decomposition is that each of these
is read off directly: head k already emits P(y > k), so P(y > 1) IS the
referable-DR score. No extra thresholding, retraining, or class marginalising
is needed -- something the plain 5-way softmax baselines cannot do natively.
We therefore score each operating point two ways:

  head    : the native head output P(y > k)                (ranking score, AUROC)
  argmax  : the decoded grade, thresholded at > k          (the deployed decision)

Reference bars used for the pass/fail column, both for referable DR:
  NHS diabetic eye screening : sensitivity >= 0.85, specificity >= 0.80
  Abramoff et al. (IDx-DR)   : sensitivity 0.872, specificity 0.907

95% CIs are stratified bootstrap percentile intervals over samples.

Outputs results/clinical_metrics_table.{md,csv} and
results/clinical_metrics.json.

Usage:
    uv run python scripts/compute_clinical_metrics.py [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import chain_rule_probs

EXPERIMENTS = [
    ("RETFound x Ordinal CapsNet",        "APTOS",      ["results/ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",      "APTOS",      ["results/mlp_ordinal/seed*/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",         "APTOS",      ["results/ordinal_capsnet_dinov2/seed*/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",       "APTOS",      ["results/mlp_ordinal_dinov2/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet + LoRA", "APTOS",      ["results/lora_ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet",        "Messidor-2", ["results/messidor2/ordinal_capsnet/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",      "Messidor-2", ["results/messidor2/mlp_k1_sigmoid/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",         "Messidor-2", ["results/messidor2_dinov2/ordinal_capsnet/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",       "Messidor-2", ["results/messidor2_dinov2/mlp_k1_sigmoid/preds_fold*.npz"]),
]

LEVELS = {0: "any DR (>=1)", 1: "rDR (>=2)", 2: "vtDR (>=3)", 3: "PDR (>=4)"}
NHS_SENS, NHS_SPEC = 0.85, 0.80


def pool(globs: list[str]) -> dict | None:
    paths: list[Path] = []
    for g in globs:
        paths.extend(sorted(Path(".").glob(g)))
    if not paths:
        return None
    yts, hls = [], []
    for p in paths:
        d = np.load(p)
        yts.append(d["y_true"])
        hls.append(d["head_lengths"])
    return {
        "y_true": np.concatenate(yts).astype(np.int64),
        "h": np.concatenate(hls, axis=0).astype(np.float64)[:, :, 1],
    }


def _rates(y_bin: np.ndarray, pred_bin: np.ndarray) -> dict:
    tp = int(np.sum(pred_bin & y_bin))
    fp = int(np.sum(pred_bin & ~y_bin))
    fn = int(np.sum(~pred_bin & y_bin))
    tn = int(np.sum(~pred_bin & ~y_bin))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "balanced_accuracy": (sens + spec) / 2 if tp + fn and tn + fp else float("nan")}


def _at_target(score: np.ndarray, y_bin: np.ndarray, target: float,
               mode: str) -> dict:
    """Sweep the score threshold; report the operating point that just meets
    a target sensitivity (mode='sens') or specificity (mode='spec')."""
    order = np.unique(score)
    best = None
    for t in order:
        r = _rates(y_bin, score >= t)
        if mode == "sens" and r["sensitivity"] >= target:
            # highest specificity among points meeting the sensitivity floor
            if best is None or r["specificity"] > best["specificity"]:
                best = r | {"threshold": float(t)}
        elif mode == "spec" and r["specificity"] >= target:
            if best is None or r["sensitivity"] > best["sensitivity"]:
                best = r | {"threshold": float(t)}
    return best or {"threshold": float("nan"), "sensitivity": float("nan"),
                    "specificity": float("nan")}


def _boot_ci(y_bin: np.ndarray, score: np.ndarray, pred_bin: np.ndarray,
             n_boot: int, rng: np.random.Generator) -> dict:
    n = len(y_bin)
    sens, spec, auc = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb, pb = y_bin[idx], score[idx], pred_bin[idx]
        if yb.all() or not yb.any():
            continue
        r = _rates(yb, pb)
        sens.append(r["sensitivity"])
        spec.append(r["specificity"])
        auc.append(roc_auc_score(yb, sb))
    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
    return {"sensitivity_ci": ci(sens), "specificity_ci": ci(spec), "auroc_ci": ci(auc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    rows: list[dict] = []
    for name, dataset, globs in EXPERIMENTS:
        p = pool(globs)
        if p is None:
            print(f"  SKIP  {dataset:<12s} {name}")
            continue
        h, y = p["h"], p["y_true"]
        probs, _ = chain_rule_probs(h)
        grade = probs.argmax(axis=1)
        rng = np.random.default_rng(42)

        rec: dict = {"dataset": dataset, "model": name, "n": int(len(y)),
                     "levels": {}}
        for k, label in LEVELS.items():
            y_bin = y > k
            if not y_bin.any() or y_bin.all():
                continue
            score = h[:, k]                       # native P(y > k)
            at_half = _rates(y_bin, score >= 0.5)
            at_argmax = _rates(y_bin, grade > k)
            entry = {
                "label": label,
                "prevalence": float(y_bin.mean()),
                "auroc_head": float(roc_auc_score(y_bin, score)),
                "head@0.5": at_half,
                "argmax": at_argmax,
                f"at_sens{NHS_SENS:.2f}": _at_target(score, y_bin, NHS_SENS, "sens"),
                f"at_spec{NHS_SPEC:.2f}": _at_target(score, y_bin, NHS_SPEC, "spec"),
            }
            if k == 1:                            # referable DR: add CIs + NHS bar
                entry |= _boot_ci(y_bin, score, score >= 0.5, args.n_boot, rng)
                entry["meets_nhs_at_0.5"] = bool(
                    at_half["sensitivity"] >= NHS_SENS and at_half["specificity"] >= NHS_SPEC
                )
            rec["levels"][str(k)] = entry
        rows.append(rec)

        r = rec["levels"]["1"]
        print(f"  {dataset:<11s} {name:<38s} rDR AUROC {r['auroc_head']:.4f}  "
              f"sens {r['head@0.5']['sensitivity']:.3f}  "
              f"spec {r['head@0.5']['specificity']:.3f}  "
              f"NHS {'PASS' if r['meets_nhs_at_0.5'] else 'FAIL'}")

    out = Path("results")
    (out / "clinical_metrics.json").write_text(json.dumps(rows, indent=2))

    md = [
        "# Clinical screening metrics: referable and vision-threatening DR",
        "",
        "Operating points on the ICDR scale, scored from the native K-1 head",
        "output P(y > k) -- no extra thresholding is required, since head k is",
        "already the binary detector for that level.",
        "",
        f"NHS diabetic eye screening bar for referable DR: sensitivity >= {NHS_SENS:.2f},",
        f"specificity >= {NHS_SPEC:.2f}. Abramoff et al. (IDx-DR) report 0.872 / 0.907.",
        "95% CIs are bootstrap percentile intervals "
        f"({args.n_boot} resamples).",
        "",
        "## Referable DR (grade >= 2) — the deployment-relevant threshold",
        "",
        "| Dataset | Model | Prev. | AUROC (95% CI) | Sens (95% CI) | Spec (95% CI) "
        "| PPV | NPV | Bal. acc | NHS |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["levels"]["1"]
        h5 = e["head@0.5"]
        md.append(
            f"| {r['dataset']} | {r['model']} | {e['prevalence']:.3f} "
            f"| {e['auroc_head']:.4f} ({e['auroc_ci'][0]:.3f}-{e['auroc_ci'][1]:.3f}) "
            f"| {h5['sensitivity']:.3f} ({e['sensitivity_ci'][0]:.3f}-{e['sensitivity_ci'][1]:.3f}) "
            f"| {h5['specificity']:.3f} ({e['specificity_ci'][0]:.3f}-{e['specificity_ci'][1]:.3f}) "
            f"| {h5['ppv']:.3f} | {h5['npv']:.3f} | {h5['balanced_accuracy']:.3f} "
            f"| {'PASS' if e['meets_nhs_at_0.5'] else 'FAIL'} |"
        )

    md += ["", "## All operating points (AUROC / sens / spec at head >= 0.5)", "",
           "| Dataset | Model | any DR | rDR (>=2) | vtDR (>=3) | PDR (>=4) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for k in (0, 1, 2, 3):
            e = r["levels"].get(str(k))
            if e is None:
                cells.append("—")
                continue
            cells.append(f"{e['auroc_head']:.3f} / {e['head@0.5']['sensitivity']:.2f} "
                         f"/ {e['head@0.5']['specificity']:.2f}")
        md.append(f"| {r['dataset']} | {r['model']} | " + " | ".join(cells) + " |")

    md += ["", "## Referable DR at a fixed clinical target", "",
           f"Left: highest specificity achievable at sensitivity >= {NHS_SENS:.2f}. "
           f"Right: highest sensitivity at specificity >= {NHS_SPEC:.2f}.", "",
           "| Dataset | Model | Spec @ sens>=0.85 | Thresh | Sens @ spec>=0.80 | Thresh |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        e = r["levels"]["1"]
        a, b = e[f"at_sens{NHS_SENS:.2f}"], e[f"at_spec{NHS_SPEC:.2f}"]
        md.append(f"| {r['dataset']} | {r['model']} | {a['specificity']:.3f} "
                  f"| {a['threshold']:.3f} | {b['sensitivity']:.3f} | {b['threshold']:.3f} |")

    (out / "clinical_metrics_table.md").write_text("\n".join(md) + "\n")

    hdr = ["dataset", "model", "n", "level", "prevalence", "auroc_head",
           "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy",
           "argmax_sensitivity", "argmax_specificity"]
    csv = [",".join(hdr)]
    for r in rows:
        for k, e in r["levels"].items():
            h5, am = e["head@0.5"], e["argmax"]
            csv.append(",".join([
                r["dataset"], r["model"], str(r["n"]), e["label"],
                f"{e['prevalence']:.6f}", f"{e['auroc_head']:.6f}",
                f"{h5['sensitivity']:.6f}", f"{h5['specificity']:.6f}",
                f"{h5['ppv']:.6f}", f"{h5['npv']:.6f}",
                f"{h5['balanced_accuracy']:.6f}",
                f"{am['sensitivity']:.6f}", f"{am['specificity']:.6f}",
            ]))
    (out / "clinical_metrics_table.csv").write_text("\n".join(csv) + "\n")
    print("\nWrote results/clinical_metrics_table.{md,csv} + "
          "results/clinical_metrics.json")


if __name__ == "__main__":
    main()
