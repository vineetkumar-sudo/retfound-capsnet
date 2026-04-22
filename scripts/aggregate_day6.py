"""Day 6 — Consolidate all APTOS experiments into one paper-ready ablation table.

Reads existing `results/**/summary.json` files plus any per-fold prediction npz
files, pools across seeds when multi-seed runs exist, recomputes metrics
(including MAE, which older runs did not report) through `compute_all_metrics`,
and writes:

  results/aptos_final_table.csv           -- machine-readable
  results/aptos_final_table.md            -- paper-ready, best-in-column bolded
  results/per_class_breakdown.csv         -- Vanilla vs Ordinal per-grade delta

Usage:
    uv run python scripts/aggregate_day6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from src.evaluate import compute_all_metrics


ROW_SPECS = [
    # (display_name, canonical_id, kind, path_or_key, regime_label)
    ("Linear Probe + CE",      "linear_probe_ce",     "baseline_pred", ("Linear Probe + CE",),      "3 seeds x 5-fold"),
    ("MLP + CE",               "mlp_ce",              "baseline_pred", ("MLP + CE",),               "3 seeds x 5-fold"),
    ("MLP + MSE (Ordinal)",    "mlp_mse",             "baseline_pred", ("MLP + MSE (Ordinal)",),    "3 seeds x 5-fold"),
    ("MLP + Weighted CE",      "mlp_weighted_ce",     "baseline_pred", ("MLP + Weighted CE",),      "3 seeds x 5-fold"),
    ("Vanilla CapsNet",        "capsnet_vanilla",     "capsnet",       ("results/capsnet",),        "3 seeds x 5-fold"),
    ("Ordinal CapsNet",        "ordinal_capsnet",     "ordinal",       ("results/ordinal_capsnet",),"3 seeds x 5-fold"),
    ("+ Asymmetric loss",      "ordinal_asymmetric",  "ordinal",       ("results/asymmetric_ordinal/A_baseline",), "3 seeds x 5-fold"),
    ("+ KC Loss (gamma=0.3)",  "ordinal_kc_loss",     "ordinal",       ("results/ordinal_kc/kc_gamma_0p30",),      "3 seeds x 5-fold"),
    ("Ordinal CapsNet + LoRA", "ordinal_lora",        "lora",          ("results/lora_ordinal_capsnet",),          "3 seeds x 5-fold, LoRA r=8"),
    ("+ Non-uniform squash",   "ordinal_nonuniform_squash", "ordinal",  ("results/ordinal_capsnet_nonuniform",),    "3 seeds x 5-fold"),
]

BASELINES_DIR = Path("results/baselines")
BASELINES_JSON = BASELINES_DIR / "summary.json"


def _safe_name(name: str) -> str:
    """Mirror run_baselines.py's safe-name convention."""
    return name.lower().replace(" ", "_").replace("+", "_").replace("__", "_")


def pool_preds(npz_paths: list[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate y_true/y_pred across fold npz files. Returns None if empty."""
    if not npz_paths:
        return None
    yts, yps = [], []
    for p in npz_paths:
        d = np.load(p)
        yts.append(d["y_true"])
        yps.append(d["y_pred"])
    return np.concatenate(yts), np.concatenate(yps)


def per_fold_metrics(npz_paths: list[Path]) -> dict:
    """Compute {qwk, accuracy, macro_f1, mae} per fold, then mean/std across folds."""
    qwks, accs, f1s, maes = [], [], [], []
    for p in npz_paths:
        d = np.load(p)
        m = compute_all_metrics(d["y_true"], d["y_pred"])
        qwks.append(m["qwk"]); accs.append(m["accuracy"])
        f1s.append(m["macro_f1"]); maes.append(m["mae"])
    def _pair(vs):
        return (float(np.mean(vs)), float(np.std(vs)))
    return {
        "qwk": _pair(qwks), "accuracy": _pair(accs),
        "macro_f1": _pair(f1s), "mae": _pair(maes),
    }


def row_from_baseline(display: str, baseline_key: str) -> dict:
    """Day 1 baseline row. Prefers per-fold preds; pools across `seed{S}/` subdirs
    when a multi-seed run has been executed, else falls back to the legacy flat
    layout at `results/baselines/`.

    Multi-seed: mean across seeds of each seed's fold-mean; std across seeds.
    Single-seed: fold-mean / fold-std from that seed's preds.
    """
    seed_dirs = sorted(BASELINES_DIR.glob("seed*"))
    safe = _safe_name(baseline_key)

    if seed_dirs:
        # Multi-seed layout: aggregate each seed's fold metrics, then pool across.
        qwks, accs, f1s, maes, mae_sources = [], [], [], [], []
        for sd in seed_dirs:
            sj = sd / "summary.json"
            if not sj.exists():
                print(f"  [row_from_baseline] WARN {sj} missing — seed skipped")
                continue
            s = json.load(sj.open())
            res = s["results"].get(baseline_key)
            if res is None:
                print(f"  [row_from_baseline] WARN {baseline_key} absent from {sj} — skipped")
                continue
            npz = sorted(sd.glob(f"preds_{safe}_fold*.npz"))
            if npz:
                per = per_fold_metrics(npz)
                qwks.append(per["qwk"][0])
                accs.append(per["accuracy"][0])
                f1s.append(per["macro_f1"][0])
                maes.append(per["mae"][0])
                mae_sources.append("preds")
            else:
                qwks.append(res["qwk_mean"])
                accs.append(res["accuracy_mean"])
                f1s.append(res["macro_f1_mean"])
                if "mae_mean" in res:
                    maes.append(res["mae_mean"])
                    mae_sources.append("stored_mean")
                else:
                    mae_sources.append("missing")
        return {
            "Model": display,
            "QWK_mean": float(np.mean(qwks)), "QWK_std": float(np.std(qwks)),
            "Accuracy_mean": float(np.mean(accs)), "Accuracy_std": float(np.std(accs)),
            "MacroF1_mean": float(np.mean(f1s)), "MacroF1_std": float(np.std(f1s)),
            "MAE_mean": float(np.mean(maes)) if maes else float("nan"),
            "MAE_std": float(np.std(maes)) if len(maes) > 1 else 0.0,
            "_n_seeds": len(qwks),
            "_mae_sources": mae_sources,
        }

    # Legacy single-seed flat layout.
    summary = json.load(BASELINES_JSON.open())
    res = summary["results"][baseline_key]
    npz = sorted(BASELINES_DIR.glob(f"preds_{safe}_fold*.npz"))
    if npz:
        per = per_fold_metrics(npz)
        mae_m, mae_s = per["mae"]
        mae_src = "preds"
    elif "mae_mean" in res:
        mae_m, mae_s = res["mae_mean"], res.get("mae_std", 0.0)
        mae_src = "stored_mean"
    else:
        mae_m, mae_s = (float("nan"), float("nan"))
        mae_src = "missing"

    return {
        "Model": display,
        "QWK_mean": res["qwk_mean"], "QWK_std": res["qwk_std"],
        "Accuracy_mean": res["accuracy_mean"], "Accuracy_std": res["accuracy_std"],
        "MacroF1_mean": res["macro_f1_mean"], "MacroF1_std": res["macro_f1_std"],
        "MAE_mean": mae_m, "MAE_std": mae_s,
        "_mae_sources": [mae_src],
    }


def row_from_lora(display: str, results_dir: Path) -> dict:
    """LoRA fine-tuned row. Pools across `seed{S}/` subdirs (multi-seed layout);
    falls back to the flat `preds_fold*.npz` at `results_dir/` for legacy
    single-seed runs.

    For multi-seed: mean across seeds of each seed's fold-mean; std across seeds.
    For single-seed: fold-mean / fold-std from that seed's preds.
    """
    seed_dirs = sorted(results_dir.glob("seed*"))
    if seed_dirs:
        qwks, accs, f1s, maes = [], [], [], []
        for sd in seed_dirs:
            npz = sorted(sd.glob("preds_fold*.npz"))
            if not npz:
                print(f"  [row_from_lora] WARN no preds in {sd} — skipped")
                continue
            per = per_fold_metrics(npz)
            qwks.append(per["qwk"][0])
            accs.append(per["accuracy"][0])
            f1s.append(per["macro_f1"][0])
            maes.append(per["mae"][0])
        return {
            "Model": display,
            "QWK_mean": float(np.mean(qwks)), "QWK_std": float(np.std(qwks)),
            "Accuracy_mean": float(np.mean(accs)), "Accuracy_std": float(np.std(accs)),
            "MacroF1_mean": float(np.mean(f1s)), "MacroF1_std": float(np.std(f1s)),
            "MAE_mean": float(np.mean(maes)) if maes else float("nan"),
            "MAE_std": float(np.std(maes)) if len(maes) > 1 else 0.0,
            "_n_seeds": len(qwks),
            "_mae_sources": ["preds"] * len(qwks),
        }

    # Legacy single-seed flat layout.
    npz = sorted(results_dir.glob("preds_fold*.npz"))
    assert npz, f"No preds_fold*.npz in {results_dir}"
    per = per_fold_metrics(npz)
    qwk_mean, qwk_std = per["qwk"]
    acc_mean, acc_std = per["accuracy"]
    f1_mean, f1_std = per["macro_f1"]
    mae_mean, mae_std = per["mae"]
    return {
        "Model": display,
        "QWK_mean": qwk_mean, "QWK_std": qwk_std,
        "Accuracy_mean": acc_mean, "Accuracy_std": acc_std,
        "MacroF1_mean": f1_mean, "MacroF1_std": f1_std,
        "MAE_mean": mae_mean, "MAE_std": mae_std,
        "_mae_sources": ["preds"],
    }


def row_from_capsnet(display: str, results_dir: Path) -> dict:
    """Day 2 vanilla CapsNet. Pools across `seed{S}/` subdirs when present
    (multi-seed invocation via `--model-seed` + `--output-dir`), else reads
    the legacy flat single-seed layout at `results_dir/`.
    """
    seed_dirs = sorted(results_dir.glob("seed*"))
    if seed_dirs:
        qwks, accs, f1s, maes = [], [], [], []
        for sd in seed_dirs:
            sj = sd / "summary.json"
            if not sj.exists():
                print(f"  [row_from_capsnet] WARN {sj} missing — seed skipped")
                continue
            npz = sorted(sd.glob("preds_fold*.npz"))
            if not npz:
                s = json.load(sj.open())
                res = s.get("result", s)
                qwks.append(res["qwk_mean"])
                accs.append(res["accuracy_mean"])
                f1s.append(res["macro_f1_mean"])
                maes.append(res.get("mae_mean", float("nan")))
                continue
            per = per_fold_metrics(npz)
            qwks.append(per["qwk"][0])
            accs.append(per["accuracy"][0])
            f1s.append(per["macro_f1"][0])
            maes.append(per["mae"][0])
        return {
            "Model": display,
            "QWK_mean": float(np.mean(qwks)), "QWK_std": float(np.std(qwks)),
            "Accuracy_mean": float(np.mean(accs)), "Accuracy_std": float(np.std(accs)),
            "MacroF1_mean": float(np.mean(f1s)), "MacroF1_std": float(np.std(f1s)),
            "MAE_mean": float(np.mean(maes)) if maes else float("nan"),
            "MAE_std": float(np.std(maes)) if len(maes) > 1 else 0.0,
            "_n_seeds": len(qwks),
            "_mae_sources": ["preds"] * len(qwks),
        }

    s = json.load((results_dir / "summary.json").open())
    res = s.get("result", s)
    npz = sorted(results_dir.glob("preds_fold*.npz"))
    if npz:
        per = per_fold_metrics(npz)
        mae_m, mae_s = per["mae"]
        mae_src = "preds"
    else:
        mae_m, mae_s = (float("nan"), float("nan"))
        mae_src = "missing"
    return {
        "Model": display,
        "QWK_mean": res["qwk_mean"], "QWK_std": res["qwk_std"],
        "Accuracy_mean": res["accuracy_mean"], "Accuracy_std": res["accuracy_std"],
        "MacroF1_mean": res["macro_f1_mean"], "MacroF1_std": res["macro_f1_std"],
        "MAE_mean": mae_m, "MAE_std": mae_s,
        "_mae_sources": [mae_src],
    }


def row_from_ordinal(display: str, variant_dir: Path) -> dict:
    """Rows 6-8. Pool across seed{S}/ subdirs; legacy top-level summary is only
    used when NO seed{S}/ subdirs exist (back-compat for single-seed runs).

    Deduplication rule: if a `seed{S}/` subdir exists, the top-level
    summary.json at `variant_dir` is skipped, since it would double-count that
    same seed (e.g. ordinal/ vs ordinal/seed42/ both come from data_seed=42).

    For multi-seed: report mean across seeds of per-seed fold-means, std across
    seeds. Regime labelled as "N seeds × 5-fold" where N == len(seed_dirs).
    """
    seed_dirs = sorted(variant_dir.glob("seed*"))
    if seed_dirs:
        seed_dirs_all = list(seed_dirs)  # multi-seed layout: legacy top-level is a duplicate, skip
    else:
        seed_dirs_all = [variant_dir]    # single-seed legacy layout

    qwks, accs, f1s, maes, mae_sources = [], [], [], [], []
    single_seed_per: dict | None = None  # per-fold metrics for single-seed fallback
    for sd in seed_dirs_all:
        sj = sd / "summary.json"
        if not sj.exists():
            print(f"  [row_from_ordinal] WARN {sd}/summary.json missing — seed skipped")
            continue
        s = json.load(sj.open())
        res = s.get("result", s)
        npz = sorted(sd.glob("preds_fold*.npz"))
        if npz:
            per = per_fold_metrics(npz)
            qwks.append(per["qwk"][0])
            accs.append(per["accuracy"][0])
            f1s.append(per["macro_f1"][0])
            maes.append(per["mae"][0])
            mae_sources.append("preds")
            single_seed_per = per
        else:
            qwks.append(res["qwk_mean"])
            accs.append(res["accuracy_mean"])
            f1s.append(res["macro_f1_mean"])
            if "mae_mean" in res:
                maes.append(res["mae_mean"])
                mae_sources.append("stored_mean")
            else:
                mae_sources.append("missing")

    # Single-seed rows: prefer the across-fold std (real variation) over the
    # degenerate across-seed std (always 0 with one seed).
    if len(seed_dirs_all) == 1 and single_seed_per is not None:
        qwk_m, qwk_s = single_seed_per["qwk"]
        acc_m, acc_s = single_seed_per["accuracy"]
        f1_m, f1_s = single_seed_per["macro_f1"]
        mae_m, mae_s = single_seed_per["mae"]
    else:
        qwk_m, qwk_s = float(np.mean(qwks)), float(np.std(qwks))
        acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
        f1_m, f1_s = float(np.mean(f1s)), float(np.std(f1s))
        mae_m = float(np.mean(maes)) if maes else float("nan")
        mae_s = float(np.std(maes)) if len(maes) > 1 else 0.0

    return {
        "Model": display,
        "QWK_mean": qwk_m, "QWK_std": qwk_s,
        "Accuracy_mean": acc_m, "Accuracy_std": acc_s,
        "MacroF1_mean": f1_m, "MacroF1_std": f1_s,
        "MAE_mean": mae_m, "MAE_std": mae_s,
        "_n_seeds": len(seed_dirs_all),
        "_mae_sources": mae_sources,
    }


def build_rows() -> list[dict]:
    rows = []
    for display, canonical_id, kind, args, regime in ROW_SPECS:
        if kind == "baseline_pred":
            row = row_from_baseline(display, args[0])
        elif kind == "capsnet":
            row = row_from_capsnet(display, Path(args[0]))
        elif kind == "ordinal":
            row = row_from_ordinal(display, Path(args[0]))
        elif kind == "lora":
            row = row_from_lora(display, Path(args[0]))
        else:
            raise ValueError(kind)
        row["CanonicalID"] = canonical_id
        row["Regime"] = regime
        rows.append(row)
    return rows


def format_pm(mean: float, std: float, width: int = 4) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.{width}f} ± {std:.{width}f}"


def write_csv(rows: list[dict], path: Path) -> None:
    """Display-friendly CSV kept for backwards compatibility with Day 6 tooling."""
    import csv
    cols = ["Model", "QWK_mean", "QWK_std", "Accuracy_mean", "Accuracy_std",
            "MacroF1_mean", "MacroF1_std", "MAE_mean", "MAE_std", "Regime"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_canonical_csv(rows: list[dict], path: Path) -> None:
    """Day 7 source-of-truth CSV: snake_case model ids + the exact column schema the paper consumes."""
    import csv
    cols = ["model", "qwk_mean", "qwk_std", "acc_mean", "acc_std",
            "f1_mean", "f1_std", "mae_mean", "mae_std", "regime"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "model": r["CanonicalID"],
                "qwk_mean": r["QWK_mean"], "qwk_std": r["QWK_std"],
                "acc_mean": r["Accuracy_mean"], "acc_std": r["Accuracy_std"],
                "f1_mean": r["MacroF1_mean"], "f1_std": r["MacroF1_std"],
                "mae_mean": r["MAE_mean"], "mae_std": r["MAE_std"],
                "regime": r["Regime"],
            })


def write_md(rows: list[dict], path: Path) -> None:
    """Paper-ready markdown table with best-in-column bolded.

    QWK / Accuracy / Macro F1: higher is better.
    MAE: lower is better.
    """
    def _best_idx(vals, mode):
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if not valid:
            return -1
        if mode == "max":
            return max(valid, key=lambda t: t[1])[0]
        return min(valid, key=lambda t: t[1])[0]

    best_qwk = _best_idx([r["QWK_mean"] for r in rows], "max")
    best_acc = _best_idx([r["Accuracy_mean"] for r in rows], "max")
    best_f1  = _best_idx([r["MacroF1_mean"] for r in rows], "max")
    best_mae = _best_idx([r["MAE_mean"] for r in rows], "min")

    lines = [
        "# APTOS 2019 — Ablation Table",
        "",
        "| # | Model | QWK | Accuracy | Macro F1 | MAE | Regime |",
        "|---|-------|-----|----------|----------|-----|--------|",
    ]
    for i, r in enumerate(rows):
        cells = [
            format_pm(r["QWK_mean"], r["QWK_std"]),
            format_pm(r["Accuracy_mean"], r["Accuracy_std"]),
            format_pm(r["MacroF1_mean"], r["MacroF1_std"]),
            format_pm(r["MAE_mean"], r["MAE_std"], width=3),
        ]
        for col, best in enumerate([best_qwk, best_acc, best_f1, best_mae]):
            if i == best:
                cells[col] = f"**{cells[col]}**"
        lines.append(f"| {i+1} | {r['Model']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {r['Regime']} |")
    lines.append("")
    lines.append("Best value per column in **bold**. QWK / Accuracy / Macro F1: higher is better; MAE: lower is better.")
    path.write_text("\n".join(lines))


def per_class_breakdown(out_path: Path) -> None:
    """Grade 0..4 accuracy: Vanilla vs Ordinal vs LoRA Ordinal.

    Vanilla and Ordinal rows pool across every seed{S}/preds_fold*.npz under the
    respective result dirs (multi-seed layout); a flat `preds_fold*.npz` directly
    at the top level is used as a single-seed fallback for backwards compat.

    Delta is computed between Vanilla and LoRA (the largest-step comparison the
    paper highlights). Ordinal (frozen) is kept as an intermediate reference
    column so readers can see both the architecture lift (Vanilla -> Ordinal)
    and the backbone-tuning lift (Ordinal -> Ordinal+LoRA).
    """
    def _pool_all_seeds(base: Path) -> list[Path]:
        seed_dirs = sorted(base.glob("seed*"))
        if seed_dirs:
            npz: list[Path] = []
            for sd in seed_dirs:
                npz.extend(sorted(sd.glob("preds_fold*.npz")))
            return npz
        return sorted(base.glob("preds_fold*.npz"))

    vanilla_npz = _pool_all_seeds(Path("results/capsnet"))
    ordinal_npz = _pool_all_seeds(Path("results/ordinal_capsnet"))
    lora_npz = _pool_all_seeds(Path("results/lora_ordinal_capsnet"))
    if not vanilla_npz or not ordinal_npz:
        print("[per_class_breakdown] Skipping — preds not on disk yet")
        return

    v = pool_preds(vanilla_npz); o = pool_preds(ordinal_npz)
    assert v is not None and o is not None
    vy, vp = v; oy, op = o

    lora_pool = pool_preds(lora_npz) if lora_npz else None
    ly, lp = lora_pool if lora_pool is not None else (None, None)

    import csv
    class_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    rows = []
    for g in range(5):
        mask_v = vy == g
        mask_o = oy == g
        acc_v = float((vp[mask_v] == g).mean()) * 100 if mask_v.sum() else float("nan")
        acc_o = float((op[mask_o] == g).mean()) * 100 if mask_o.sum() else float("nan")
        if ly is not None and lp is not None:
            mask_l = ly == g
            acc_l = float((lp[mask_l] == g).mean()) * 100 if mask_l.sum() else float("nan")
        else:
            acc_l = float("nan")
        # Delta is Vanilla -> LoRA (the paper's biggest comparison)
        delta = acc_l - acc_v if not np.isnan(acc_l) else acc_o - acc_v
        reference = "LoRA" if not np.isnan(acc_l) else "Ordinal"
        if delta > 5:   note = "Major improvement"
        elif delta > 1: note = "Improvement"
        elif delta > -1: note = "Stable"
        elif delta > -5: note = "Minor drop"
        else:            note = "Regression"
        rows.append({
            "Grade": f"{g} ({class_names[g]})",
            "Vanilla_%": f"{acc_v:.1f}",
            "Ordinal_%": f"{acc_o:.1f}",
            "LoRA_%": f"{acc_l:.1f}" if not np.isnan(acc_l) else "—",
            "Delta_pp": f"{delta:+.1f}  (Vanilla→{reference})",
            "Note": note,
        })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Grade", "Vanilla_%", "Ordinal_%", "LoRA_%", "Delta_pp", "Note"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nPer-class breakdown -> {out_path}")
    for r in rows:
        lora_disp = f"  LoRA={r['LoRA_%']}%" if r["LoRA_%"] != "—" else ""
        print(f"  {r['Grade']:<14s}  Vanilla={r['Vanilla_%']}%  "
              f"Ordinal={r['Ordinal_%']}%{lora_disp}  "
              f"delta={r['Delta_pp']}  ({r['Note']})")


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows = build_rows()

    csv_path = out_dir / "aptos_final_table.csv"
    md_path = out_dir / "aptos_final_table.md"
    pc_path = out_dir / "per_class_breakdown.csv"
    canonical_path = out_dir / "aptos_final_results.csv"

    write_csv(rows, csv_path)
    write_canonical_csv(rows, canonical_path)
    write_md(rows, md_path)
    per_class_breakdown(pc_path)

    # --- MAE source audit (Fix #2) ---
    print("\nMAE source per row:")
    any_stale = False
    for r in rows:
        srcs = r.get("_mae_sources", [])
        counts: dict[str, int] = {}
        for s in srcs:
            counts[s] = counts.get(s, 0) + 1
        desc = ", ".join(f"{v}x {k}" for k, v in counts.items())
        print(f"  {r['Model']:<28s}  {desc}")
        if any(s != "preds" for s in srcs):
            any_stale = True
    if any_stale:
        print("  WARNING: at least one row used stored_mean / missing MAE "
              "instead of fresh preds — regenerate preds for that row.")

    # Pretty-print to console
    print("\nAPTOS ablation table:")
    print(f"{'#':<3}{'Model':<28}{'QWK':<20}{'Acc':<20}{'F1':<20}{'MAE':<18}{'Regime':<22}")
    for i, r in enumerate(rows):
        print(f"{i+1:<3}{r['Model']:<28}"
              f"{format_pm(r['QWK_mean'], r['QWK_std']):<20}"
              f"{format_pm(r['Accuracy_mean'], r['Accuracy_std']):<20}"
              f"{format_pm(r['MacroF1_mean'], r['MacroF1_std']):<20}"
              f"{format_pm(r['MAE_mean'], r['MAE_std'], width=3):<18}"
              f"{r['Regime']:<22}")
    print(f"\nWrote:\n  {csv_path}\n  {canonical_path}\n  {md_path}\n  {pc_path}")


if __name__ == "__main__":
    main()
