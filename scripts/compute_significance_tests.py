"""Per-fold Wilcoxon signed-rank tests for the key head × backbone claims.

Loads per-fold (seed × fold) QWK values from `summary_multiseed.json` files
(APTOS, 3 seeds x 5 folds = 15 pairs per config) and `summary.json` files
(Messidor-2, 5 folds per config) and runs paired Wilcoxon tests on the folds.

This addresses the Gemini review's request for significance tests on the
capsule-vs-MLP-K1 and DINOv2-vs-RETFound deltas, which are small in absolute
terms and need a p-value to be defensible.

Emits:
  results/significance_tests.md
  results/significance_tests.json

Usage:
  uv run python scripts/compute_significance_tests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from scipy.stats import wilcoxon


def load_per_fold_qwk_multiseed(root: Path) -> list[float]:
    """Returns the pooled per-fold QWK across all seeds (3x5 = 15 values for
    APTOS-style runs). Handles three layout variants: (a) APTOS
    `summary_multiseed.json` -> per_seed[i].result.per_fold_qwk, (b) LoRA
    `seed{S}/summary.json` -> per_fold (list of dicts with "qwk" key), and
    (c) fallback to `pooled_across_seeds.json` if present."""
    # (a) APTOS multiseed layout.
    ms = root / "summary_multiseed.json"
    if ms.exists():
        d = json.loads(ms.read_text())
        qwks: list[float] = []
        for seed_entry in d["per_seed"]:
            qwks.extend(seed_entry["result"].get("per_fold_qwk", []))
        if qwks:
            return qwks
    # (b) LoRA layout: per-seed subdirs with summary.json containing per_fold.
    seed_dirs = sorted(root.glob("seed*"))
    if seed_dirs:
        qwks = []
        for sd in seed_dirs:
            sp = sd / "summary.json"
            if not sp.exists():
                continue
            dd = json.loads(sp.read_text())
            for fold_entry in dd.get("per_fold", []):
                q = fold_entry.get("qwk") or fold_entry.get("val_qwk")
                if q is not None:
                    qwks.append(float(q))
        if qwks:
            return qwks
    return []


def load_per_fold_qwk_flat(root: Path) -> list[float]:
    """For Messidor-2 / LoRA: keys differ (`per_fold_qwk` vs `per_fold_val_qwk`
    vs per-seed `per_fold`). Try every known location in priority order."""
    path = root / "summary.json"
    path_multi = root / "summary_multiseed.json"
    if path_multi.exists():
        d = json.loads(path_multi.read_text())
        # LoRA multi-seed: per_seed[i].result.per_fold_qwk
        qwks: list[float] = []
        for s in d.get("per_seed", []):
            qwks.extend(s["result"].get("per_fold_qwk", []))
        if qwks:
            return qwks
    if path.exists():
        d = json.loads(path.read_text())
        r = d.get("result", d)
        for key in ("per_fold_qwk", "per_fold_val_qwk"):
            v = r.get(key)
            if v:
                return list(v)
    return []


def paired(a: list[float], b: list[float], name_a: str, name_b: str) -> dict:
    """Wilcoxon signed-rank on paired (a_i, b_i). Returns p-value + effect."""
    if len(a) != len(b) or len(a) == 0:
        return {"error": f"unequal lengths: |{name_a}|={len(a)} |{name_b}|={len(b)}"}
    # Wilcoxon uses the two-sided alternative by default; tests H0: median(a-b)=0.
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    delta = np.array(a) - np.array(b)
    return {
        "a": name_a, "b": name_b,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "wilcoxon_stat": float(stat),
        "p_value_two_sided": float(p),
        "n_pairs": int(len(a)),
        "a_wins_n_pairs": int((delta > 0).sum()),
    }


def main() -> None:
    out_root = Path("results")
    tests: list[dict] = []

    # -------- APTOS (3 seeds x 5-fold, 15 paired values per config) --------
    r = Path("results")
    aptos_caps = load_per_fold_qwk_multiseed(r / "ordinal_capsnet")
    aptos_mlp = load_per_fold_qwk_multiseed(r / "mlp_ordinal")
    aptos_caps_dv = load_per_fold_qwk_multiseed(r / "ordinal_capsnet_dinov2")
    aptos_mlp_dv = load_per_fold_qwk_multiseed(r / "mlp_ordinal_dinov2")
    aptos_lora = load_per_fold_qwk_multiseed(r / "lora_ordinal_capsnet")
    aptos_mlp_ce = None
    # Day-1 baselines saved per-seed in results/baselines; their schema differs.
    # We skip the MLP+CE vs MLP-K1 test here since they use different runners.

    tests.append({"family": "APTOS  head effect (RETFound)",
                  **paired(aptos_caps, aptos_mlp,
                           "RETFound x Ordinal CapsNet",
                           "RETFound x MLP + K-1 sigmoid")})

    tests.append({"family": "APTOS  head effect (DINOv2)",
                  **paired(aptos_caps_dv, aptos_mlp_dv,
                           "DINOv2 x Ordinal CapsNet",
                           "DINOv2 x MLP + K-1 sigmoid")})

    tests.append({"family": "APTOS  backbone effect (Ordinal CapsNet head)",
                  **paired(aptos_caps_dv, aptos_caps,
                           "DINOv2 x Ordinal CapsNet",
                           "RETFound x Ordinal CapsNet")})

    tests.append({"family": "APTOS  backbone effect (MLP-K1 head)",
                  **paired(aptos_mlp_dv, aptos_mlp,
                           "DINOv2 x MLP + K-1 sigmoid",
                           "RETFound x MLP + K-1 sigmoid")})

    tests.append({"family": "APTOS  LoRA effect (RETFound + CapsNet)",
                  **paired(aptos_lora, aptos_caps,
                           "RETFound x Ordinal CapsNet + LoRA",
                           "RETFound x Ordinal CapsNet")})

    # -------- Messidor-2 within-dataset (5 folds) --------
    mess_caps = load_per_fold_qwk_flat(r / "messidor2" / "ordinal_capsnet")
    mess_mlp = load_per_fold_qwk_flat(r / "messidor2" / "mlp_k1_sigmoid")
    mess_caps_dv = load_per_fold_qwk_flat(r / "messidor2_dinov2" / "ordinal_capsnet")
    mess_mlp_dv = load_per_fold_qwk_flat(r / "messidor2_dinov2" / "mlp_k1_sigmoid")

    tests.append({"family": "Messidor-2  head effect (RETFound)",
                  **paired(mess_caps, mess_mlp,
                           "RETFound x Ordinal CapsNet",
                           "RETFound x MLP + K-1 sigmoid")})

    tests.append({"family": "Messidor-2  head effect (DINOv2)",
                  **paired(mess_caps_dv, mess_mlp_dv,
                           "DINOv2 x Ordinal CapsNet",
                           "DINOv2 x MLP + K-1 sigmoid")})

    tests.append({"family": "Messidor-2  backbone effect (Ordinal CapsNet head)",
                  **paired(mess_caps_dv, mess_caps,
                           "DINOv2 x Ordinal CapsNet",
                           "RETFound x Ordinal CapsNet")})

    tests.append({"family": "Messidor-2  backbone effect (MLP-K1 head)",
                  **paired(mess_mlp_dv, mess_mlp,
                           "DINOv2 x MLP + K-1 sigmoid",
                           "RETFound x MLP + K-1 sigmoid")})

    # -------- Emit --------
    md: list[str] = []
    md.append("# Paired Wilcoxon signed-rank tests")
    md.append("")
    md.append("Each row is a paired test on per-fold (seed x fold) QWK values. "
              "APTOS pairs pool 3 seeds x 5 folds = 15 paired comparisons; "
              "Messidor-2 pairs use 5 folds. Data splits are fixed across seeds "
              "via `cfg.data.seed`, so folds are paired between configs and the "
              "Wilcoxon assumption of exchangeable pairs under H0 is met.")
    md.append("")
    md.append("| Family | A | B | n | mean A | mean B | mean $\\Delta$ | A wins | p (two-sided) |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for t in tests:
        if "error" in t:
            continue
        md.append(f"| {t['family']} | {t['a']} | {t['b']} | "
                  f"{t['n_pairs']} | "
                  f"{t['mean_a']:.4f} | {t['mean_b']:.4f} | "
                  f"{t['mean_delta']:+.4f} | "
                  f"{t['a_wins_n_pairs']}/{t['n_pairs']} | "
                  f"{t['p_value_two_sided']:.4f} |")
    md.append("")
    md.append("Interpretation: p < 0.05 indicates the observed delta is unlikely "
              "under H0 ``both configs have the same median per-fold QWK''. "
              "A wins counts folds where QWK(A) > QWK(B).")

    (out_root / "significance_tests.md").write_text("\n".join(md) + "\n")
    (out_root / "significance_tests.json").write_text(json.dumps(tests, indent=2))

    print("\n".join(md[4:]))
    print(f"\nWrote {out_root/'significance_tests.md'}")


if __name__ == "__main__":
    main()
