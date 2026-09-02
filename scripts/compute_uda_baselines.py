"""Unsupervised domain adaptation baselines for APTOS -> Messidor-2 (R1-10).

Reviewer comment 10: the cross-dataset transfer collapses (QWK ~ 0.01), and
although the paper's SLD prior-shift correction recovers part of it
(QWK ~ 0.22) that is still not clinically acceptable, "thus indicating the
necessity of testing existing UDA techniques."

This evaluates the standard label-free adaptations that are applicable when
the backbone is frozen and features are cached, factorised into the two
distinct shifts they address:

  Feature-space alignment (covariate shift)
    none      target features used as-is
    z-score   per-domain standardisation (mean/variance matching)
    CORAL     Sun & Saenko correlation alignment: whiten the target
              covariance and re-colour it with the source covariance, which
              matches second-order statistics, not just first-order

  Label-space correction (prior / label shift)
    none      argmax as-is
    BBSE      Lipton et al. black-box shift estimation, inverting the source
              confusion matrix to estimate the target prior
    EM/SLD    Saerens et al. expectation-maximisation prior correction
              (what the paper already reports)
    oracle    the true target prior -- an upper bound, uses target labels

All adaptations are unsupervised except `oracle`. The head is trained once
per fold on APTOS and never sees a Messidor-2 label; feature alignment maps
the target into the source space, so the same trained head applies unchanged.

Outputs results/uda_table.{md,csv} and results/uda_metrics.json.

Usage:
    uv run python scripts/compute_uda_baselines.py [--folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import compute_all_metrics
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.ordinal_capsnet import OrdinalCapsNet

APTOS_F = Path("data/aptos/features/train_features.npy")
APTOS_Y = Path("data/aptos/features/train_labels.npy")
MESS_F = Path("data/messidor2/features/features.npy")
MESS_Y = Path("data/messidor2/features/grades.npy")

SEED, K, HOLDOUT = 42, 5, 0.10
EPOCHS, BATCH, LR, WD = 60, 64, 1e-3, 1e-4


# --------------------------------------------------------------------------
# feature-space alignment
# --------------------------------------------------------------------------
def align_none(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    return Xt


def align_zscore(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Map target to zero-mean/unit-variance, then to the source moments."""
    mt, st = Xt.mean(0), Xt.std(0) + 1e-8
    ms, ss = Xs.mean(0), Xs.std(0) + 1e-8
    return (Xt - mt) / st * ss + ms


def align_coral(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """CORAL (Sun & Saenko 2016): whiten target covariance, recolour with source.

        Xt' = (Xt - mu_t) Cov_t^{-1/2} Cov_s^{1/2} + mu_s

    Matrix square roots via symmetric eigendecomposition with a ridge, which
    is numerically safer than Cholesky when the 1024-d covariance is
    ill-conditioned (it is: N_target is only 1744).
    """
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov((Xs - ms).T) + eps * np.eye(Xs.shape[1])
    Ct = np.cov((Xt - mt).T) + eps * np.eye(Xt.shape[1])

    def _pow(C, p):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return (V * w**p) @ V.T

    return (Xt - mt) @ _pow(Ct, -0.5) @ _pow(Cs, 0.5) + ms


ALIGNERS = {"none": align_none, "z-score": align_zscore, "CORAL": align_coral}


# --------------------------------------------------------------------------
# label-space correction
# --------------------------------------------------------------------------
def em_prior(probs: np.ndarray, src_prior: np.ndarray, iters: int = 100) -> np.ndarray:
    """Saerens EM: iteratively re-estimate the target prior."""
    p = src_prior.copy()
    for _ in range(iters):
        w = probs * (p / np.maximum(src_prior, 1e-12))
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        new = w.mean(0)
        if np.abs(new - p).max() < 1e-8:
            break
        p = new
    return p


def bbse_prior(probs_t: np.ndarray, conf_src: np.ndarray,
               src_prior: np.ndarray) -> np.ndarray:
    """BBSE (Lipton 2018): solve C_src^T q = mean predicted target distribution."""
    q_hat = probs_t.mean(0)
    try:
        w = np.linalg.solve(conf_src.T + 1e-6 * np.eye(len(q_hat)), q_hat)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(conf_src.T, q_hat, rcond=None)[0]
    w = np.clip(w, 0, None)
    return w / max(w.sum(), 1e-12)


def rank_calibrate(score: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Assign grades by sorting on a severity score and filling buckets sized
    by `prior` (the paper's existing post-hoc correction, reproduced here so
    the UDA comparison is like-for-like).

    This differs fundamentally from posterior reweighting: it acts on the
    RANKING and forcibly reassigns labels, so it still works when the decoded
    posterior has saturated onto one class -- exactly the regime this transfer
    collapses into.
    """
    n = len(score)
    order = np.argsort(score, kind="stable")
    cumulative = np.cumsum(prior) * n
    b = np.concatenate(([0], np.round(cumulative).astype(int)))
    b[-1] = n
    out = np.zeros(n, dtype=np.int64)
    for k in range(len(prior)):
        out[order[b[k]:b[k + 1]]] = k
    return out


def reweight(probs: np.ndarray, src_prior: np.ndarray,
             tgt_prior: np.ndarray) -> np.ndarray:
    w = probs * (tgt_prior / np.maximum(src_prior, 1e-12))
    return w / np.maximum(w.sum(1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
def chain_probs(h: np.ndarray) -> np.ndarray:
    """Monotone-projected cumulative decode (see src/calibration.py).

    `head_lengths` is (N, K-1, 2); index 1 is the capsule length for the
    positive branch, i.e. P(y > k).
    """
    from src.calibration import chain_rule_probs
    if h.ndim == 3:
        h = h[:, :, 1]
    p, _ = chain_rule_probs(h, decode="difference", project=True)
    return p


def train_fold(Xtr, ytr, device) -> OrdinalCapsNet:
    m = OrdinalCapsNet().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    lf = OrdinalMarginLoss(num_classes=5)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    n = len(yt)
    for _ in range(EPOCHS):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lf(m(Xt[idx])["head_lengths"], yt[idx])
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def infer(m: OrdinalCapsNet, X: np.ndarray, device) -> np.ndarray:
    m.eval()
    out = []
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    for i in range(0, len(Xt), 512):
        out.append(m(Xt[i:i + 512])["head_lengths"].cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=K)
    args = ap.parse_args()
    device = torch.device("cpu")          # frozen-feature head: CPU is as fast

    Xs_all = np.load(APTOS_F).astype(np.float32)
    ys_all = np.load(APTOS_Y).astype(np.int64)
    Xt = np.load(MESS_F).astype(np.float32)
    yt = np.load(MESS_Y).astype(np.int64)
    keep = yt >= 0
    Xt, yt = Xt[keep], yt[keep]
    print(f"APTOS {Xs_all.shape}  Messidor-2 {Xt.shape}")

    pool_idx, _ = train_test_split(np.arange(len(ys_all)), test_size=HOLDOUT,
                                   stratify=ys_all, random_state=SEED)
    Xs, ys = Xs_all[pool_idx], ys_all[pool_idx]

    # Pre-compute the aligned target views once (independent of fold).
    aligned = {k: f(Xs, Xt).astype(np.float32) for k, f in ALIGNERS.items()}
    for k, v in aligned.items():
        print(f"  align {k:8s}: target mean|delta| vs source "
              f"{np.abs(v.mean(0) - Xs.mean(0)).mean():.4f}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    src_prior = np.bincount(ys, minlength=5) / len(ys)
    tgt_oracle = np.bincount(yt, minlength=5) / len(yt)

    acc: dict[tuple[str, str], list[dict]] = {}
    for fold, (tr, va) in enumerate(skf.split(Xs, ys), 1):
        m = train_fold(Xs[tr], ys[tr], device)
        # source-side confusion matrix for BBSE, from the held-out fold
        h_va = infer(m, Xs[va], device)
        p_va = chain_probs(h_va)
        pred_va = p_va.argmax(1)
        conf = np.zeros((5, 5))
        for t, q in zip(ys[va], pred_va):
            conf[t, q] += 1
        conf /= np.maximum(conf.sum(1, keepdims=True), 1e-12)

        for aname, Xa in aligned.items():
            probs = chain_probs(infer(m, Xa, device))
            # Severity score for rank-based corrections: expected grade.
            sev = probs @ np.arange(5)
            for cname in ("none", "BBSE", "EM/SLD", "oracle",
                          "rank(src prior)", "rank(oracle)"):
                if cname.startswith("rank"):
                    pri = src_prior if "src" in cname else tgt_oracle
                    pred = rank_calibrate(sev, pri)
                    mm = compute_all_metrics(yt, pred, num_classes=5)
                    rd, rt = (pred >= 2), (yt >= 2)
                    tp_, fn_ = int((rd & rt).sum()), int((~rd & rt).sum())
                    tn_, fp_ = int((~rd & ~rt).sum()), int((rd & ~rt).sum())
                    mm["rdr_sensitivity"] = tp_ / max(tp_ + fn_, 1)
                    mm["rdr_specificity"] = tn_ / max(tn_ + fp_, 1)
                    acc.setdefault((aname, cname), []).append(mm)
                    continue
                if cname == "none":
                    pr = probs
                else:
                    tp = {"BBSE": lambda: bbse_prior(probs, conf, src_prior),
                          "EM/SLD": lambda: em_prior(probs, src_prior),
                          "oracle": lambda: tgt_oracle}[cname]()
                    pr = reweight(probs, src_prior, tp)
                pred = pr.argmax(1)
                mm = compute_all_metrics(yt, pred, num_classes=5)
                rd = (pred >= 2)
                rt = (yt >= 2)
                tp_, fn_ = int((rd & rt).sum()), int((~rd & rt).sum())
                tn_, fp_ = int((~rd & ~rt).sum()), int((rd & ~rt).sum())
                mm["rdr_sensitivity"] = tp_ / max(tp_ + fn_, 1)
                mm["rdr_specificity"] = tn_ / max(tn_ + fp_, 1)
                acc.setdefault((aname, cname), []).append(mm)
        print(f"  fold {fold}/{args.folds} done")

    rows = []
    for (aname, cname), ms in acc.items():
        rows.append({
            "align": aname, "correct": cname,
            **{k: float(np.mean([m[k] for m in ms])) for k in
               ("qwk", "accuracy", "macro_f1", "mae",
                "rdr_sensitivity", "rdr_specificity")},
            "qwk_std": float(np.std([m["qwk"] for m in ms])),
        })
    rows.sort(key=lambda r: -r["qwk"])

    out = Path("results")
    (out / "uda_metrics.json").write_text(json.dumps(rows, indent=2))
    md = ["# Unsupervised domain adaptation: APTOS -> Messidor-2",
          "",
          "Ordinal CapsNet on frozen RETFound features, trained on APTOS only.",
          "Feature alignment maps the target into the source space; label",
          "correction reweights the decoded posterior. Everything except",
          "`oracle` is label-free on the target. 5 folds, mean over folds.",
          "",
          "| Align | Correct | QWK | Acc | Macro-F1 | MAE | rDR sens | rDR spec |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['align']} | {r['correct']} | {r['qwk']:.4f} ± {r['qwk_std']:.4f} "
                  f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['mae']:.3f} "
                  f"| {r['rdr_sensitivity']:.3f} | {r['rdr_specificity']:.3f} |")
    (out / "uda_table.md").write_text("\n".join(md) + "\n")

    csv = ["align,correct,qwk,qwk_std,accuracy,macro_f1,mae,rdr_sensitivity,rdr_specificity"]
    for r in rows:
        csv.append(f"{r['align']},{r['correct']},{r['qwk']:.6f},{r['qwk_std']:.6f},"
                   f"{r['accuracy']:.6f},{r['macro_f1']:.6f},{r['mae']:.6f},"
                   f"{r['rdr_sensitivity']:.6f},{r['rdr_specificity']:.6f}")
    (out / "uda_table.csv").write_text("\n".join(csv) + "\n")

    print("\nTop rows:")
    for r in rows[:6]:
        print(f"  {r['align']:8s} + {r['correct']:8s}  QWK {r['qwk']:.4f}  "
              f"sens {r['rdr_sensitivity']:.3f}  spec {r['rdr_specificity']:.3f}")
    print("\nWrote results/uda_table.{md,csv} + results/uda_metrics.json")


if __name__ == "__main__":
    main()
