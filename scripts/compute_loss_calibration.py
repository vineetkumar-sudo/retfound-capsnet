"""Which component causes the miscalibration: the loss, or the architecture?

Reviewer 2 comment 7: "Why is poor calibration associated with margin loss
without component-wise analysis?"

The submitted paper attributes the capsule head's poor ECE to the Sabour
margin loss, but never separates the loss from the architecture -- every
capsule row uses the margin loss and every comparison changes both at once.
This runs the full 2x3 factorial so the two factors are identified:

    head   in {Ordinal CapsNet, MLP + K-1 sigmoid}
    loss   in {margin (Sabour), BCE, focal}

Everything else is held fixed: same cached RETFound features, same K-1
decomposition, same splits, same optimiser, 3 seeds x 5 folds per cell.

ECE is reported under both decodings, since Item-2 showed the decode itself
dominates calibration:
    ECE(product)    product decode + clip + renormalise (as submitted)
    ECE(projected)  monotone PAV projection + cumulative differences

Outputs results/loss_calibration_table.{md,csv} and
results/loss_calibration_metrics.json.

Usage:
    uv run python scripts/compute_loss_calibration.py [--seeds 42,123,456]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (
    chain_rule_probs,
    expected_calibration_error,
    max_confidence,
)
from src.evaluate import compute_all_metrics
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.mlp_ordinal import MLPOrdinal
from src.models.ordinal_capsnet import OrdinalCapsNet

FEATS = Path("data/aptos/features/train_features.npy")
LABELS = Path("data/aptos/features/train_labels.npy")
SEED, K, HOLDOUT = 42, 5, 0.10
EPOCHS, BATCH, LR, WD = 100, 64, 1e-3, 1e-4
PATIENCE = 30


def targets(y: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """K-1 binary targets: t_k = 1[y > k]."""
    ks = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > ks).float()


class HeadBCE(nn.Module):
    """Binary cross-entropy on P(y>k), read off the positive capsule length."""

    def forward(self, head_lengths: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = head_lengths[:, :, 1].clamp(1e-6, 1 - 1e-6)
        return nn.functional.binary_cross_entropy(p, targets(y, p.shape[1] + 1))


class HeadFocal(nn.Module):
    """Focal loss (gamma=2) on the same per-head probabilities."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, head_lengths: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = head_lengths[:, :, 1].clamp(1e-6, 1 - 1e-6)
        t = targets(y, p.shape[1] + 1)
        pt = p * t + (1 - p) * (1 - t)
        return (-(1 - pt).pow(self.gamma) * pt.log()).mean()


LOSSES = {"margin": lambda: OrdinalMarginLoss(num_classes=5),
          "BCE": HeadBCE, "focal": HeadFocal}
HEADS = {"Ordinal CapsNet": OrdinalCapsNet, "MLP + K-1 sigmoid": MLPOrdinal}


def run_fold(head_name, loss_name, Xtr, ytr, Xva, yva, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HEADS[head_name]().to(device)
    lf = LOSSES[loss_name]()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)

    best_qwk, best_state, bad = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lf(model(Xt[idx])["head_lengths"], yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            L = model(Xv)["head_lengths"].cpu()
        q = compute_all_metrics(yva, predict_grade_from_heads(L).numpy())["qwk"]
        if q > best_qwk:
            best_qwk, bad = q, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        L = model(Xv)["head_lengths"].cpu().numpy()
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,456")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cpu")

    X = np.load(FEATS).astype(np.float32)
    y = np.load(LABELS).astype(np.int64)
    pool, _ = train_test_split(np.arange(len(y)), test_size=HOLDOUT,
                               stratify=y, random_state=SEED)
    X, y = X[pool], y[pool]
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    rows = []
    for head_name in HEADS:
        for loss_name in LOSSES:
            per_seed = []
            for seed in seeds:
                Ls, ys = [], []
                for tr, va in folds:
                    Ls.append(run_fold(head_name, loss_name, X[tr], y[tr],
                                       X[va], y[va], seed, device))
                    ys.append(y[va])
                L = np.concatenate(Ls)
                yy = np.concatenate(ys)
                h = L[:, :, 1]
                pred = predict_grade_from_heads(torch.tensor(L)).numpy()
                m = compute_all_metrics(yy, pred)
                out = {"qwk": m["qwk"], "accuracy": m["accuracy"]}
                for tag, kw in (("product", dict()),
                                ("projected", dict(decode="difference", project=True))):
                    p, renorm = chain_rule_probs(h, **kw)
                    c, pr = max_confidence(p)
                    out[f"ece_{tag}"] = expected_calibration_error(
                        c, (pr == yy).astype(float), n_bins=10, mass=True)
                    out[f"renorm_{tag}"] = renorm
                viol = float((np.diff(h, axis=1) > 0).any(axis=1).mean())
                out["violation_rate"] = viol
                per_seed.append(out)
            agg = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
            agg |= {k + "_std": float(np.std([d[k] for d in per_seed]))
                    for k in ("qwk", "ece_product", "ece_projected")}
            agg |= {"head": head_name, "loss": loss_name, "seeds": seeds}
            rows.append(agg)
            print(f"  {head_name:<20s} {loss_name:<7s} QWK {agg['qwk']:.4f}  "
                  f"ECE(prod) {agg['ece_product']:.4f}  "
                  f"ECE(proj) {agg['ece_projected']:.4f}  "
                  f"viol {100 * agg['violation_rate']:.1f}%")

    out_dir = Path("results")
    (out_dir / "loss_calibration_metrics.json").write_text(json.dumps(rows, indent=2))
    md = ["# Loss x head factorial: what actually drives the miscalibration?",
          "",
          f"APTOS, frozen RETFound features, {len(seeds)} seeds x {K}-fold CV per cell.",
          "Only the head architecture and the training loss vary; splits,",
          "optimiser and decomposition are identical throughout.",
          "",
          "`ECE(product)` uses the submitted decode (product + clip + renormalise);",
          "`ECE(projected)` uses the monotone PAV projection + cumulative decode.",
          "`viol%` is the rate of rank-monotonicity violations in the raw heads.",
          "",
          "| Head | Loss | QWK | Acc | ECE (product) | ECE (projected) | viol% |",
          "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['head']} | {r['loss']} | {r['qwk']:.4f} ± {r['qwk_std']:.4f} "
                  f"| {r['accuracy']:.4f} | {r['ece_product']:.4f} ± {r['ece_product_std']:.4f} "
                  f"| {r['ece_projected']:.4f} | {100 * r['violation_rate']:.1f}% |")
    (out_dir / "loss_calibration_table.md").write_text("\n".join(md) + "\n")
    csv = ["head,loss,qwk,qwk_std,accuracy,ece_product,ece_projected,violation_rate"]
    for r in rows:
        csv.append(f"{r['head']},{r['loss']},{r['qwk']:.6f},{r['qwk_std']:.6f},"
                   f"{r['accuracy']:.6f},{r['ece_product']:.6f},"
                   f"{r['ece_projected']:.6f},{r['violation_rate']:.6f}")
    (out_dir / "loss_calibration_table.csv").write_text("\n".join(csv) + "\n")
    print("\nWrote results/loss_calibration_table.{md,csv} + "
          "results/loss_calibration_metrics.json")


if __name__ == "__main__":
    main()
