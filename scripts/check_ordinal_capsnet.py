"""Day 3 — sanity checks for the Ordinal CapsNet before full training.

Runs each red-flag check from the Day 3 plan in order, prints OK/WARN/FAIL,
and short-circuits on a hard failure so we can debug incrementally.

Usage: uv run python scripts/check_ordinal_capsnet.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score

from src.losses.margin_loss import MarginLoss  # for comparison baseline
from src.losses.ordinal_loss import (
    OrdinalMarginLoss,
    count_non_monotonic,
    ordinal_labels,
    predict_grade_from_heads,
)
from src.models.ordinal_capsnet import OrdinalCapsNet


def banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Check 1: Label generation
# =============================================================================

def check_labels() -> bool:
    banner("Check 1: Ordinal label encoding y -> [y>0, y>1, y>2, y>3]")
    expected = {
        0: [0, 0, 0, 0],
        1: [1, 0, 0, 0],
        2: [1, 1, 0, 0],
        3: [1, 1, 1, 0],
        4: [1, 1, 1, 1],
    }
    y = torch.tensor(list(expected.keys()))
    got = ordinal_labels(y, num_classes=5).long().tolist()

    ok = True
    for i, grade in enumerate(expected):
        got_i = got[i]
        exp_i = expected[grade]
        match = got_i == exp_i
        status = "OK  " if match else "FAIL"
        print(f"  [{status}] y={grade} -> {got_i}   (expected {exp_i})")
        if not match:
            ok = False
    return ok


# =============================================================================
# Check 2: Shape + independence of head weights
# =============================================================================

def check_shapes_and_weights(device: torch.device) -> bool:
    banner("Check 2: Forward-pass shapes and independent head weights")
    B = 4
    model = OrdinalCapsNet(feature_dim=1024, num_primary=32, primary_dim=8,
                           num_classes=5, caps_dim=16, routing_iters=3).to(device)
    x = torch.randn(B, 1024, device=device)

    out = model(x)
    caps = out["head_caps"]      # (B, num_heads, 2, caps_dim)
    lengths = out["head_lengths"]  # (B, num_heads, 2)
    couplings = out["head_couplings"]  # list

    ok = True

    # Shape checks
    print(f"  head_caps.shape    = {tuple(caps.shape)} (expected (4, 4, 2, 16))")
    if tuple(caps.shape) != (4, 4, 2, 16):
        ok = False; print("    FAIL")
    print(f"  head_lengths.shape = {tuple(lengths.shape)} (expected (4, 4, 2))")
    if tuple(lengths.shape) != (4, 4, 2):
        ok = False; print("    FAIL")

    # Each head's output slice shape
    head_0 = caps[:, 0]
    print(f"  caps[:, 0].shape   = {tuple(head_0.shape)} (expected (4, 2, 16))")
    if tuple(head_0.shape) != (4, 2, 16):
        ok = False

    # Each head's lengths slice shape
    head_0_len = lengths[:, 0]
    print(f"  lengths[:, 0].shape = {tuple(head_0_len.shape)} (expected (4, 2))")
    if tuple(head_0_len.shape) != (4, 2):
        ok = False

    # Independence of weight matrices
    print()
    ids = [id(h.W) for h in model.heads]
    print(f"  head W tensor ids:  {ids}")
    unique = len(set(ids))
    if unique != len(ids):
        print("  FAIL: at least two heads share the same weight tensor")
        ok = False
    else:
        print(f"  OK: {len(ids)} heads have {unique} distinct weight tensors")

    # Couplings per head
    for k, c in enumerate(couplings):
        expected_shape = (B, 32, 2)
        if tuple(c.shape) != expected_shape:
            ok = False
            print(f"  FAIL: head {k} couplings shape {tuple(c.shape)} != {expected_shape}")
    print(f"  OK: all couplings are {(B, 32, 2)} per head")
    return ok


# =============================================================================
# Check 3: Gradient flow per head (before training)
# =============================================================================

def check_gradients(device: torch.device) -> bool:
    banner("Check 3: Each head receives a gradient")
    B = 16
    model = OrdinalCapsNet(num_classes=5).to(device)
    x = torch.randn(B, 1024, device=device)
    y = torch.randint(0, 5, (B,), device=device)

    out = model(x)
    loss_fn = OrdinalMarginLoss(num_classes=5)
    loss, per_head = loss_fn(out["head_lengths"], y, return_per_head=True)
    loss.backward()

    ok = True
    print(f"  overall loss = {loss.item():.4f}")
    print(f"  per-head losses: {[f'{x:.4f}' for x in per_head.tolist()]}")
    for k, h in enumerate(model.heads):
        g = h.W.grad
        gnorm = g.norm().item() if g is not None else 0.0
        print(f"    head {k}: W.grad.norm = {gnorm:.6f}",
              end="")
        if g is None or gnorm == 0.0:
            print("   FAIL: no gradient")
            ok = False
        else:
            print("   OK")
    # also check shared primary
    gnorm_p = model.primary.linear.weight.grad.norm().item()
    print(f"  primary.linear.grad.norm = {gnorm_p:.6f}", end="")
    if gnorm_p == 0.0:
        print("   FAIL")
        ok = False
    else:
        print("   OK")
    return ok


# =============================================================================
# Check 4 + 5: First epoch & 10 epoch sanity on real cached features
# =============================================================================

def per_head_binary_accuracy(head_lengths: torch.Tensor, y: torch.Tensor,
                             num_classes: int = 5) -> list[float]:
    """For each head, compute binary accuracy: argmax of 2 capsules vs y>k."""
    p_gt_k = head_lengths[:, :, 1]  # (B, K-1)
    bin_preds = (p_gt_k > 0.5).long().cpu().numpy()
    bin_true = ordinal_labels(y, num_classes).long().cpu().numpy()
    return [accuracy_score(bin_true[:, k], bin_preds[:, k])
            for k in range(num_classes - 1)]


@torch.no_grad()
def eval_full(model, features: np.ndarray, labels: np.ndarray, device, batch: int = 128):
    model.eval()
    all_caps_lengths = []
    n = len(features)
    for i in range(0, n, batch):
        X = torch.from_numpy(features[i:i+batch]).float().to(device)
        out = model(X)
        all_caps_lengths.append(out["head_lengths"].cpu())
    return torch.cat(all_caps_lengths, dim=0)


def check_training(device: torch.device) -> bool:
    banner("Check 4+5: First epoch + 10-epoch convergence on real features")
    # Load cached features (use small random val subset for quick diagnostic)
    features = np.load("data/aptos/features/train_features.npy")
    labels = np.load("data/aptos/features/train_labels.npy")
    rng = np.random.default_rng(42)
    # stratified-ish: sample 2000 train / 500 val
    idx = rng.permutation(len(features))
    tr_idx, va_idx = idx[:2000], idx[2000:2500]
    X_tr, y_tr = features[tr_idx], labels[tr_idx]
    X_va, y_va = features[va_idx], labels[va_idx]

    # Class balance per head (on train subset) — informational
    bin_labels = ordinal_labels(torch.from_numpy(y_tr), 5).numpy()
    print("  Train-subset positive rate per head (head_k = y>k):")
    for k in range(4):
        print(f"    head {k} (y > {k}): {bin_labels[:, k].mean() * 100:.1f}%  "
              f"(spec: ~{[54, 37, 16, 5][k]}%)")

    # Set up model + optimizer
    model = OrdinalCapsNet(num_classes=5, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = OrdinalMarginLoss(num_classes=5)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    X_va_t = torch.from_numpy(X_va).float()
    y_va_t = torch.from_numpy(y_va).long()

    ok = True
    epoch_diag: list[dict] = []

    for epoch in range(1, 11):
        # Train
        model.train()
        perm = torch.randperm(len(X_tr_t))
        batch_size = 128
        per_head_loss_sum = torch.zeros(4)
        n_batches = 0
        for i in range(0, len(X_tr_t), batch_size):
            bi = perm[i:i+batch_size]
            X, y = X_tr_t[bi].to(device), y_tr_t[bi].to(device)
            optimizer.zero_grad()
            out = model(X)
            loss, per_head = loss_fn(out["head_lengths"], y, return_per_head=True)
            loss.backward()
            optimizer.step()
            per_head_loss_sum += per_head.detach().cpu()
            n_batches += 1
        train_per_head = (per_head_loss_sum / n_batches).tolist()

        # Val diag
        val_lengths = eval_full(model, X_va, y_va, device)
        val_grade_pred = predict_grade_from_heads(val_lengths).numpy()
        val_qwk = cohen_kappa_score(y_va, val_grade_pred, weights="quadratic")
        val_acc = accuracy_score(y_va, val_grade_pred)
        per_head_acc = per_head_binary_accuracy(val_lengths, y_va_t, num_classes=5)

        epoch_diag.append({
            "epoch": epoch,
            "train_per_head_loss": train_per_head,
            "val_qwk": val_qwk,
            "val_acc": val_acc,
            "per_head_val_acc": per_head_acc,
        })

        if epoch in (1, 5, 10):
            print(f"\n  -- Epoch {epoch} --")
            print(f"  per-head train loss:  {[f'{x:.3f}' for x in train_per_head]}")
            print(f"  per-head val binary accuracy: {[f'{x:.3f}' for x in per_head_acc]}")
            print(f"  val QWK (grade)={val_qwk:.4f}  Acc={val_acc:.4f}")

            # Spec checks at each epoch
            if epoch == 1:
                # Per-head loss spread
                mx, mn = max(train_per_head), min(train_per_head)
                if mx > 10 * mn:
                    print(f"  FAIL: head losses differ by >10x ({mx:.3f} vs {mn:.3f})")
                    ok = False
                # Per-head val accuracy > 55%
                for k, acc in enumerate(per_head_acc):
                    if acc < 0.55:
                        print(f"  WARN: head {k} epoch-1 acc = {acc:.3f} (< 0.55). "
                              f"May be slow to learn minority class.")
                if not any(acc < 0.55 for acc in per_head_acc):
                    print("  OK: all heads above random after 1 epoch")

    # Convergence check — any head's loss INCREASING over the 10 epochs?
    print("\n  Per-head loss trajectory (epochs 1 -> 10):")
    for k in range(4):
        trajectory = [d["train_per_head_loss"][k] for d in epoch_diag]
        ep1, ep10 = trajectory[0], trajectory[-1]
        direction = "decreasing" if ep10 < ep1 else "FLAT/INCREASING"
        print(f"    head {k}: {ep1:.3f} -> {ep10:.3f}  ({direction})")
        if ep10 >= ep1:
            print(f"      FAIL: head {k} loss did not decrease")
            ok = False

    final_qwk = epoch_diag[-1]["val_qwk"]
    print(f"\n  Val QWK after 10 epochs = {final_qwk:.4f}")
    if final_qwk < 0.80:
        print(f"  WARN: QWK below 0.80 after 10 epochs — may indicate architectural issue")

    # Keep the model for check 6 (monotonicity)
    return ok, model, X_va, y_va, device


# =============================================================================
# Check 6: Monotonicity + per-head accuracy on the trained model
# =============================================================================

def check_monotonicity(model, X_va: np.ndarray, y_va: np.ndarray,
                      device: torch.device) -> bool:
    banner("Check 6: Monotonicity + per-head eval on 10-epoch model")
    val_lengths = eval_full(model, X_va, y_va, device)
    n = val_lengths.size(0)

    # Per-head accuracy
    y_va_t = torch.from_numpy(y_va).long()
    per_head_acc = per_head_binary_accuracy(val_lengths, y_va_t, 5)
    names = ["y>0", "y>1", "y>2", "y>3"]
    thresholds = [0.70, 0.80, 0.75, 0.70]
    print(f"  Per-head val binary accuracy (10 epochs):")
    ok = True
    for k, (nm, acc, thr) in enumerate(zip(names, per_head_acc, thresholds)):
        status = "OK" if acc >= thr else "WARN"
        print(f"    head {k} {nm:>4s}: acc={acc:.3f}  (target > {thr:.2f})  [{status}]")
        if acc < thr - 0.10:  # allow 10pt slack since only 10 epochs
            ok = False

    # Monotonicity rate
    n_non_mono = count_non_monotonic(val_lengths).item()
    rate = n_non_mono / n
    print(f"\n  Non-monotonic predictions: {n_non_mono}/{n} = {rate * 100:.2f}%")
    if rate < 0.03:
        print("  OK: < 3% non-monotonic (joint training is consistent)")
    elif rate < 0.10:
        print("  NOTE: 3-10% non-monotonic (acceptable; sum-based inference still works)")
    else:
        print("  WARN: > 10% non-monotonic — consider adding a consistency penalty")
        ok = False

    # Grade-level metrics
    preds = predict_grade_from_heads(val_lengths).numpy()
    qwk = cohen_kappa_score(y_va, preds, weights="quadratic")
    acc = accuracy_score(y_va, preds)
    print(f"  Val grade QWK={qwk:.4f}  Acc={acc:.4f}")
    return ok


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    passed: list[tuple[str, bool]] = []

    passed.append(("labels", check_labels()))
    if not passed[-1][1]:
        print("\nSTOP: label encoding is wrong — downstream will be silently broken.")
        sys.exit(1)

    passed.append(("shapes+weights", check_shapes_and_weights(device)))
    passed.append(("gradients", check_gradients(device)))
    if not passed[-1][1]:
        print("\nSTOP: gradient flow is broken.")
        sys.exit(1)

    t0 = time.time()
    train_ok, model, X_va, y_va, _ = check_training(device)
    passed.append(("training_10ep", train_ok))
    print(f"\n  (10-epoch diagnostic took {time.time() - t0:.1f}s on {device})")

    passed.append(("monotonicity", check_monotonicity(model, X_va, y_va, device)))

    # Summary
    banner("Summary")
    all_ok = True
    for name, ok in passed:
        status = "PASS" if ok else "FAIL/WARN"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False
    print()
    print("All checks passed" if all_ok else "Some checks need attention — see above")


if __name__ == "__main__":
    main()
