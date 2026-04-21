"""LoRA fine-tuning of RETFound + Ordinal CapsNet on APTOS.

Same 10% holdout + 5-fold stratified CV split as Day 3 (seed=42) so results are
directly comparable to the frozen-backbone ablation row. LoRA rank 8, alpha 16,
targets `attn.qkv` in all 24 ViT-L blocks. No W&B logging by design.

Optimizer has two param groups:
    LoRA adapters       lr 1e-4   (gentle adaptation)
    OrdinalCapsNet head lr 1e-3   (same as Day 3)

Outputs per fold:
    results/lora_ordinal_capsnet/preds_fold{N}.npz
    results/lora_ordinal_capsnet/weights_fold{N}.pt        (LoRA + head only)
Final:
    results/lora_ordinal_capsnet/summary.json

Usage:
    uv run python scripts/run_lora_ordinal.py                  # full 5-fold
    uv run python scripts/run_lora_ordinal.py --folds 0        # fold 0 only
    uv run python scripts/run_lora_ordinal.py --sanity         # 1 fold, 1 epoch, tiny batch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.data.image_dataset import FundusImageDataset
from src.evaluate import compute_all_metrics
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.retfound_lora_capsnet import RetfoundLoraOrdinalCapsNet


APTOS_CSV = Path("data/aptos/train.csv")
APTOS_IMG = Path("data/aptos/train_images")
OUT_DIR = Path("results/lora_ordinal_capsnet")

NUM_CLASSES = 5
SEED = 42
HOLDOUT_FRAC = 0.10
N_FOLDS = 5

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 8
DEFAULT_BATCH = 8
LR_LORA = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_aptos_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_ids, train_labels) after dropping the 10% holdout
    (same seed as Day 3 so CV folds match exactly)."""
    ids: list[str] = []
    labels: list[int] = []
    with APTOS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id_code"])
            labels.append(int(row["diagnosis"]))
    ids_arr = np.array(ids)
    labels_arr = np.array(labels, dtype=np.int64)
    pool_idx, holdout_idx = train_test_split(
        np.arange(len(ids_arr)), test_size=HOLDOUT_FRAC,
        stratify=labels_arr, random_state=SEED,
    )
    return ids_arr[pool_idx], labels_arr[pool_idx], ids_arr[holdout_idx], labels_arr[holdout_idx]


def make_loaders(ids_tr, y_tr, ids_va, y_va, batch_size: int, num_workers: int):
    ds_tr = FundusImageDataset(ids_tr, y_tr, APTOS_IMG, image_ext=".png")
    ds_va = FundusImageDataset(ids_va, y_va, APTOS_IMG, image_ext=".png")
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=False),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=False),
    )


def build_optimizer(model: RetfoundLoraOrdinalCapsNet) -> torch.optim.Optimizer:
    """Two param groups: LoRA adapters + OrdinalCapsNet head."""
    lora_params, head_params, other_trainable = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(p)
        elif "lora_" in name:
            lora_params.append(p)
        else:
            other_trainable.append((name, p))
    if other_trainable:
        print("  WARN: trainable params outside LoRA + head:")
        for name, _ in other_trainable:
            print(f"    {name}")
        # Include them with LoRA lr so they still train (but warn)
        lora_params.extend(p for _, p in other_trainable)

    return torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])


@torch.no_grad()
def evaluate(model: RetfoundLoraOrdinalCapsNet, loader: DataLoader,
             loss_fn: OrdinalMarginLoss, device: torch.device) -> dict:
    model.eval()
    all_lengths, all_labels, loss_sum, n = [], [], 0.0, 0
    use_bf16 = device.type == "cuda"
    for X, y in loader:
        X = X.to(device)
        y_dev = y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_bf16):
            L = model(X)["head_lengths"]
            loss_val = loss_fn(L, y_dev)
        loss_sum += loss_val.item() * X.size(0)
        n += X.size(0)
        # Cast back to fp32 before CPU transfer — downstream metrics / UQ assume fp32.
        all_lengths.append(L.float().cpu())
        all_labels.append(y)
    lengths = torch.cat(all_lengths, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    preds = predict_grade_from_heads(lengths).numpy().astype(np.int64)
    m = compute_all_metrics(labels, preds, num_classes=NUM_CLASSES)
    m["loss"] = loss_sum / n
    m["_y_true"] = labels
    m["_y_pred"] = preds
    m["_head_lengths"] = lengths.numpy().astype(np.float32)
    return m


def train_fold(fold_idx: int, ids_tr, y_tr, ids_va, y_va,
               device: torch.device, epochs: int, patience: int,
               batch_size: int, num_workers: int,
               out_dir: Path, log_every: int) -> dict:
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)

    tr_loader, va_loader = make_loaders(ids_tr, y_tr, ids_va, y_va,
                                        batch_size, num_workers)
    model = RetfoundLoraOrdinalCapsNet(
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(device)

    report = model.trainable_params_report()
    print(f"  trainable: {report['trainable']:,}  |  frozen: {report['frozen']:,}")

    optim = build_optimizer(model)
    loss_fn = OrdinalMarginLoss(num_classes=NUM_CLASSES)

    best_qwk, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    history: list[dict] = []

    t_fold = time.time()
    # bf16 autocast on CUDA (H100/A100/T4+) ~2-3x speedup; no-op on MPS/CPU.
    use_bf16 = device.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum, n = 0.0, 0
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                L = model(X)["head_lengths"]
                loss = loss_fn(L, y)
            loss.backward()
            optim.step()
            train_loss_sum += loss.item() * X.size(0)
            n += X.size(0)
        train_loss = train_loss_sum / n
        train_sec = time.time() - t0

        val_m = evaluate(model, va_loader, loss_fn, device)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_m["loss"], "val_qwk": val_m["qwk"],
            "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
            "train_sec": train_sec,
        })
        improved = val_m["qwk"] > best_qwk
        if improved:
            best_qwk = val_m["qwk"]
            best_epoch = epoch
            # Save only the trainable subset (LoRA + head) so the .pt is tiny
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()
                          if ("lora_" in k) or k.startswith("head.")}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % log_every == 0 or improved or no_improve >= patience:
            print(f"    ep {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_qwk={val_m['qwk']:.4f}  val_acc={val_m['accuracy']:.4f}  "
                  f"val_f1={val_m['macro_f1']:.4f}  ({train_sec:.1f}s train)"
                  + ("  *best*" if improved else ""))

        if no_improve >= patience:
            print(f"    early stop @ ep {epoch} (best qwk={best_qwk:.4f} @ ep {best_epoch})")
            break

    # Restore best state for final eval + saving
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    final_m = evaluate(model, va_loader, loss_fn, device)

    # Persist preds + weights
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"preds_fold{fold_idx + 1}.npz",
        y_true=final_m["_y_true"], y_pred=final_m["_y_pred"],
        head_lengths=final_m["_head_lengths"],
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"weights_fold{fold_idx + 1}.pt")

    fold_summary = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "final_epoch": history[-1]["epoch"],
        "qwk": final_m["qwk"], "accuracy": final_m["accuracy"],
        "macro_f1": final_m["macro_f1"], "mae": final_m["mae"],
        "loss": final_m["loss"],
        "trainable_params": report["trainable"],
        "frozen_params": report["frozen"],
        "elapsed_sec": time.time() - t_fold,
        "history": history,
    }
    print(f"  fold {fold_idx + 1} DONE  best val QWK={best_qwk:.4f}  "
          f"final QWK={final_m['qwk']:.4f}  ({fold_summary['elapsed_sec']:.1f}s)")
    return fold_summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="Which fold indices (0..4) to run. Default: all 5.")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--sanity", action="store_true",
                    help="1 fold, 1 epoch, batch 4 — for plumbing verification.")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--resume", action="store_true",
                    help="Skip folds whose preds_fold{N}.npz already exists in --out-dir. "
                         "Safe to re-run after interruption: finished folds are preserved; "
                         "only missing folds are re-trained.")
    ap.add_argument("--force-redo", action="store_true",
                    help="Opposite of --resume: re-train every requested fold even if "
                         "preds already exist (overwriting them). Use with care.")
    return ap.parse_args()


def _load_existing_summary(out_dir: Path) -> list[dict]:
    """Recover per_fold entries from an earlier interrupted run, if any."""
    s_path = out_dir / "summary.json"
    if not s_path.exists():
        return []
    try:
        data = json.loads(s_path.read_text())
        return list(data.get("per_fold", []))
    except Exception:
        return []


def main() -> None:
    args = parse_args()
    if args.sanity:
        args.folds = [0]
        args.epochs = 1
        args.batch_size = 4
        args.patience = 1
        args.log_every = 1

    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}  patience: {args.patience}  "
          f"batch: {args.batch_size}  lora_r: {LORA_R}  lora_alpha: {LORA_ALPHA}\n")

    ids_pool, y_pool, ids_holdout, y_holdout = load_aptos_split()
    print(f"APTOS CV pool: {len(ids_pool)}  holdout: {len(ids_holdout)}")
    print(f"Pool label dist: {np.bincount(y_pool).tolist()}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(ids_pool, y_pool))
    if args.folds is None:
        args.folds = list(range(N_FOLDS))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume / skip logic ------------------------------------------------
    # Default behaviour (no flags): NEW run — overwrite summary.json, re-train
    #   every requested fold. This matches the original contract.
    # --resume: reuse prior run's summary.json + preds; only train missing folds.
    # --force-redo: ignore on-disk artifacts and re-train everything.
    previously_done: dict[int, dict] = {}
    if args.resume and not args.force_redo:
        for entry in _load_existing_summary(out_dir):
            fold_number = entry.get("fold")  # 1-indexed as written by train_fold
            if isinstance(fold_number, int):
                previously_done[fold_number - 1] = entry
        # Also accept any bare preds_fold{N}.npz whose summary got lost.
        for npz in out_dir.glob("preds_fold*.npz"):
            try:
                n = int(npz.stem.replace("preds_fold", ""))
                if (n - 1) not in previously_done:
                    # Unknown status — skip it anyway so we don't clobber a completed fold.
                    previously_done[n - 1] = {
                        "fold": n, "resumed_from_preds_only": True,
                        "qwk": float("nan"), "accuracy": float("nan"),
                        "macro_f1": float("nan"), "mae": float("nan"),
                    }
            except ValueError:
                continue
        if previously_done:
            print(f"\nResume mode: {len(previously_done)} fold(s) already on disk — "
                  f"fold indices {sorted(previously_done.keys())}  (will skip).\n")

    per_fold: list[dict] = [previously_done[k] for k in sorted(previously_done)]

    t_global = time.time()
    for fold_idx in args.folds:
        if fold_idx in previously_done and not args.force_redo:
            print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS}  SKIPPED (resume; preds on disk) ===")
            continue
        tr_idx, va_idx = splits[fold_idx]
        print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS}  "
              f"(train={len(tr_idx)}, val={len(va_idx)}) ===")
        summary = train_fold(
            fold_idx, ids_pool[tr_idx], y_pool[tr_idx],
            ids_pool[va_idx], y_pool[va_idx],
            device, args.epochs, args.patience,
            args.batch_size, args.num_workers,
            out_dir, args.log_every,
        )
        per_fold.append(summary)
        # Save partial summary after every fold — so if interrupted we still have data.
        (out_dir / "summary.json").write_text(json.dumps({
            "config": {
                "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "epochs": args.epochs, "patience": args.patience,
                "batch_size": args.batch_size,
                "lr_lora": LR_LORA, "lr_head": LR_HEAD,
                "weight_decay": WEIGHT_DECAY, "seed": SEED,
                "holdout_frac": HOLDOUT_FRAC, "n_folds": N_FOLDS,
                "sanity": args.sanity,
            },
            "per_fold": per_fold,
            "n_folds_completed": len(per_fold),
            "elapsed_sec": time.time() - t_global,
        }, indent=2))

    # Aggregate
    qwks = [f["qwk"] for f in per_fold]
    accs = [f["accuracy"] for f in per_fold]
    f1s = [f["macro_f1"] for f in per_fold]
    maes = [f["mae"] for f in per_fold]

    print("\n" + "=" * 78)
    print(f"LoRA OrdinalCapsNet  —  {len(per_fold)} fold(s)  —  "
          f"total {time.time() - t_global:.1f}s")
    print("=" * 78)
    print(f"QWK     : {np.mean(qwks):.4f} ± {np.std(qwks):.4f}  "
          f"(fold QWKs: {[f'{q:.4f}' for q in qwks]})")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"MAE     : {np.mean(maes):.4f} ± {np.std(maes):.4f}")


if __name__ == "__main__":
    main()
