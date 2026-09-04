"""LoRA fine-tuning on Messidor-2, mirroring the APTOS LoRA protocol.

The submitted paper evaluates four tuning tiers on APTOS (frozen, LoRA,
progressive unfreezing, full fine-tune) but only the frozen tier on
Messidor-2, which leaves the experimental design asymmetric: every
fine-tuning conclusion rests on a single dataset. This runs the LoRA tier on
Messidor-2 so the claim "adapters recover most of the backbone gap" can be
checked on a second, harder cohort.

Protocol matches `run_lora_ordinal.py` exactly -- same rank-8 adapters on
attn.qkv, same two-group optimiser and learning rates, same OrdinalMarginLoss,
same early stopping -- so the only differences are the dataset and its 5-fold
split. Messidor-2 has no official train/test partition, so we use stratified
5-fold CV over the 1,744 gradable images, matching the frozen Messidor-2 rows.

Reads images from a pre-resized cache (see the note in build_image_cache.py):
Messidor-2 ships mixed .png/.JPG extensions, so the cache normalises every
file to a `<stem>.png` and the dataset addresses images by stem.

Usage:
    uv run python scripts/run_lora_messidor2.py --seeds 42 \
        --image-dir data/messidor2/images_256
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.image_dataset import FundusImageDataset
from src.evaluate import compute_all_metrics
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.retfound_lora_capsnet import RetfoundLoraOrdinalCapsNet
from src.utils import enable_tf32

CSV = Path("data/messidor2/messidor_data.csv")
NUM_CLASSES, N_FOLDS = 5, 5
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
LR_LORA, LR_HEAD, WEIGHT_DECAY = 1e-4, 1e-3, 1e-4


def load_split() -> tuple[np.ndarray, np.ndarray]:
    ids, labels = [], []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["adjudicated_gradable"] != "1":
                continue
            g = r["adjudicated_dr_grade"]
            if g in ("", "NA"):
                continue
            ids.append(Path(r["image_id"]).stem)
            labels.append(int(g))
    return np.array(ids), np.array(labels, dtype=np.int64)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    L, Y, tot, n = [], [], 0.0, 0
    for X, y in loader:
        X = X.to(device)
        out = model(X)["head_lengths"]
        tot += loss_fn(out, y.to(device)).item() * X.size(0)
        n += X.size(0)
        L.append(out.cpu())
        Y.append(y)
    lengths = torch.cat(L)
    labels = torch.cat(Y).numpy().astype(np.int64)
    preds = predict_grade_from_heads(lengths).numpy().astype(np.int64)
    m = compute_all_metrics(labels, preds, num_classes=NUM_CLASSES)
    m["loss"] = tot / n
    m["_y_true"], m["_y_pred"] = labels, preds
    m["_head_lengths"] = lengths.numpy().astype(np.float32)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--image-dir", default="data/messidor2/images_256")
    ap.add_argument("--out-dir", default="results/lora_messidor2")
    ap.add_argument("--tune-mode", choices=["lora", "full", "progressive"], default="lora")
    ap.add_argument("--unfreeze-blocks", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        enable_tf32()
    img_dir = Path(args.image_dir)
    ids, y = load_split()
    print(f"Device: {device} | images: {img_dir} | N={len(ids)} "
          f"| dist={np.bincount(y).tolist()}")

    for seed in [int(s) for s in args.seeds.split(",")]:
        out = Path(args.out_dir) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        per_fold = []
        for k, (tr, va) in enumerate(skf.split(ids, y), 1):
            torch.manual_seed(seed + k)
            np.random.seed(seed + k)
            dl = lambda idx, sh: DataLoader(
                FundusImageDataset(ids[idx], y[idx], img_dir, image_ext=".png"),
                batch_size=args.batch_size, shuffle=sh,
                num_workers=args.num_workers, pin_memory=False)
            tr_loader, va_loader = dl(tr, True), dl(va, False)

            model = RetfoundLoraOrdinalCapsNet(
                lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT, num_classes=NUM_CLASSES).to(device)
            rep = model.set_tuning_mode(args.tune_mode, args.unfreeze_blocks)
            print(f"\n=== seed {seed} fold {k}/{N_FOLDS} "
                  f"(train={len(tr)}, val={len(va)})  trainable {rep['trainable']:,} ===")

            groups = {"lora": [], "head": []}
            for n_, p_ in model.named_parameters():
                if p_.requires_grad:
                    groups["head" if n_.startswith("head.") else "lora"].append(p_)
            opt = torch.optim.AdamW([
                {"params": groups["lora"], "lr": LR_LORA, "weight_decay": WEIGHT_DECAY},
                {"params": groups["head"], "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY}])
            loss_fn = OrdinalMarginLoss(num_classes=NUM_CLASSES)

            best, best_state, bad = -1.0, None, 0
            t0 = time.time()
            for ep in range(1, args.epochs + 1):
                model.train()
                for X, yy in tr_loader:
                    opt.zero_grad()
                    loss_fn(model(X.to(device))["head_lengths"], yy.to(device)).backward()
                    opt.step()
                m = evaluate(model, va_loader, loss_fn, device)
                if m["qwk"] > best:
                    best, bad = m["qwk"], 0
                    trained = {n_ for n_, p_ in model.named_parameters() if p_.requires_grad}
                    best_state = {kk: v.detach().cpu().clone()
                                  for kk, v in model.state_dict().items() if kk in trained}
                else:
                    bad += 1
                print(f"    ep {ep:2d}/{args.epochs}  val_qwk={m['qwk']:.4f} "
                      f"acc={m['accuracy']:.4f}" + ("  *best*" if bad == 0 else ""))
                if bad >= args.patience:
                    break
            if best_state:
                model.load_state_dict(best_state, strict=False)
            m = evaluate(model, va_loader, loss_fn, device)
            np.savez_compressed(out / f"preds_fold{k}.npz",
                                y_true=m["_y_true"], y_pred=m["_y_pred"],
                                head_lengths=m["_head_lengths"])
            per_fold.append({"fold": k, "qwk": m["qwk"], "accuracy": m["accuracy"],
                             "macro_f1": m["macro_f1"], "mae": m["mae"],
                             "elapsed_sec": time.time() - t0})
            print(f"  fold {k} DONE  QWK={m['qwk']:.4f}  ({time.time() - t0:.0f}s)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        agg = {k_: float(np.mean([f[k_] for f in per_fold]))
               for k_ in ("qwk", "accuracy", "macro_f1", "mae")}
        agg |= {k_ + "_std": float(np.std([f[k_] for f in per_fold]))
                for k_ in ("qwk", "accuracy", "macro_f1", "mae")}
        (out / "summary.json").write_text(json.dumps(
            {"seed": seed, "tune_mode": args.tune_mode,
             "per_fold": per_fold, **agg}, indent=2))
        print(f"\nseed {seed}: QWK {agg['qwk']:.4f} +/- {agg['qwk_std']:.4f}  "
              f"Acc {agg['accuracy']:.4f}  MAE {agg['mae']:.3f}")


if __name__ == "__main__":
    main()
