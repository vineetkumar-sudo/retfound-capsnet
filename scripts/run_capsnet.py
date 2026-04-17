"""Day 2: Train a vanilla CapsNet head on cached RETFound features.

5-fold stratified CV matching Day 1. Margin loss (Sabour). Optional reconstruction
decoder. Local MPS/CPU. Clean W&B logging (or --no-wandb).

Demo:    uv run python scripts/run_capsnet.py --demo
Full:    uv run python scripts/run_capsnet.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.losses.margin_loss import MarginLoss
from src.models.capsnet import CapsNet


# ---------------------------------------------------------------------------
# Config / setup
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/capsnet.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(
    X_train: np.ndarray, X_val: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_capsnet(model: CapsNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(device))
            preds.extend(out["lengths"].argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())
    p, l = np.array(preds), np.array(labels)
    return {
        "qwk": cohen_kappa_score(l, p, weights="quadratic"),
        "accuracy": accuracy_score(l, p),
        "macro_f1": f1_score(l, p, average="macro"),
        "confusion_matrix": confusion_matrix(l, p, labels=range(5)),
    }


# ---------------------------------------------------------------------------
# Training (one fold)
# ---------------------------------------------------------------------------

def train_one_fold(
    model: CapsNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
    num_classes: int,
    reconstruction_weight: float,
    log_every: int = 10,
    verbose: bool = False,
) -> tuple[dict, list[dict]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    margin = MarginLoss(num_classes=num_classes)

    best_qwk, best_metrics, best_epoch, no_improve = -1.0, {}, 0, 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, total_margin, total_recon, n = 0.0, 0.0, 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, labels=y)
            m_loss = margin(out["lengths"], y)
            loss = m_loss
            r_loss_val = 0.0
            if "reconstruction" in out:
                r_loss = F.mse_loss(out["reconstruction"], X)
                loss = m_loss + reconstruction_weight * r_loss
                r_loss_val = r_loss.item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_margin += m_loss.item() * X.size(0)
            total_recon += r_loss_val * X.size(0)
            n += X.size(0)
        train_loss = total_loss / n
        train_margin = total_margin / n
        train_recon = total_recon / n

        # Validation
        metrics = eval_capsnet(model, val_loader, device)
        model.eval()
        val_loss_sum, vn = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X, labels=y)
                l = margin(out["lengths"], y)
                if "reconstruction" in out:
                    l = l + reconstruction_weight * F.mse_loss(out["reconstruction"], X)
                val_loss_sum += l.item() * X.size(0)
                vn += X.size(0)
        val_loss = val_loss_sum / vn

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_margin": train_margin,
            "train_recon": train_recon,
            "val_loss": val_loss,
            "qwk": metrics["qwk"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        })

        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            best_metrics = metrics
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % log_every == 0 or no_improve >= patience):
            print(f"    ep {epoch:3d}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"qwk={metrics['qwk']:.4f}  acc={metrics['accuracy']:.4f}  "
                  f"f1={metrics['macro_f1']:.4f}"
                  + (f"  *best*" if epoch == best_epoch else ""))

        if no_improve >= patience:
            if verbose:
                print(f"    early stop @ ep {epoch} (best qwk={best_qwk:.4f} @ ep {best_epoch})")
            break

    best_metrics["best_epoch"] = best_epoch
    best_metrics["final_epoch"] = history[-1]["epoch"]
    return best_metrics, history


# ---------------------------------------------------------------------------
# Sanity checks (run once on first batch)
# ---------------------------------------------------------------------------

def sanity_check_pretrain(
    model: CapsNet, X_sample: torch.Tensor, y_sample: torch.Tensor,
    num_classes: int, device: torch.device,
) -> None:
    """Pre-training sanity: shapes, NaN, gradient flow. Prints results."""
    print("  --- Pre-training sanity checks ---")
    model.to(device)
    model.train()
    X_sample = X_sample.to(device)
    y_sample = y_sample.to(device)

    out = model(X_sample, labels=y_sample)
    B = X_sample.size(0)

    assert out["digit_caps"].shape == (B, num_classes, model.digit.caps_dim), out["digit_caps"].shape
    assert out["lengths"].shape == (B, num_classes), out["lengths"].shape
    print(f"    shapes OK: digit_caps={tuple(out['digit_caps'].shape)}, "
          f"lengths={tuple(out['lengths'].shape)}, coupling={tuple(out['coupling'].shape)}")

    assert not torch.isnan(out["digit_caps"]).any(), "NaN in digit_caps"
    assert not torch.isnan(out["lengths"]).any(), "NaN in lengths"
    print("    NaN check OK")

    margin = MarginLoss(num_classes=num_classes)
    loss = margin(out["lengths"], y_sample)
    loss.backward()
    primary_grad = model.primary.linear.weight.grad
    digit_grad = model.digit.W.grad
    assert primary_grad is not None and primary_grad.abs().sum().item() > 0, "no grad in PrimaryCaps"
    assert digit_grad is not None and digit_grad.abs().sum().item() > 0, "no grad in DigitCaps"
    print(f"    gradient flow OK: "
          f"primary.grad.norm={primary_grad.norm().item():.4f}, "
          f"digit.W.grad.norm={digit_grad.norm().item():.4f}")

    # Note on collapse: untrained caps will have near-uniform lengths. That's expected.
    init_std = out["lengths"].std(dim=1).mean().item()
    print(f"    note: per-sample length std on untrained model = {init_std:.4f} "
          f"(expected ~0; watch this after training)")

    model.zero_grad(set_to_none=True)
    print("  --- Pre-training sanity OK ---\n")


def check_collapse_posttrain(model: CapsNet, val_loader: DataLoader, device: torch.device) -> float:
    """After training, lengths should differentiate across classes. Returns mean per-sample std."""
    model.eval()
    stds = []
    with torch.no_grad():
        for X, _ in val_loader:
            out = model(X.to(device))
            stds.append(out["lengths"].std(dim=1).cpu().numpy())
    return float(np.concatenate(stds).mean())


# ---------------------------------------------------------------------------
# Plotting (mirrors run_baselines.py patterns — copied not imported to keep Day 2 self-contained)
# ---------------------------------------------------------------------------

def plot_fold_histories(fold_histories: list[list[dict]], plots_dir: Path, model_name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    pairs = [
        ("train_loss", "val_loss", "Loss"),
        ("qwk", None, "QWK"),
        ("accuracy", None, "Accuracy"),
        ("macro_f1", None, "Macro F1"),
    ]
    for train_key, val_key, ylabel in pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        max_len = 0
        for fi, hist in enumerate(fold_histories):
            xs = [h["epoch"] for h in hist]
            ax.plot(xs, [h[train_key] for h in hist], alpha=0.3, color="C0",
                    linewidth=0.8, label="train (folds)" if fi == 0 else None)
            if val_key:
                ax.plot(xs, [h[val_key] for h in hist], alpha=0.3, color="C1",
                        linewidth=0.8, label="val (folds)" if fi == 0 else None)
            max_len = max(max_len, len(hist))

        mean_train = [np.mean([h[e][train_key] for h in fold_histories if e < len(h)])
                       for e in range(max_len)]
        ax.plot(range(1, len(mean_train) + 1), mean_train, color="C0", linewidth=2, label="train (mean)")
        if val_key:
            mean_val = [np.mean([h[e][val_key] for h in fold_histories if e < len(h)])
                        for e in range(max_len)]
            ax.plot(range(1, len(mean_val) + 1), mean_val, color="C1", linewidth=2, label="val (mean)")

        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {ylabel}")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(plots_dir / f"{safe}_{train_key}.png", dpi=150)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, plots_dir: Path, model_name: str) -> None:
    class_names = [f"Grade {i}" for i in range(5)]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} (% per row, pooled over folds)")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_summary(result: dict, plots_dir: Path, model_name: str) -> None:
    metrics = ["qwk", "accuracy", "macro_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, metrics):
        mean = result[f"{m}_mean"]; std = result[f"{m}_std"]
        ax.bar([model_name], [mean], yerr=[std], capsize=5, color="C2", edgecolor="black")
        ax.text(0, mean + std + 0.005, f"{mean:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel(m.upper()); ax.set_title(m.replace("_", " ").upper())
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{model_name} — 5-fold CV (mean +/- std)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_bars.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Fast sanity run: small subset, 1 fold, few epochs, no W&B")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging regardless of config")
    p.add_argument("--use-decoder", action="store_true", help="Enable reconstruction decoder (override config)")
    p.add_argument("--config", default="configs/capsnet.yaml")
    return p.parse_args()


def build_model(cfg: dict, feature_dim: int, num_classes: int, use_decoder: bool) -> CapsNet:
    mc = cfg["model"]
    return CapsNet(
        feature_dim=feature_dim,
        num_primary=mc["num_primary"],
        primary_dim=mc["primary_dim"],
        num_classes=num_classes,
        caps_dim=mc["caps_dim"],
        routing_iters=mc["routing_iters"],
        use_decoder=use_decoder,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    seed = cfg["data"]["seed"]
    feature_dim = cfg["data"]["feature_dim"]
    num_classes = cfg["data"]["num_classes"]
    patience = cfg["training"]["early_stopping_patience"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]
    epochs = cfg["training"]["epochs"]
    n_folds = cfg["data"]["n_folds"]
    use_decoder = args.use_decoder or cfg["model"].get("use_decoder", False)
    recon_w = cfg["model"].get("reconstruction_weight", 0.0005)
    use_wandb = cfg["wandb"].get("enabled", True) and not args.no_wandb
    plots_dir = Path(cfg["output"]["plots_dir"])

    # --- Demo overrides ---
    if args.demo:
        d = cfg["demo"]
        n_folds = d["n_folds"]
        epochs = d["epochs"]
        use_wandb = False
        plots_dir = plots_dir / "demo"
        print("=== DEMO MODE ===")
        print(f"Using {d['n_samples']} samples, {n_folds} fold, {epochs} epochs, no W&B\n")

    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Config: feature_dim={feature_dim}, num_classes={num_classes}, "
          f"decoder={use_decoder}, n_folds={n_folds}, epochs<={epochs}, patience={patience}\n")

    # --- Load data ---
    features = np.load(cfg["data"]["features_path"])
    labels = np.load(cfg["data"]["labels_path"])
    if args.demo:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=cfg["demo"]["n_samples"], replace=False)
        features = features[idx]
        labels = labels[idx]
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]}-d features")
    print(f"Label distribution: {np.bincount(labels, minlength=num_classes)}\n")

    # --- K-fold ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if n_folds > 1 \
        else None

    name = cfg["wandb"]["name"]
    if use_decoder and "decoder" not in name:
        name = f"{name}-decoder"

    run = None
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            group=cfg["wandb"]["group"],
            name=name,
            config={
                "model": "CapsNet (5-class)",
                "feature_dim": feature_dim,
                "num_classes": num_classes,
                "num_primary": cfg["model"]["num_primary"],
                "primary_dim": cfg["model"]["primary_dim"],
                "caps_dim": cfg["model"]["caps_dim"],
                "routing_iters": cfg["model"]["routing_iters"],
                "use_decoder": use_decoder,
                "reconstruction_weight": recon_w,
                "lr": lr, "batch_size": batch_size, "epochs": epochs,
                "patience": patience, "n_folds": n_folds, "seed": seed,
            },
            reinit="finish_previous",
        )

    fold_metrics: list[dict] = []
    fold_histories: list[list[dict]] = []
    sum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    sanity_done = False

    # Construct splits (single fold for demo; stratified k-fold otherwise)
    if skf is None:
        # Demo: 80/20 single split
        from sklearn.model_selection import train_test_split
        tr_idx, va_idx = train_test_split(
            np.arange(len(features)), test_size=0.2, stratify=labels, random_state=seed,
        )
        splits = [(tr_idx, va_idx)]
    else:
        splits = list(skf.split(features, labels))

    t_global = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_train, X_val = features[tr_idx], features[va_idx]
        y_train, y_val = labels[tr_idx], labels[va_idx]

        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, batch_size)
        model = build_model(cfg, feature_dim, num_classes, use_decoder)

        # Run sanity check once, on the very first fold's first batch
        if not sanity_done:
            Xb, yb = next(iter(train_loader))
            sanity_check_pretrain(model, Xb[:8], yb[:8], num_classes, device)
            sanity_done = True

        t0 = time.time()
        print(f"Fold {fold_idx + 1}/{len(splits)}:")
        metrics, history = train_one_fold(
            model, train_loader, val_loader, device,
            lr=lr, max_epochs=epochs, patience=patience,
            num_classes=num_classes, reconstruction_weight=recon_w,
            verbose=True, log_every=max(1, epochs // 10),
        )
        t_fold = time.time() - t0
        sec_per_epoch = t_fold / metrics["final_epoch"]

        post_std = check_collapse_posttrain(model, val_loader, device)

        fold_metrics.append(metrics)
        fold_histories.append(history)
        sum_cm += metrics["confusion_matrix"]

        print(f"  -> QWK={metrics['qwk']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['macro_f1']:.4f}  (best@{metrics['best_epoch']}, "
              f"stopped@{metrics['final_epoch']}, {sec_per_epoch:.2f}s/epoch, "
              f"post-train length std={post_std:.4f})")
        if post_std < 0.02:
            print(f"  WARN: post-training length std is low ({post_std:.4f}) — "
                  f"routing may be stuck in uniform mode\n")
        else:
            print()

        if use_wandb:
            wandb.log({
                f"fold_{fold_idx+1}/qwk": metrics["qwk"],
                f"fold_{fold_idx+1}/accuracy": metrics["accuracy"],
                f"fold_{fold_idx+1}/macro_f1": metrics["macro_f1"],
                f"fold_{fold_idx+1}/best_epoch": metrics["best_epoch"],
                f"fold_{fold_idx+1}/sec_per_epoch": sec_per_epoch,
            })

    # --- Aggregate ---
    qwks = [m["qwk"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    result = {
        "qwk_mean": float(np.mean(qwks)), "qwk_std": float(np.std(qwks)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "per_fold_qwk": qwks,
    }

    elapsed = time.time() - t_global
    avg_sec_per_epoch = elapsed / sum(m["final_epoch"] for m in fold_metrics)
    print("=" * 70)
    print(f"CapsNet (vanilla{'+decoder' if use_decoder else ''}) — "
          f"{len(splits)}-fold CV — total {elapsed:.1f}s "
          f"(avg {avg_sec_per_epoch:.2f}s/epoch)")
    print("=" * 70)

    # Project full-run time if this was a demo
    if args.demo:
        full_N = 3662  # full APTOS train set
        demo_N = features.shape[0]
        projected_per_epoch = avg_sec_per_epoch * (full_N / demo_N)
        full_cfg_epochs = cfg["training"]["epochs"]
        full_cfg_folds = cfg["data"]["n_folds"]
        projected_total = projected_per_epoch * full_cfg_epochs * full_cfg_folds
        print(f"\nProjected full run ({full_cfg_folds}-fold, {full_cfg_epochs} max epochs, "
              f"{full_N} samples):")
        print(f"  ~{projected_per_epoch:.2f}s/epoch  ->  "
              f"~{projected_total / 60:.1f} minutes worst case "
              f"(less with early stopping)")
    print(f"{'QWK':>16s}  {'Accuracy':>16s}  {'Macro F1':>16s}")
    print(f"{result['qwk_mean']:.4f}+/-{result['qwk_std']:.4f}  "
          f"{result['accuracy_mean']:.4f}+/-{result['accuracy_std']:.4f}  "
          f"{result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}")
    print("=" * 70)

    # Reference — Day 1
    print("\nDay 1 baseline reference:")
    print(f"  MLP + CE            QWK 0.8780+/-...  Acc 0.8078+/-...  F1 0.6102+/-...")
    print(f"  MLP + MSE (Ordinal) QWK 0.8855+/-...  Acc 0.7496+/-...  F1 0.5498+/-...")

    # --- Plots ---
    print(f"\nGenerating plots -> {plots_dir}/")
    plot_fold_histories(fold_histories, plots_dir, name)
    plot_confusion_matrix(sum_cm, plots_dir, name)
    plot_summary(result, plots_dir, name)

    # --- Save per-fold raw numbers ---
    import json
    (plots_dir / "summary.json").write_text(json.dumps({
        "model": name,
        "config": {
            "use_decoder": use_decoder,
            "num_primary": cfg["model"]["num_primary"],
            "primary_dim": cfg["model"]["primary_dim"],
            "caps_dim": cfg["model"]["caps_dim"],
            "routing_iters": cfg["model"]["routing_iters"],
            "lr": lr, "batch_size": batch_size,
        },
        "result": result,
        "per_fold": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in m.items() if k != "confusion_matrix"}
            for m in fold_metrics
        ],
    }, indent=2))

    if use_wandb and run is not None:
        wandb.log({
            "mean/qwk": result["qwk_mean"],
            "mean/accuracy": result["accuracy_mean"],
            "mean/macro_f1": result["macro_f1_mean"],
            "std/qwk": result["qwk_std"],
            "std/accuracy": result["accuracy_std"],
            "std/macro_f1": result["macro_f1_std"],
        })
        wandb.summary.update({
            "qwk_mean": result["qwk_mean"], "qwk_std": result["qwk_std"],
            "accuracy_mean": result["accuracy_mean"], "accuracy_std": result["accuracy_std"],
            "macro_f1_mean": result["macro_f1_mean"], "macro_f1_std": result["macro_f1_std"],
        })
        for img in sorted(plots_dir.glob("*.png")):
            wandb.log({img.stem: wandb.Image(str(img))})
        run.finish()

    print(f"\nDone. Results JSON at {plots_dir}/summary.json")


if __name__ == "__main__":
    main()
