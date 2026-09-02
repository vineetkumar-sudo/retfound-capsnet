"""Day 1: Train three baselines on cached RETFound features.

5-fold stratified CV, W&B tracking, local metric plots saved to results/baselines/.

Usage: uv run python scripts/run_baselines.py [--seeds 42,123,456] [--no-wandb]

Multi-seed runs write per-seed outputs to `results/baselines/seed{S}/`;
single-seed (default or `--seeds 42`) preserves the legacy flat layout at
`results/baselines/` for backward compatibility with prior aggregator runs.
Fold splits and holdout are always governed by `cfg.data.seed` — only model
init RNG varies across seeds (matching scripts/run_ordinal_capsnet.py).
"""

import argparse
import sys

sys.path.insert(0, ".")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import wandb
import yaml
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.models.baselines import LinearProbe, MLPClassifier, MLPRegressor
from src.utils import enable_tf32

# ---------------------------------------------------------------------------
# Config & data
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/baselines.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        enable_tf32()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(
    X_train: np.ndarray, X_val: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray,
    batch_size: int = 128,
    y_float: bool = False,
) -> tuple[DataLoader, DataLoader]:
    yt = torch.from_numpy(y_train.astype(np.float32)) if y_float else torch.from_numpy(y_train).long()
    yv = torch.from_numpy(y_val.astype(np.float32)) if y_float else torch.from_numpy(y_val).long()
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), yt)
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), yv)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_with_loss(model, loader, device, criterion, is_regression: bool) -> dict:
    """Single pass returning both metrics AND mean loss, for any classifier or regressor."""
    model.eval()
    raw_outputs, labels, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_sum += criterion(out, y).item() * X.size(0)
            n += X.size(0)
            raw_outputs.append(out.cpu().numpy())
            labels.append(y.cpu().numpy())

    raw = np.concatenate(raw_outputs, axis=0)
    lbls = np.concatenate(labels, axis=0).astype(int)
    if is_regression:
        preds = np.clip(np.round(raw), 0, 4).astype(int)
    else:
        preds = raw.argmax(axis=1)

    return {
        "qwk": cohen_kappa_score(lbls, preds, weights="quadratic"),
        "accuracy": accuracy_score(lbls, preds),
        "macro_f1": f1_score(lbls, preds, average="macro"),
        "mae": float(np.abs(lbls - preds).mean()),
        "confusion_matrix": confusion_matrix(lbls, preds, labels=range(5)),
        "loss": loss_sum / n,
        "_y_true": lbls,
        "_y_pred": preds,
    }


# ---------------------------------------------------------------------------
# Single-fold training (returns per-epoch history)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so they sum to num_classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts)
    return torch.from_numpy(weights).float()


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
    weight_decay: float = 1e-4,
    is_regression: bool = False,
    class_weights: torch.Tensor | None = None,
) -> tuple[dict, list[dict], dict]:
    """Train one fold. Returns (best_val_metrics, epoch_history, best_state_dict).

    `epoch_history` records BOTH train and val metrics per epoch so we can
    detect overfitting. `best_state_dict` is the model snapshot at the epoch
    with the highest val QWK, loaded back before return for holdout eval.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    if is_regression:
        criterion = nn.MSELoss()
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_qwk, best_metrics, best_epoch = -1.0, {}, 0
    best_state: dict | None = None
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        # --- train pass (gradient update) ---
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # --- train-set eval (metrics on post-update model, no grad) ---
        train_m = eval_with_loss(model, train_loader, device, criterion, is_regression)
        val_m = eval_with_loss(model, val_loader, device, criterion, is_regression)

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_qwk": train_m["qwk"],
            "train_accuracy": train_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_m["loss"],
            "qwk": val_m["qwk"],  # kept as unprefixed "qwk" for plotting back-compat
            "accuracy": val_m["accuracy"],
            "macro_f1": val_m["macro_f1"],
        })

        if val_m["qwk"] > best_qwk:
            best_qwk = val_m["qwk"]
            best_metrics = val_m
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    best_metrics["best_epoch"] = best_epoch
    best_metrics["final_epoch"] = history[-1]["epoch"]

    # Restore best snapshot so the caller can eval it on the holdout set
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_metrics, history, best_state or {}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fold_histories(
    all_histories: dict[str, list[list[dict]]],
    plots_dir: Path,
):
    """Plot per-fold and mean training curves for each baseline.

    all_histories: {model_name: [fold0_history, fold1_history, ...]}
    Each fold_history is a list of dicts with keys: epoch, train_loss, val_loss, qwk, accuracy, macro_f1.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Each entry: (train_key, val_key, plot_filename_stem, ylabel)
    metrics_to_plot = [
        ("train_loss", "val_loss", "loss", "Loss"),
        ("train_qwk", "qwk", "qwk", "QWK"),
        ("train_accuracy", "accuracy", "accuracy", "Accuracy"),
        ("train_macro_f1", "macro_f1", "macro_f1", "Macro F1"),
    ]

    # --- Per-model curve plots (train vs val on same axes) ---
    for model_name, fold_histories in all_histories.items():
        safe_name = model_name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")

        for train_key, val_key, fname_stem, ylabel in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(8, 5))

            max_len = 0
            for fold_idx, hist in enumerate(fold_histories):
                epochs = [h["epoch"] for h in hist]
                ax.plot(epochs, [h[train_key] for h in hist], alpha=0.3, color="C0", linewidth=0.8,
                        label="train (folds)" if fold_idx == 0 else None)
                ax.plot(epochs, [h[val_key] for h in hist], alpha=0.3, color="C1", linewidth=0.8,
                        label="val (folds)" if fold_idx == 0 else None)
                max_len = max(max_len, len(hist))

            mean_train = [np.mean([h[e][train_key] for h in fold_histories if e < len(h)])
                          for e in range(max_len)]
            mean_val = [np.mean([h[e][val_key] for h in fold_histories if e < len(h)])
                        for e in range(max_len)]
            ax.plot(range(1, len(mean_train) + 1), mean_train, color="C0", linewidth=2, label="train (mean)")
            ax.plot(range(1, len(mean_val) + 1), mean_val, color="C1", linewidth=2, label="val (mean)")

            # Mean gap annotation
            if mean_train and mean_val:
                final_gap = mean_train[-1] - mean_val[-1]
                ax.text(0.02, 0.02, f"final train-val gap: {final_gap:+.4f}",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model_name} — {ylabel} (train vs val)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            fig.savefig(plots_dir / f"{safe_name}_{fname_stem}.png", dpi=150)
            plt.close(fig)

    # --- QWK comparison across all models (mean + std band) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    for idx, (model_name, fold_histories) in enumerate(all_histories.items()):
        max_len = max(len(h) for h in fold_histories)
        qwk_matrix = np.full((len(fold_histories), max_len), np.nan)
        for fi, hist in enumerate(fold_histories):
            for ei, h in enumerate(hist):
                qwk_matrix[fi, ei] = h["qwk"]

        mean_qwk = np.nanmean(qwk_matrix, axis=0)
        std_qwk = np.nanstd(qwk_matrix, axis=0)
        epochs = np.arange(1, max_len + 1)

        ax.plot(epochs, mean_qwk, color=colors[idx % len(colors)], linewidth=2, label=model_name)
        ax.fill_between(epochs, mean_qwk - std_qwk, mean_qwk + std_qwk,
                         color=colors[idx % len(colors)], alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("QWK")
    ax.set_title("Validation QWK — All Baselines (mean +/- std across folds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_qwk.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


def plot_confusion_matrices(
    all_cms: dict[str, np.ndarray],
    plots_dir: Path,
):
    """Plot averaged confusion matrices for each baseline."""
    class_names = [f"Grade {i}" for i in range(5)]
    n_models = len(all_cms)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, all_cms.items()):
        # Normalize to percentages
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\n(% per row)")

    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_summary_bars(results: dict[str, dict], plots_dir: Path):
    """Bar chart comparing mean +/- std for each metric across models."""
    metric_names = ["qwk", "accuracy", "macro_f1"]
    labels = list(results.keys())
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metric_names):
        means = [results[m][f"{metric}_mean"] for m in labels]
        stds = [results[m][f"{metric}_std"] for m in labels]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=[f"C{i}" for i in range(len(labels))],
                      edgecolor="black", linewidth=0.5)
        # Value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.replace("_", " ").upper())
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Day 1 Baselines — 5-Fold CV Results (mean +/- std)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_bars.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(bcfg: dict, feature_dim: int, num_classes: int) -> nn.Module:
    t = bcfg["type"]
    if t == "linear":
        return LinearProbe(feature_dim, num_classes)
    elif t == "mlp":
        return MLPClassifier(feature_dim, bcfg["hidden_dims"], num_classes, bcfg["dropout"])
    elif t == "mlp_regression":
        return MLPRegressor(feature_dim, bcfg["hidden_dims"], bcfg["dropout"])
    else:
        raise ValueError(f"Unknown model type: {t}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=None,
                   help="Comma-separated model-init seeds. If set, each seed's outputs "
                        "go to plots_dir/seed{S}/. If unset, uses cfg.data.seed "
                        "(backwards-compatible flat layout).")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return p.parse_args()


def parse_seeds(arg: str | None, default_seed: int) -> list[int]:
    if arg is None:
        return [default_seed]
    return [int(s) for s in arg.split(",") if s.strip()]


def run_one_seed(
    cfg: dict,
    device: torch.device,
    data_seed: int,
    model_seed: int,
    n_folds: int,
    holdout_frac: float,
    patience: int,
    batch_size: int,
    weight_decay: float,
    feature_dim: int,
    num_classes: int,
    features: np.ndarray,
    labels: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    plots_dir: Path,
    use_wandb: bool,
    seed_suffix: str,
) -> dict:
    """Run all baselines for one model-init seed. Writes per-seed outputs under `plots_dir`.

    Data splits (train_test_split holdout + StratifiedKFold folds) are governed by
    `data_seed` and are identical across seeds; only the model init RNG varies.
    Returns the per-seed `all_results` dict.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=data_seed)

    all_histories: dict[str, list[list[dict]]] = {}
    all_results: dict[str, dict] = {}
    all_cms: dict[str, np.ndarray] = {}

    baseline_keys = list(cfg["baselines"].keys())

    for bkey in baseline_keys:
        bcfg = cfg["baselines"][bkey]
        name = bcfg["name"]
        is_reg = bcfg["type"] == "mlp_regression"
        use_weighted_ce = bcfg.get("loss") == "weighted_cross_entropy"

        print(f"{'=' * 60}")
        print(f"{name} — {n_folds}-fold CV + holdout eval")
        print(f"{'=' * 60}")

        run = None
        if use_wandb:
            run = wandb.init(
                project=cfg["wandb"]["project"],
                group=cfg["wandb"]["group"],
                name=f"{name}{seed_suffix}",
                config={
                    "model": name,
                    **{k: v for k, v in bcfg.items() if k != "name"},
                    "n_folds": n_folds, "data_seed": data_seed, "model_seed": model_seed,
                    "patience": patience,
                    "weight_decay": weight_decay, "holdout_frac": holdout_frac,
                },
                reinit="finish_previous",
            )

        fold_metrics = []
        fold_histories = []
        fold_holdout_metrics: list[dict] = []
        sum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Holdout loader (same for every fold)
        _, holdout_loader = make_loaders(
            X_holdout[:1], X_holdout, y_holdout[:1], y_holdout,
            batch_size=batch_size, y_float=is_reg,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            torch.manual_seed(model_seed + fold_idx)
            np.random.seed(model_seed + fold_idx)

            train_loader, val_loader = make_loaders(
                X_train, X_val, y_train, y_val,
                batch_size=batch_size, y_float=is_reg,
            )
            model = build_model(bcfg, feature_dim, num_classes)

            cw = compute_class_weights(y_train, num_classes) if use_weighted_ce else None

            metrics, history, _ = train_one_fold(
                model, train_loader, val_loader, device,
                lr=bcfg["lr"], max_epochs=bcfg["epochs"],
                patience=patience, weight_decay=weight_decay,
                is_regression=is_reg, class_weights=cw,
            )
            # Model is loaded with best-val-QWK state; eval on frozen holdout
            if is_reg:
                criterion_h = nn.MSELoss()
            elif cw is not None:
                criterion_h = nn.CrossEntropyLoss(weight=cw.to(device))
            else:
                criterion_h = nn.CrossEntropyLoss()
            holdout_m = eval_with_loss(model, holdout_loader, device, criterion_h, is_reg)

            # Save per-fold preds for Day 6 MAE + confusion matrices
            safe_name = name.lower().replace(" ", "_").replace("+", "_").replace("__", "_")
            np.savez_compressed(
                plots_dir / f"preds_{safe_name}_fold{fold_idx + 1}.npz",
                y_true=metrics["_y_true"], y_pred=metrics["_y_pred"],
                holdout_y_true=holdout_m["_y_true"], holdout_y_pred=holdout_m["_y_pred"],
            )
            # Strip prediction arrays before storing metrics dict
            for d in (metrics, holdout_m):
                d.pop("_y_true", None)
                d.pop("_y_pred", None)

            fold_metrics.append(metrics)
            fold_histories.append(history)
            fold_holdout_metrics.append(holdout_m)
            sum_cm += metrics["confusion_matrix"]

            print(f"  Fold {fold_idx + 1}/{n_folds}: "
                  f"val QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}  "
                  f"|  holdout QWK={holdout_m['qwk']:.4f} Acc={holdout_m['accuracy']:.4f} F1={holdout_m['macro_f1']:.4f}  "
                  f"(best@{metrics['best_epoch']}, stopped@{metrics['final_epoch']})")

            if use_wandb:
                wandb.log({
                    f"fold_{fold_idx+1}/val_qwk": metrics["qwk"],
                    f"fold_{fold_idx+1}/val_accuracy": metrics["accuracy"],
                    f"fold_{fold_idx+1}/val_macro_f1": metrics["macro_f1"],
                    f"fold_{fold_idx+1}/holdout_qwk": holdout_m["qwk"],
                    f"fold_{fold_idx+1}/holdout_accuracy": holdout_m["accuracy"],
                    f"fold_{fold_idx+1}/holdout_macro_f1": holdout_m["macro_f1"],
                    f"fold_{fold_idx+1}/best_epoch": metrics["best_epoch"],
                })

        qwks = [m["qwk"] for m in fold_metrics]
        accs = [m["accuracy"] for m in fold_metrics]
        f1s = [m["macro_f1"] for m in fold_metrics]
        maes = [m["mae"] for m in fold_metrics]
        h_qwks = [m["qwk"] for m in fold_holdout_metrics]
        h_accs = [m["accuracy"] for m in fold_holdout_metrics]
        h_f1s = [m["macro_f1"] for m in fold_holdout_metrics]
        h_maes = [m["mae"] for m in fold_holdout_metrics]

        # Also capture train-val gap at best epoch (overfitting diagnostic)
        gaps_qwk = []
        for hist, m in zip(fold_histories, fold_metrics):
            best_row = hist[m["best_epoch"] - 1]
            gaps_qwk.append(best_row["train_qwk"] - best_row["qwk"])

        result = {
            "qwk_mean": float(np.mean(qwks)), "qwk_std": float(np.std(qwks)),
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "holdout_qwk_mean": float(np.mean(h_qwks)), "holdout_qwk_std": float(np.std(h_qwks)),
            "holdout_accuracy_mean": float(np.mean(h_accs)), "holdout_accuracy_std": float(np.std(h_accs)),
            "holdout_macro_f1_mean": float(np.mean(h_f1s)), "holdout_macro_f1_std": float(np.std(h_f1s)),
            "holdout_mae_mean": float(np.mean(h_maes)), "holdout_mae_std": float(np.std(h_maes)),
            "train_val_qwk_gap_mean": float(np.mean(gaps_qwk)),
            "per_fold_qwk": qwks,
            "per_fold_holdout_qwk": h_qwks,
        }
        all_results[name] = result
        all_histories[name] = fold_histories
        all_cms[name] = sum_cm

        print(f"  ---")
        print(f"  Val:     QWK={result['qwk_mean']:.4f}+/-{result['qwk_std']:.4f}  "
              f"Acc={result['accuracy_mean']:.4f}+/-{result['accuracy_std']:.4f}  "
              f"F1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}")
        print(f"  Holdout: QWK={result['holdout_qwk_mean']:.4f}+/-{result['holdout_qwk_std']:.4f}  "
              f"Acc={result['holdout_accuracy_mean']:.4f}+/-{result['holdout_accuracy_std']:.4f}  "
              f"F1={result['holdout_macro_f1_mean']:.4f}+/-{result['holdout_macro_f1_std']:.4f}")
        print(f"  Train-Val QWK gap (at best epoch, mean): {result['train_val_qwk_gap_mean']:+.4f}")
        print()

        if use_wandb and run is not None:
            wandb.log({
                "mean/val_qwk": result["qwk_mean"],
                "mean/val_accuracy": result["accuracy_mean"],
                "mean/val_macro_f1": result["macro_f1_mean"],
                "mean/holdout_qwk": result["holdout_qwk_mean"],
                "mean/holdout_accuracy": result["holdout_accuracy_mean"],
                "mean/holdout_macro_f1": result["holdout_macro_f1_mean"],
                "mean/train_val_qwk_gap": result["train_val_qwk_gap_mean"],
            })
            wandb.summary.update({
                "qwk_mean": result["qwk_mean"], "qwk_std": result["qwk_std"],
                "accuracy_mean": result["accuracy_mean"], "accuracy_std": result["accuracy_std"],
                "macro_f1_mean": result["macro_f1_mean"], "macro_f1_std": result["macro_f1_std"],
                "holdout_qwk_mean": result["holdout_qwk_mean"],
                "holdout_accuracy_mean": result["holdout_accuracy_mean"],
                "holdout_macro_f1_mean": result["holdout_macro_f1_mean"],
                "train_val_qwk_gap_mean": result["train_val_qwk_gap_mean"],
            })
            run.finish()

    # -----------------------------------------------------------------------
    # Generate local plots (per seed)
    # -----------------------------------------------------------------------
    print("Generating plots...")
    plot_fold_histories(all_histories, plots_dir)
    plot_confusion_matrices(all_cms, plots_dir)
    plot_summary_bars(all_results, plots_dir)

    # -----------------------------------------------------------------------
    # Summary table (per seed)
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print(f"DAY 1 RESULTS — RETFound Feature Baselines (5-Fold CV + Holdout){seed_suffix}")
    print("=" * 100)
    print(f"{'Model':<27s} {'Val QWK':>14s} {'Holdout QWK':>14s} {'Val Acc':>14s} {'Holdout Acc':>14s} {'T-V gap':>10s}")
    print("-" * 100)
    for name, r in all_results.items():
        print(f"{name:<27s} "
              f"{r['qwk_mean']:.4f}+/-{r['qwk_std']:.4f} "
              f"{r['holdout_qwk_mean']:.4f}+/-{r['holdout_qwk_std']:.4f} "
              f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f} "
              f"{r['holdout_accuracy_mean']:.4f}+/-{r['holdout_accuracy_std']:.4f} "
              f"{r['train_val_qwk_gap_mean']:+.4f}")
    print("-" * 100)
    print("T-V gap = train_qwk - val_qwk at best epoch (mean across folds). "
          "Big positive = overfitting.")
    print()

    # Save per-seed summary JSON
    import json
    (plots_dir / "summary.json").write_text(json.dumps({
        "config": {
            "n_folds": n_folds, "data_seed": data_seed, "model_seed": model_seed,
            "patience": patience,
            "weight_decay": weight_decay, "holdout_frac": holdout_frac,
        },
        "results": all_results,
    }, indent=2))

    # Optional W&B summary run with comparison table (per seed)
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            group=cfg["wandb"]["group"],
            name=f"day1-summary-5fold{seed_suffix}",
            reinit="finish_previous",
        )
        summary_table = wandb.Table(
            columns=["Model", "QWK (mean)", "QWK (std)", "Accuracy (mean)", "Accuracy (std)",
                     "Macro F1 (mean)", "Macro F1 (std)"],
            data=[
                [name, r["qwk_mean"], r["qwk_std"], r["accuracy_mean"], r["accuracy_std"],
                 r["macro_f1_mean"], r["macro_f1_std"]]
                for name, r in all_results.items()
            ],
        )
        wandb.log({"day1_5fold_results": summary_table})
        for img_path in sorted(plots_dir.glob("*.png")):
            wandb.log({img_path.stem: wandb.Image(str(img_path))})
        run.finish()

    print(f"\nPlots saved to {plots_dir}/")
    return all_results


def main():
    args = parse_args()
    cfg = load_config()
    device = get_device()
    data_seed = cfg["data"]["seed"]
    n_folds = cfg["data"]["n_folds"]
    holdout_frac = cfg["data"].get("holdout_split", 0.10)
    patience = cfg["training"]["early_stopping_patience"]
    batch_size = cfg["training"]["batch_size"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    feature_dim = cfg["data"]["feature_dim"]
    num_classes = cfg["data"]["num_classes"]
    base_plots_dir = Path(cfg["output"]["plots_dir"])
    base_plots_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (not args.no_wandb) and cfg.get("wandb", {}).get("enabled", True)
    model_seeds = parse_seeds(args.seeds, data_seed)

    print(f"Device: {device}")
    print(f"K-Fold CV: {n_folds} folds, data_seed={data_seed}, "
          f"patience={patience}, weight_decay={weight_decay}")
    print(f"Holdout: {holdout_frac * 100:.0f}% frozen, never touched by CV")
    print(f"Model seeds: {model_seeds}\n")

    # Load data once (splits are identical across seeds via data_seed)
    features_all = np.load(cfg["data"]["features_path"])
    labels_all = np.load(cfg["data"]["labels_path"])
    print(f"Dataset: {features_all.shape[0]} samples, {features_all.shape[1]}-dim features")
    print(f"Full label distribution: {np.bincount(labels_all)}")

    from sklearn.model_selection import train_test_split
    pool_idx, holdout_idx = train_test_split(
        np.arange(len(features_all)),
        test_size=holdout_frac, stratify=labels_all, random_state=data_seed,
    )
    features = features_all[pool_idx]
    labels = labels_all[pool_idx]
    X_holdout = features_all[holdout_idx]
    y_holdout = labels_all[holdout_idx]
    print(f"CV pool: {len(features)} samples, dist={np.bincount(labels)}")
    print(f"Holdout: {len(X_holdout)} samples, dist={np.bincount(y_holdout)}\n")

    # Default single-seed with no --seeds flag writes to the flat legacy layout
    # to keep prior aggregator runs comparable. Multi-seed or explicit --seeds
    # always uses per-seed subdirs.
    is_default_single = (args.seeds is None and len(model_seeds) == 1
                         and model_seeds[0] == data_seed)

    per_seed_results: dict[int, dict] = {}
    for model_seed in model_seeds:
        if is_default_single:
            seed_plots_dir = base_plots_dir
            seed_suffix = ""
        else:
            seed_plots_dir = base_plots_dir / f"seed{model_seed}"
            seed_suffix = f" (seed={model_seed})"
        print("\n" + "#" * 100)
        print(f"### MODEL SEED {model_seed}   ->   {seed_plots_dir}")
        print("#" * 100 + "\n")
        per_seed_results[model_seed] = run_one_seed(
            cfg=cfg, device=device,
            data_seed=data_seed, model_seed=model_seed,
            n_folds=n_folds, holdout_frac=holdout_frac,
            patience=patience, batch_size=batch_size,
            weight_decay=weight_decay,
            feature_dim=feature_dim, num_classes=num_classes,
            features=features, labels=labels,
            X_holdout=X_holdout, y_holdout=y_holdout,
            plots_dir=seed_plots_dir,
            use_wandb=use_wandb,
            seed_suffix=seed_suffix,
        )

    # Pooled across-seeds summary at the base dir (only when multi-seed)
    if len(model_seeds) > 1:
        import json
        model_names = list(next(iter(per_seed_results.values())).keys())
        pooled = {}
        for name in model_names:
            per_seed_qwk = [per_seed_results[s][name]["qwk_mean"] for s in model_seeds]
            per_seed_acc = [per_seed_results[s][name]["accuracy_mean"] for s in model_seeds]
            per_seed_f1 = [per_seed_results[s][name]["macro_f1_mean"] for s in model_seeds]
            per_seed_mae = [per_seed_results[s][name]["mae_mean"] for s in model_seeds]
            pooled[name] = {
                "qwk_mean_across_seeds": float(np.mean(per_seed_qwk)),
                "qwk_std_across_seeds": float(np.std(per_seed_qwk)),
                "accuracy_mean_across_seeds": float(np.mean(per_seed_acc)),
                "accuracy_std_across_seeds": float(np.std(per_seed_acc)),
                "macro_f1_mean_across_seeds": float(np.mean(per_seed_f1)),
                "macro_f1_std_across_seeds": float(np.std(per_seed_f1)),
                "mae_mean_across_seeds": float(np.mean(per_seed_mae)),
                "mae_std_across_seeds": float(np.std(per_seed_mae)),
                "per_seed_qwk": per_seed_qwk,
            }
        (base_plots_dir / "pooled_across_seeds.json").write_text(json.dumps({
            "model_seeds": model_seeds, "data_seed": data_seed,
            "pooled": pooled,
        }, indent=2))
        print("\n" + "=" * 100)
        print(f"POOLED ACROSS {len(model_seeds)} SEEDS")
        print("=" * 100)
        for name, p in pooled.items():
            print(f"{name:<27s} "
                  f"QWK={p['qwk_mean_across_seeds']:.4f}+/-{p['qwk_std_across_seeds']:.4f}  "
                  f"Acc={p['accuracy_mean_across_seeds']:.4f}+/-{p['accuracy_std_across_seeds']:.4f}  "
                  f"F1={p['macro_f1_mean_across_seeds']:.4f}+/-{p['macro_f1_std_across_seeds']:.4f}")
        print(f"Pooled JSON -> {base_plots_dir}/pooled_across_seeds.json")


if __name__ == "__main__":
    main()
