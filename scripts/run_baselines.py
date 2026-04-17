"""Day 1: Train three baselines on cached RETFound features.

5-fold stratified CV, W&B tracking, local metric plots saved to results/baselines/.

Usage: uv run python scripts/run_baselines.py
"""

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

# ---------------------------------------------------------------------------
# Config & data
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/baselines.yaml") -> dict:
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

def eval_classification(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    p, l = np.array(preds), np.array(labels)
    return {
        "qwk": cohen_kappa_score(l, p, weights="quadratic"),
        "accuracy": accuracy_score(l, p),
        "macro_f1": f1_score(l, p, average="macro"),
        "confusion_matrix": confusion_matrix(l, p, labels=range(5)),
    }


def eval_regression(model, loader, device) -> dict:
    model.eval()
    raw, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            raw.extend(model(X.to(device)).cpu().numpy())
            labels.extend(y.numpy())
    raw = np.array(raw)
    l = np.array(labels).astype(int)
    p = np.clip(np.round(raw), 0, 4).astype(int)
    return {
        "qwk": cohen_kappa_score(l, p, weights="quadratic"),
        "accuracy": accuracy_score(l, p),
        "macro_f1": f1_score(l, p, average="macro"),
        "confusion_matrix": confusion_matrix(l, p, labels=range(5)),
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
    is_regression: bool = False,
    class_weights: torch.Tensor | None = None,
) -> tuple[dict, list[dict]]:
    """Train one fold. Returns (best_metrics, epoch_history)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    if is_regression:
        criterion = nn.MSELoss()
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    eval_fn = eval_regression if is_regression else eval_classification

    best_qwk, best_metrics, best_epoch = -1.0, {}, 0
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        # --- train ---
        model.train()
        total_loss, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            n += X.size(0)
        train_loss = total_loss / n

        # --- val ---
        metrics = eval_fn(model, val_loader, device)
        model.eval()
        val_loss_sum, vn = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss_sum += criterion(model(X), y).item() * X.size(0)
                vn += X.size(0)
        val_loss = val_loss_sum / vn

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
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

        if no_improve >= patience:
            break

    best_metrics["best_epoch"] = best_epoch
    best_metrics["final_epoch"] = history[-1]["epoch"]
    return best_metrics, history


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
    metrics_to_plot = [
        ("train_loss", "val_loss", "Loss"),
        ("qwk", None, "QWK"),
        ("accuracy", None, "Accuracy"),
        ("macro_f1", None, "Macro F1"),
    ]

    # --- Per-model curve plots ---
    for model_name, fold_histories in all_histories.items():
        safe_name = model_name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")

        for train_key, val_key, ylabel in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot each fold as a thin line
            max_len = 0
            for fold_idx, hist in enumerate(fold_histories):
                epochs = [h["epoch"] for h in hist]
                vals = [h[train_key] for h in hist]
                ax.plot(epochs, vals, alpha=0.3, color="C0", linewidth=0.8,
                        label="train (folds)" if fold_idx == 0 else None)
                if val_key:
                    val_vals = [h[val_key] for h in hist]
                    ax.plot(epochs, val_vals, alpha=0.3, color="C1", linewidth=0.8,
                            label="val (folds)" if fold_idx == 0 else None)
                max_len = max(max_len, len(hist))

            # Compute and plot mean curve
            mean_vals = []
            mean_val_vals = []
            for e in range(max_len):
                vs = [h[e][train_key] for h in fold_histories if e < len(h)]
                mean_vals.append(np.mean(vs))
                if val_key:
                    vvs = [h[e][val_key] for h in fold_histories if e < len(h)]
                    mean_val_vals.append(np.mean(vvs))

            ax.plot(range(1, len(mean_vals) + 1), mean_vals, color="C0", linewidth=2, label="train (mean)")
            if val_key:
                ax.plot(range(1, len(mean_val_vals) + 1), mean_val_vals, color="C1", linewidth=2, label="val (mean)")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model_name} — {ylabel}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            fname = f"{safe_name}_{train_key}.png"
            fig.savefig(plots_dir / fname, dpi=150)
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


def main():
    cfg = load_config()
    device = get_device()
    seed = cfg["data"]["seed"]
    n_folds = cfg["data"]["n_folds"]
    patience = cfg["training"]["early_stopping_patience"]
    batch_size = cfg["training"]["batch_size"]
    feature_dim = cfg["data"]["feature_dim"]
    num_classes = cfg["data"]["num_classes"]
    plots_dir = Path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"K-Fold CV: {n_folds} folds, seed={seed}, patience={patience}\n")

    # Load data
    features = np.load(cfg["data"]["features_path"])
    labels = np.load(cfg["data"]["labels_path"])
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]}-dim features")
    print(f"Label distribution: {np.bincount(labels)}\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Store everything for plotting
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
        print(f"{name} — {n_folds}-fold CV")
        print(f"{'=' * 60}")

        # W&B run for this baseline (logs fold-averaged metrics)
        run = wandb.init(
            project=cfg["wandb"]["project"],
            group=cfg["wandb"]["group"],
            name=name,
            config={
                "model": name,
                **{k: v for k, v in bcfg.items() if k != "name"},
                "n_folds": n_folds,
                "seed": seed,
                "patience": patience,
            },
            reinit="finish_previous",
        )

        fold_metrics = []
        fold_histories = []
        sum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            torch.manual_seed(seed + fold_idx)
            np.random.seed(seed + fold_idx)

            train_loader, val_loader = make_loaders(
                X_train, X_val, y_train, y_val,
                batch_size=batch_size, y_float=is_reg,
            )
            model = build_model(bcfg, feature_dim, num_classes)

            # Compute class weights from this fold's training labels
            cw = compute_class_weights(y_train, num_classes) if use_weighted_ce else None

            metrics, history = train_one_fold(
                model, train_loader, val_loader, device,
                lr=bcfg["lr"], max_epochs=bcfg["epochs"],
                patience=patience, is_regression=is_reg,
                class_weights=cw,
            )
            fold_metrics.append(metrics)
            fold_histories.append(history)
            sum_cm += metrics["confusion_matrix"]

            print(f"  Fold {fold_idx + 1}/{n_folds}: "
                  f"QWK={metrics['qwk']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['macro_f1']:.4f}  "
                  f"(best@{metrics['best_epoch']}, stopped@{metrics['final_epoch']})")

            # Log per-fold to wandb
            wandb.log({
                f"fold_{fold_idx+1}/qwk": metrics["qwk"],
                f"fold_{fold_idx+1}/accuracy": metrics["accuracy"],
                f"fold_{fold_idx+1}/macro_f1": metrics["macro_f1"],
                f"fold_{fold_idx+1}/best_epoch": metrics["best_epoch"],
            })

        # Aggregate fold results
        qwks = [m["qwk"] for m in fold_metrics]
        accs = [m["accuracy"] for m in fold_metrics]
        f1s = [m["macro_f1"] for m in fold_metrics]

        result = {
            "qwk_mean": np.mean(qwks), "qwk_std": np.std(qwks),
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
            "per_fold_qwk": qwks,
        }
        all_results[name] = result
        all_histories[name] = fold_histories
        all_cms[name] = sum_cm

        print(f"  ---")
        print(f"  Mean: QWK={result['qwk_mean']:.4f}+/-{result['qwk_std']:.4f}  "
              f"Acc={result['accuracy_mean']:.4f}+/-{result['accuracy_std']:.4f}  "
              f"F1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}")
        print()

        # Log aggregates to wandb
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
        run.finish()

    # -----------------------------------------------------------------------
    # Generate local plots
    # -----------------------------------------------------------------------
    print("Generating plots...")
    plot_fold_histories(all_histories, plots_dir)
    plot_confusion_matrices(all_cms, plots_dir)
    plot_summary_bars(all_results, plots_dir)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DAY 1 RESULTS — RETFound Feature Baselines (5-Fold CV)")
    print("=" * 70)
    print(f"{'Model':<25s} {'QWK':>14s} {'Accuracy':>14s} {'Macro F1':>14s}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<25s} "
              f"{r['qwk_mean']:.4f}+/-{r['qwk_std']:.4f} "
              f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f} "
              f"{r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}")
    print("-" * 70)

    # W&B summary run with comparison table
    run = wandb.init(
        project=cfg["wandb"]["project"],
        group=cfg["wandb"]["group"],
        name="day1-summary-5fold",
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

    # Upload local plots as wandb images
    for img_path in sorted(plots_dir.glob("*.png")):
        wandb.log({img_path.stem: wandb.Image(str(img_path))})

    run.finish()

    print(f"\nPlots saved to {plots_dir}/")
    print("All results logged to W&B project: retfound-capsnet")


if __name__ == "__main__":
    main()
