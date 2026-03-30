"""
Galaxy Image Classification - Training Script
===============================================
Full training pipeline with early stopping, metrics,
confusion matrix plotting, and model checkpointing.
"""

from model import build_model
from dataset import download_dataset, organize_dataset, prepare_data
import config
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend


# ─── Training One Epoch ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="    Train", leave=False, ncols=80)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─── Validation ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model. Returns average loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="    Valid", leave=False, ncols=80)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ─── Plotting ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save a beautiful confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrix saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax1.set_title("Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Acc")
    ax2.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Training history saved: {save_path}")


# ─── Main Training Pipeline ─────────────────────────────────────────────────────

def train():
    """Complete training pipeline."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║    🔭  Galaxy Image Classification — Training            ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # ── Step 1: Dataset ──
    print("  📦  Step 1/4: Preparing Dataset...")
    source = download_dataset()
    if not organize_dataset(source):
        print("\n  ❌  Cannot proceed without data. Exiting.")
        return

    train_loader, val_loader, class_names, num_classes = prepare_data()

    # ── Step 2: Model ──
    print(f"\n  🧠  Step 2/4: Building Model ({num_classes} classes)...")
    model, device = build_model(num_classes=num_classes, pretrained=True)

    # ── Step 3: Training Setup ──
    print(f"\n  ⚙️  Step 3/4: Setting up training...")

    # Compute class weights from training data
    criterion = nn.CrossEntropyLoss()

    # Optimizer with differential learning rates
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config.LEARNING_RATE * 0.1},
        {"params": classifier_params, "lr": config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    print(
        f"    Optimizer:  AdamW (backbone lr={config.LEARNING_RATE * 0.1:.0e}, head lr={config.LEARNING_RATE:.0e})")
    print(f"    Scheduler:  CosineAnnealingLR")
    print(f"    Epochs:     {config.NUM_EPOCHS}")
    print(f"    Batch size: {config.BATCH_SIZE}")
    print(f"    Early stop: {config.EARLY_STOP_PATIENCE} epochs patience")

    # ── Step 4: Training Loop ──
    print(f"\n  🚀  Step 4/4: Training...")
    print("  " + "─" * 56)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[-1]["lr"]

        # Print epoch summary
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            flag = " ★ Best"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:2d}/{config.NUM_EPOCHS} │ "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} │ "
              f"Acc: {train_acc:.3f}/{val_acc:.3f} │ "
              f"LR: {lr:.1e} │ {elapsed:.0f}s{flag}")

        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(
                f"\n  ⏹️  Early stopping at epoch {epoch} (no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    total_time = time.time() - start_time
    print("  " + "─" * 56)
    print(f"  ⏱️  Total training time: {total_time / 60:.1f} minutes")
    print(f"  🏆  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.1f}%)")

    # ── Save best model ──
    if best_model_state is not None:
        checkpoint = {
            "model_state_dict": best_model_state,
            "class_names": class_names,
            "num_classes": num_classes,
            "best_val_acc": best_val_acc,
            "image_size": config.IMAGE_SIZE,
        }
        torch.save(checkpoint, config.BEST_MODEL_PATH)
        print(f"  💾  Model saved: {config.BEST_MODEL_PATH}")

    # ── Load best model for final evaluation ──
    model.load_state_dict(best_model_state)
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)

    # ── Classification Report ──
    print("\n  📋  Classification Report:")
    print("  " + "─" * 56)
    report = classification_report(final_labels, final_preds, target_names=class_names, digits=3)
    for line in report.strip().split("\n"):
        print(f"    {line}")

    # ── Plot confusion matrix ──
    plot_confusion_matrix(final_labels, final_preds, class_names, config.CONFUSION_MATRIX_PATH)

    # ── Plot training history ──
    plot_training_history(history, config.TRAINING_HISTORY_PATH)

    print()
    print("  ✅  Training complete!")
    print(f"  📁  Model:            {config.BEST_MODEL_PATH}")
    print(f"  📊  Confusion matrix: {config.CONFUSION_MATRIX_PATH}")
    print(f"  📈  Training curves:  {config.TRAINING_HISTORY_PATH}")
    print(f"\n  💡  Next: Add images to '{config.NEW_IMAGES_DIR}/' and run `python predict.py`")
    print()


if __name__ == "__main__":
    train()
