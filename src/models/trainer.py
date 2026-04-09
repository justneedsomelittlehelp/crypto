import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    RUNS_DIR,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    run_id: str | None = None,
) -> dict:
    device = get_device()

    if run_id is None:
        run_id = f"run_{int(time.time())}"

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Compute class weight from training data to counter majority-class collapse
    all_labels = torch.cat([y for _, y in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 else torch.tensor([1.0], device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y)
            preds = (logits > 0).float()
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * len(y)
                preds = (logits > 0).float()
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        # --- Early stopping + checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        "lr": lr,
        "epochs_run": epoch,
        "patience": patience,
        "best_val_loss": best_val_loss,
        "model_class": model.__class__.__name__,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {run_dir / 'model.pt'}")

    # Load best weights back into model (on CPU for compatibility)
    model = model.cpu()
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))

    return {"run_id": run_id, "run_dir": run_dir, "history": history, "best_val_loss": best_val_loss}
