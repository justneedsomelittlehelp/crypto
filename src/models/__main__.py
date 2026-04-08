"""
CLI entry point: python -m src.models

Usage:
    python -m src.models train              # train RNN model
    python -m src.models evaluate <run_id>  # evaluate a saved checkpoint on test set
"""

import sys

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, EXPERIMENTS_DIR
from src.features.pipeline import build_feature_matrix
from src.features.dataset import create_splits, get_dataloaders
from src.models.architecture import RNNClassifier, LSTMClassifier, CNNClassifier, TransformerClassifier
from src.models.trainer import train_model
from src.models.evaluate import evaluate, print_results

import torch


MODELS = {
    "rnn": RNNClassifier,
    "lstm": LSTMClassifier,
    "cnn": CNNClassifier,
    "transformer": TransformerClassifier,
}


def cmd_train(model_name: str = "rnn"):
    print("Loading data...")
    df = build_feature_matrix()
    train_df, val_df, test_df = create_splits(df)
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df,
        lookback=LOOKBACK_BARS_MODEL,
        batch_size=BATCH_SIZE,
    )

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val:   {len(val_loader.dataset)} samples")

    model_cls = MODELS[model_name]
    model = model_cls()
    print(f"\nModel: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    result = train_model(model, train_loader, val_loader)

    # Evaluate on test set with best checkpoint
    _, _, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        lookback=LOOKBACK_BARS_MODEL,
        batch_size=BATCH_SIZE,
    )
    metrics = evaluate(model, test_loader)
    print_results(metrics)

    return result


def cmd_evaluate(run_id: str):
    run_dir = EXPERIMENTS_DIR / run_id
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        print(f"No checkpoint found at {model_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {model_path}")
    model = RNNClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))

    df = build_feature_matrix()
    train_df, val_df, test_df = create_splits(df)
    _, _, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        lookback=LOOKBACK_BARS_MODEL,
        batch_size=BATCH_SIZE,
    )

    metrics = evaluate(model, test_loader)
    print_results(metrics)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.models <command>")
        print("Commands: train, evaluate <run_id>")
        sys.exit(1)

    command = sys.argv[1]

    model_name = "rnn"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_name = sys.argv[idx + 1]
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}. Options: {list(MODELS.keys())}")
            sys.exit(1)

    if command == "train":
        cmd_train(model_name)
    elif command == "evaluate":
        if len(sys.argv) < 3:
            print("Usage: python -m src.models evaluate <run_id>")
            sys.exit(1)
        cmd_evaluate(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Commands: train, evaluate <run_id>")
        sys.exit(1)


if __name__ == "__main__":
    main()
