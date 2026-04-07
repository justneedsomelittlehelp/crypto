import torch
import torch.nn as nn

from src.config import (
    FEATURE_COLS,
    RNN_HIDDEN_SIZES,
    RNN_DROPOUT,
    RNN_ACTIVATION,
)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = len(FEATURE_COLS),
        hidden_sizes: list[int] = RNN_HIDDEN_SIZES,
        dropout: float = RNN_DROPOUT,
        activation: str = RNN_ACTIVATION,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.RNN(
                input_size=prev_size,
                hidden_size=h,
                num_layers=1,
                nonlinearity=activation,
                batch_first=True,
            ))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = h

        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback, features]
        out = x
        for rnn, drop in zip(self.layers, self.dropouts):
            out, _ = rnn(out)
            out = drop(out)

        # Take the last timestep's hidden state
        last = out[:, -1, :]
        logit = self.fc(last).squeeze(-1)
        return logit
