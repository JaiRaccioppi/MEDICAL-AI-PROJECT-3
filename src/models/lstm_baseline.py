"""LSTM-based baseline model for sentence classification."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Embedding + BiLSTM + linear head classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of shape [batch, seq_len].
            lengths: True lengths before padding.

        Returns:
            Logits tensor [batch, num_classes].
        """
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]
        return self.classifier(self.dropout(hidden_cat))
