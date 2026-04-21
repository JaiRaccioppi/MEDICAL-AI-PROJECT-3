"""Data loading and preprocessing for PubMed RCT 20k sentence classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


LABEL_NAMES = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]


def normalize_text(text: str) -> str:
    """Normalize sentence text for the recurrent baseline.

    Args:
        text: Raw sentence.

    Returns:
        Lower-cased sentence with trimmed whitespace.
    """
    return " ".join(text.lower().strip().split())


def load_pubmed_rct20k(dataset_name: str, quick_mode: bool, cfg: Dict) -> DatasetDict:
    """Load dataset splits from Hugging Face datasets hub.

    Args:
        dataset_name: Hugging Face dataset identifier.
        quick_mode: Whether to use reduced subsets.
        cfg: Configuration dictionary.

    Returns:
        DatasetDict with train, validation, and test splits.
    """
    dataset = load_dataset(dataset_name)
    if not quick_mode:
        return dataset

    return DatasetDict(
        {
            "train": dataset["train"].shuffle(seed=cfg["seed"]).select(range(cfg["quick_train_size"])),
            "validation": dataset["validation"]
            .shuffle(seed=cfg["seed"])
            .select(range(cfg["quick_validation_size"])),
            "test": dataset["test"].shuffle(seed=cfg["seed"]).select(range(cfg["quick_test_size"])),
        }
    )


def build_vocab(texts: List[str], max_vocab_size: int, min_token_frequency: int) -> Dict[str, int]:
    """Build token vocabulary for recurrent models.

    Args:
        texts: List of input sentences.
        max_vocab_size: Maximum vocabulary size excluding special tokens.
        min_token_frequency: Minimum count to keep a token.

    Returns:
        Mapping from token to index.
    """
    counter = Counter()
    for text in texts:
        counter.update(normalize_text(text).split())

    sorted_tokens = [
        token
        for token, count in counter.most_common(max_vocab_size)
        if count >= min_token_frequency
    ]

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_tokens(text: str, vocab: Dict[str, int], max_length: int) -> Tuple[List[int], int]:
    """Tokenize sentence with a whitespace tokenizer and map into vocabulary.

    Args:
        text: Input sentence.
        vocab: Token-to-index mapping.
        max_length: Maximum sequence length.

    Returns:
        Padded token id sequence and original truncated length.
    """
    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in normalize_text(text).split()]
    token_ids = token_ids[:max_length]
    seq_len = len(token_ids)

    if seq_len < max_length:
        token_ids.extend([vocab["<PAD>"]] * (max_length - seq_len))
    return token_ids, seq_len


@dataclass
class RNNBatch:
    """Container for recurrent model batch tensors."""

    input_ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


class RNNDataset(Dataset):
    """PyTorch dataset wrapper for recurrent baseline model."""

    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids, seq_len = encode_tokens(self.texts[idx], self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(seq_len, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def rnn_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> RNNBatch:
    """Collate function for recurrent dataloader.

    Args:
        batch: List of dataset examples.

    Returns:
        RNNBatch dataclass with stacked tensors.
    """
    return RNNBatch(
        input_ids=torch.stack([item["input_ids"] for item in batch]),
        lengths=torch.stack([item["length"] for item in batch]),
        labels=torch.stack([item["labels"] for item in batch]),
    )


def create_rnn_dataloaders(dataset: DatasetDict, cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create train/validation/test dataloaders for recurrent baseline.

    Args:
        dataset: Hugging Face dataset dictionary.
        cfg: Configuration dictionary.

    Returns:
        Tuple of train, validation, test dataloaders and vocabulary.
    """
    train_texts = [row[cfg["text_column"]] for row in dataset["train"]]
    vocab = build_vocab(
        train_texts,
        max_vocab_size=cfg["lstm"]["max_vocab_size"],
        min_token_frequency=cfg["lstm"]["min_token_frequency"],
    )

    def make_split(split: str) -> RNNDataset:
        texts = [row[cfg["text_column"]] for row in dataset[split]]
        labels = [int(row[cfg["label_column"]]) for row in dataset[split]]
        return RNNDataset(texts, labels, vocab, cfg["max_length"])

    train_ds = make_split("train")
    val_ds = make_split("validation")
    test_ds = make_split("test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=rnn_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=rnn_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=rnn_collate_fn,
    )
    return train_loader, val_loader, test_loader, vocab


def create_transformer_tokenizer(model_name: str) -> AutoTokenizer:
    """Instantiate a Hugging Face tokenizer.

    Args:
        model_name: Name or path for pretrained transformer tokenizer.

    Returns:
        Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)
