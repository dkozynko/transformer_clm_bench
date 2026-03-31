from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
DEFAULT_WIKITEXT2_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "validation": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}


def tokenize_line(line: str) -> list[str]:
    body = line.strip().split()
    return ["<bos>", *body, "<eos>"]


def build_vocabulary(
    token_sequences: Iterable[Sequence[str]],
    *,
    min_freq: int = 1,
    max_size: int | None = None,
) -> dict[str, int]:
    counts = Counter(token for sequence in token_sequences for token in sequence)
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    items = [(token, count) for token, count in counts.items() if count >= min_freq and token not in vocab]
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        items = items[: max(0, max_size - len(vocab))]
    for token, _ in items:
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: Sequence[str], vocab: dict[str, int]) -> list[int]:
    unk_id = vocab["<unk>"]
    return [vocab.get(token, unk_id) for token in tokens]


def decode_ids(token_ids: Sequence[int], vocab: dict[str, int]) -> list[str]:
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    return [reverse_vocab.get(token_id, "<unk>") for token_id in token_ids]


def load_token_sequences(path: Path) -> list[list[str]]:
    return [tokenize_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten_encoded_sequences(sequences: Iterable[Sequence[str]], vocab: dict[str, int]) -> list[int]:
    encoded: list[int] = []
    for tokens in sequences:
        encoded.extend(encode_tokens(tokens, vocab))
    return encoded


def ensure_wikitext2_dataset(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for split, url in DEFAULT_WIKITEXT2_URLS.items():
        path = data_dir / f"{split}.txt"
        if not path.exists():
            urlretrieve(url, path)
        split_paths[split] = path
    return split_paths


@dataclass(slots=True)
class CorpusBundle:
    vocab: dict[str, int]
    train_ids: torch.Tensor
    valid_ids: torch.Tensor
    test_ids: torch.Tensor


def load_corpus_bundle(
    data_dir: Path,
    *,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
) -> CorpusBundle:
    split_paths = ensure_wikitext2_dataset(data_dir)
    train_sequences = load_token_sequences(split_paths["train"])
    valid_sequences = load_token_sequences(split_paths["validation"])
    test_sequences = load_token_sequences(split_paths["test"])
    vocab = build_vocabulary(train_sequences, min_freq=min_freq, max_size=max_vocab_size)
    return CorpusBundle(
        vocab=vocab,
        train_ids=torch.tensor(flatten_encoded_sequences(train_sequences, vocab), dtype=torch.long),
        valid_ids=torch.tensor(flatten_encoded_sequences(valid_sequences, vocab), dtype=torch.long),
        test_ids=torch.tensor(flatten_encoded_sequences(test_sequences, vocab), dtype=torch.long),
    )


class LanguageModelingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        usable = self.token_ids.numel() - 1
        return max(0, usable // self.seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        stop = start + self.seq_len + 1
        window = self.token_ids[start:stop]
        if window.numel() != self.seq_len + 1:
            raise IndexError(index)
        return window[:-1], window[1:]
