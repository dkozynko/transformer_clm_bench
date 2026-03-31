from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset


WORD_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
SPECIAL_TOKENS = WORD_SPECIAL_TOKENS
BYTE_PAD_ID = 256
BYTE_BOS_ID = 257
BYTE_EOS_ID = 258
BYTE_VOCAB_SIZE = 259
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
    vocab = {token: idx for idx, token in enumerate(WORD_SPECIAL_TOKENS)}
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


def encode_text(text: str, *, tokenizer_name: str, vocab: dict[str, int] | None = None) -> list[int]:
    if tokenizer_name == "word":
        if vocab is None:
            raise ValueError("Word tokenization requires a vocabulary.")
        tokens = tokenize_line(text)
        return encode_tokens(tokens, vocab)
    if tokenizer_name == "byte":
        return [BYTE_BOS_ID, *text.encode("utf-8"), BYTE_EOS_ID]
    raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")


def decode_token_ids(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    tokenizer_name: str,
    vocab: dict[str, int] | None = None,
) -> str:
    ids = [int(token_id) for token_id in token_ids]
    if tokenizer_name == "word":
        if vocab is None:
            raise ValueError("Word decoding requires a vocabulary.")
        reverse_vocab = {idx: token for token, idx in vocab.items()}
        tokens: list[str] = []
        for token_id in ids:
            token = reverse_vocab.get(token_id, "<unk>")
            if token == "<eos>":
                break
            if token in {"<pad>", "<bos>"}:
                continue
            tokens.append(token)
        return " ".join(tokens)
    if tokenizer_name == "byte":
        raw_bytes = bytearray()
        for token_id in ids:
            if token_id == BYTE_EOS_ID:
                break
            if token_id in {BYTE_PAD_ID, BYTE_BOS_ID}:
                continue
            if 0 <= token_id < 256:
                raw_bytes.append(token_id)
        return raw_bytes.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")


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
    tokenizer_name: str
    vocab: dict[str, int] | None
    vocab_size: int
    train_ids: torch.Tensor
    valid_ids: torch.Tensor
    test_ids: torch.Tensor


def load_corpus_bundle(
    data_dir: Path,
    *,
    tokenizer_name: str = "word",
    min_freq: int = 1,
    max_vocab_size: int | None = None,
) -> CorpusBundle:
    split_paths = ensure_wikitext2_dataset(data_dir)
    if tokenizer_name == "word":
        train_sequences = load_token_sequences(split_paths["train"])
        valid_sequences = load_token_sequences(split_paths["validation"])
        test_sequences = load_token_sequences(split_paths["test"])
        vocab = build_vocabulary(train_sequences, min_freq=min_freq, max_size=max_vocab_size)
        return CorpusBundle(
            tokenizer_name=tokenizer_name,
            vocab=vocab,
            vocab_size=len(vocab),
            train_ids=torch.tensor(flatten_encoded_sequences(train_sequences, vocab), dtype=torch.long),
            valid_ids=torch.tensor(flatten_encoded_sequences(valid_sequences, vocab), dtype=torch.long),
            test_ids=torch.tensor(flatten_encoded_sequences(test_sequences, vocab), dtype=torch.long),
        )
    if tokenizer_name == "byte":
        train_text = split_paths["train"].read_text(encoding="utf-8")
        valid_text = split_paths["validation"].read_text(encoding="utf-8")
        test_text = split_paths["test"].read_text(encoding="utf-8")
        return CorpusBundle(
            tokenizer_name=tokenizer_name,
            vocab=None,
            vocab_size=BYTE_VOCAB_SIZE,
            train_ids=torch.tensor(encode_text(train_text, tokenizer_name="byte"), dtype=torch.long),
            valid_ids=torch.tensor(encode_text(valid_text, tokenizer_name="byte"), dtype=torch.long),
            test_ids=torch.tensor(encode_text(test_text, tokenizer_name="byte"), dtype=torch.long),
        )
    raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")


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
