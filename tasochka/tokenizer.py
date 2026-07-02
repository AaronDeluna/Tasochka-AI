"""BPE tokenizer wrapping HuggingFace `tokenizers`.

Same interface as the previous CharTokenizer: encode/decode/vocab_size/save/load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Union

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>"]


class BPETokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok

    @classmethod
    def train(
        cls,
        source: Union[str, Iterable[str]],
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
        )
        if isinstance(source, str):
            iter_input: Iterable[str] = _split_chunks(source)
        else:
            iter_input = source
        tok.train_from_iterator(iter_input, trainer=trainer, length=None)
        return cls(tok)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, s: str) -> np.ndarray:
        ids = self.tok.encode(s).ids
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode([int(i) for i in ids])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))


def _split_chunks(text: str, max_chunk: int = 1_000_000) -> List[str]:
    """Yield text in ~1MB chunks split on newlines."""
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chunk)
        if end < n:
            # extend to the next newline to avoid splitting a line
            nl = text.find("\n", end)
            if nl != -1:
                end = nl + 1
        chunks.append(text[start:end])
        start = end
    return chunks
