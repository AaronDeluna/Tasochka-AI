"""Streaming corpus loader and dataset.

Cleaning is more permissive than the old char-level version: BPE will learn
em-dashes, ellipses, latin letters etc. as their own tokens, so there is no
benefit in stripping them.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .tokenizer import BPETokenizer

# Fast path: plain ASCII, Latin-1/Latin Extended, Cyrillic and common
# typographic punctuation (dashes, curly quotes, ellipsis) are always kept.
# Anything else goes through a unicode-category check in `_filter_rare`.
_RARE_RE = re.compile(r"[^\n\x20-\x7e\u00a1-\u024f\u0400-\u04ff\u2010-\u2027\u2030-\u205e]")
_INNER_SPACES_RE = re.compile(r"(?<=\S) {2,}")
_TRAILING_SPACES_RE = re.compile(r" +\n")
_TRIPLE_NL_RE = re.compile(r"\n{3,}")


def _filter_rare(match: re.Match) -> str:
    ch = match.group()
    if ch == "\t":
        return "    "
    cat = unicodedata.category(ch)
    if cat == "Zs":  # nbsp, thin space, ...
        return " "
    # Letters/digits/marks of any script, punctuation, math/currency/modifier
    # symbols. Dropped: control/format chars, emoji and other "So" symbols.
    if cat[0] in "LNMP" or cat in ("Sm", "Sc", "Sk"):
        return ch
    return ""


def clean(text: str) -> str:
    """Normalise whitespace and drop junk characters.

    Leading indentation is preserved (code in chat data stays valid Python);
    only runs of spaces *inside* a line are collapsed.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RARE_RE.sub(_filter_rare, text)
    text = _INNER_SPACES_RE.sub(" ", text)
    text = _TRAILING_SPACES_RE.sub("\n", text)
    return _TRIPLE_NL_RE.sub("\n\n", text)


def corpus_signature(path: Path) -> str:
    if path.is_dir():
        parts = []
        for file_path in sorted(path.rglob("*.txt")):
            stat = file_path.stat()
            rel = file_path.relative_to(path).as_posix()
            parts.append(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}")
        return f"txtdir:{path.resolve()}|" + "|".join(parts)
    stat = path.stat()
    return f"file:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def _read_until_blank_line(handle, size: int) -> str:
    """Read ~`size` chars, then extend to the next blank line.

    Documents and chat blocks are separated by blank lines, so cutting there
    (instead of mid-paragraph) keeps text from another source from being
    spliced into the middle of a document.
    """
    piece = handle.read(size)
    if len(piece) < size or piece.endswith("\n\n"):
        return piece
    tail = [handle.readline()]
    while tail[-1] and tail[-1] != "\n":
        tail.append(handle.readline())
    return piece + "".join(tail)


def iter_cleaned_chunks(path: Path, max_chars: int,
                        chunk_size: int = 8_000_000) -> Iterator[str]:
    """Stream cleaned text chunks from a file or directory.

    For a directory of .txt files, files are interleaved in proportion to their
    size: each round reads `chunk_size` chars from the largest file and a
    proportionally smaller piece from every other one, so all sources run out
    at the same time and every region of the stream (including the last 2%
    used for validation) has the same mix as the corpus as a whole. To give a
    source more weight, make its file bigger — nothing is repeated implicitly.
    Holds at most one chunk in memory at a time — safe for arbitrarily large
    corpora.
    """
    if path.is_dir():
        files = sorted(path.rglob("*.txt"))
        if not files:
            return
        file_sizes = [fp.stat().st_size for fp in files]
        largest = max(1, max(file_sizes))
        quotas = [max(4096, chunk_size * size // largest) for size in file_sizes]
        handles = [fp.open("r", encoding="utf-8", errors="replace") for fp in files]
        try:
            total = 0
            active = [i for i, size in enumerate(file_sizes) if size > 0]
            while active and total < max_chars:
                still_active: list[int] = []
                for i in active:
                    if total >= max_chars:
                        break
                    piece = _read_until_blank_line(handles[i], quotas[i])
                    if not piece:
                        continue
                    cleaned = clean(piece)
                    if cleaned:
                        if not cleaned.endswith("\n"):
                            cleaned += "\n"
                        yield cleaned
                        total += len(cleaned)
                    still_active.append(i)
                active = still_active
        finally:
            for h in handles:
                h.close()
        return

    # Single file: JSONL or plain text
    if path.suffix.lower() == ".jsonl":
        # JSONL still needs whole-line parsing; we stream lines and yield
        # cleaned blocks of ~chunk_size chars
        buffer: list[str] = []
        buffer_len = 0
        total = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if total >= max_chars:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = obj.get("conversation")
                if not isinstance(value, str):
                    continue
                buffer.append(value + "\n")
                buffer_len += len(value) + 1
                if buffer_len >= chunk_size:
                    cleaned = clean("".join(buffer))
                    buffer.clear()
                    buffer_len = 0
                    if cleaned:
                        yield cleaned
                        total += len(cleaned)
        if buffer:
            cleaned = clean("".join(buffer))
            if cleaned:
                yield cleaned
        return

    # Plain .txt single file
    with path.open("r", encoding="utf-8", errors="replace") as f:
        total = 0
        while total < max_chars:
            piece = f.read(chunk_size)
            if not piece:
                break
            piece += f.readline()
            cleaned = clean(piece)
            if cleaned:
                if not cleaned.endswith("\n"):
                    cleaned += "\n"
                yield cleaned
                total += len(cleaned)


def token_storage_dtype(vocab_size: int) -> np.dtype:
    """Smallest integer type that can hold every token id.

    A vocab below 65536 fits in uint16, which halves corpus_ids.bin on disk and
    halves the bytes read per batch — on a 16 GB Mac that is the difference
    between the corpus living in the page cache and thrashing the SSD.
    """
    return np.dtype(np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.int32)


def encode_corpus_to_file(
    path: Path,
    tokenizer: BPETokenizer,
    max_chars: int,
    output_path: Path,
    progress_every_chars: int = 100_000_000,
    dtype: np.dtype | None = None,
) -> tuple[int, int]:
    """Encode corpus to a binary file on disk, returning (cleaned_chars, total_tokens).

    Token ids are stored as `dtype` (uint16 when the vocab allows). The output
    is meant to be read back via numpy memmap, so the encoded corpus never
    needs to fit fully in RAM.
    """
    if dtype is None:
        dtype = token_storage_dtype(tokenizer.vocab_size)
    import time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_chars = 0
    total_tokens = 0
    started = time.time()
    last_report_chars = 0
    with tmp_path.open("wb") as f:
        for chunk in iter_cleaned_chunks(path, max_chars):
            ids = tokenizer.encode_parallel(chunk)
            f.write(ids.astype(dtype, copy=False).tobytes())
            total_tokens += int(len(ids))
            total_chars += len(chunk)
            if total_chars - last_report_chars >= progress_every_chars:
                elapsed = time.time() - started
                mb = total_chars / 1_000_000
                rate = mb / max(0.001, elapsed)
                print(f"  encoded {mb:.0f} MB → {total_tokens / 1e6:.1f}M tokens "
                      f"({rate:.1f} MB/s)")
                last_report_chars = total_chars
    tmp_path.replace(output_path)
    return total_chars, total_tokens


@dataclass
class TokenDataset:
    ids: np.ndarray
    rng: np.random.Generator
    seed: int = 0

    @classmethod
    def from_text(cls, text: str, tokenizer: BPETokenizer, seed: int) -> "TokenDataset":
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            raise ValueError("Dataset must contain at least two tokens")
        return cls(ids=ids, rng=np.random.default_rng(seed), seed=seed)

    @classmethod
    def from_ids(cls, ids: np.ndarray, seed: int) -> "TokenDataset":
        if len(ids) < 2:
            raise ValueError("Dataset must contain at least two tokens")
        return cls(ids=ids, rng=np.random.default_rng(seed), seed=seed)

    def reset_rng(self) -> None:
        """Restart the sampling stream from the original seed.

        Called before every validation pass so each evaluation scores the model
        on the *same* windows. Otherwise the RNG keeps advancing and every
        val loss is measured on different data — the number jumps around and
        "best checkpoint" selection turns into a coin flip.
        """
        self.rng = np.random.default_rng(self.seed)

    def next_batch(self, batch_size: int, ctx: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.ids) <= ctx + 1:
            raise ValueError(f"Dataset shorter than context length {ctx}")
        max_start = len(self.ids) - ctx - 1
        starts = self.rng.integers(0, max_start, size=batch_size)
        x = np.stack([self.ids[s:s + ctx] for s in starts]).astype(np.int32)
        y = np.stack([self.ids[s + 1:s + 1 + ctx] for s in starts]).astype(np.int32)
        return x, y


def split_train_val_ids(ids: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0.0 or len(ids) < 1024:
        return ids, ids[-min(512, len(ids)):]
    val_size = max(512, round(len(ids) * val_fraction))
    val_size = min(val_size, len(ids) - 512)
    cut = len(ids) - val_size
    return ids[:cut], ids[cut:]
