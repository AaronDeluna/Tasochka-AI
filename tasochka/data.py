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

# Keep these characters explicitly; otherwise everything in the allowed unicode
# categories is kept and control codes / weird symbols are dropped.
EXTRA_ALLOWED = set(" \n.,!?;:-—–…\"'«»()[]{}<>/+*=%&@#№$")


_WS_RE = re.compile(r"[ \t]+")
_DOUBLE_SPACE_RE = re.compile(r" {2,}")
_TRIPLE_NL_RE = re.compile(r"\n{3,}")


def is_allowed(ch: str) -> bool:
    if ch in EXTRA_ALLOWED:
        return True
    cat = unicodedata.category(ch)
    # Letters (any script), digits
    if cat[0] in ("L", "N"):
        return True
    return False


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    out = []
    for ch in text:
        if is_allowed(ch):
            out.append(ch)
    cleaned = "".join(out)
    cleaned = _DOUBLE_SPACE_RE.sub(" ", cleaned)
    cleaned = _TRIPLE_NL_RE.sub("\n\n", cleaned)
    return cleaned


def stream_jsonl_conversations(path: Path, max_chars: int, field: str = "conversation") -> str:
    """Read JSONL, extract `field` from each line, accumulate up to max_chars."""
    chunks: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if total >= max_chars:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = obj.get(field)
            if not isinstance(value, str):
                continue
            chunks.append(value)
            chunks.append("\n")
            total += len(value) + 1
    raw = "".join(chunks)
    if len(raw) > max_chars:
        raw = raw[:max_chars]
    return raw


def stream_text_file(path: Path, max_chars: int) -> str:
    """Read a plain UTF-8 text file, accumulating up to max_chars."""
    chunks: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while total < max_chars:
            chunk = f.read(min(1024 * 1024, max_chars - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    return "".join(chunks)


def stream_text_directory(path: Path, max_chars: int, chunk_bytes: int = 256 * 1024) -> str:
    """Read all .txt files in a directory and interleave their chunks.

    Instead of concatenating files end-to-end (which would put each source in a
    contiguous block and break train/val distribution matching), we read a
    small chunk from each file in round-robin until the char budget is hit.
    Result: every region of the corpus contains a mix of all sources, so the
    last 2% used for validation has the same distribution as training.
    """
    files = sorted(path.glob("*.txt"))
    if not files:
        return ""
    per_file_budget = max_chars // len(files) + max_chars  # generous upper bound
    handles = [fp.open("r", encoding="utf-8", errors="replace") for fp in files]
    chunks: list[str] = []
    total = 0
    try:
        active = list(range(len(handles)))
        while active and total < max_chars:
            still_active: list[int] = []
            for i in active:
                if total >= max_chars:
                    break
                want = min(chunk_bytes, max_chars - total)
                piece = handles[i].read(want)
                if not piece:
                    continue
                # ensure we don't cut mid-line at the boundary
                if len(piece) == want:
                    tail = handles[i].readline()
                    piece += tail
                chunks.append(piece)
                if not piece.endswith("\n"):
                    chunks.append("\n")
                    total += 1
                total += len(piece)
                still_active.append(i)
            active = still_active
    finally:
        for h in handles:
            h.close()
    return "".join(chunks)


def corpus_signature(path: Path) -> str:
    if path.is_dir():
        parts = []
        for file_path in sorted(path.glob("*.txt")):
            stat = file_path.stat()
            parts.append(f"{file_path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return f"txtdir:{path.resolve()}|" + "|".join(parts)
    stat = path.stat()
    return f"file:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


@dataclass
class CleanedCorpus:
    text: str
    raw_chars: int
    cleaned_chars: int
    removed: int
    signature: str


def iter_cleaned_chunks(path: Path, max_chars: int,
                        chunk_size: int = 8_000_000) -> Iterator[str]:
    """Stream cleaned text chunks from a file or directory.

    For a directory of .txt files, files are read in round-robin so every
    region of the stream contains a mix of all sources. Holds at most one
    chunk in memory at a time — safe for arbitrarily large corpora.
    """
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            return
        # Files smaller than this threshold get looped on EOF so they show up
        # throughout the whole training stream, not just at the start.
        loop_threshold = 100 * 1024 * 1024  # 100 MB
        file_sizes = [fp.stat().st_size for fp in files]
        loop_file = [size < loop_threshold for size in file_sizes]
        handles = [fp.open("r", encoding="utf-8", errors="replace") for fp in files]
        try:
            total = 0
            active = list(range(len(handles)))
            while active and total < max_chars:
                still_active: list[int] = []
                for i in active:
                    if total >= max_chars:
                        break
                    piece = handles[i].read(chunk_size)
                    if not piece:
                        if loop_file[i]:
                            handles[i].seek(0)
                            piece = handles[i].read(chunk_size)
                        if not piece:
                            continue
                    if len(piece) == chunk_size:
                        piece += handles[i].readline()
                    cleaned = clean(piece)
                    if cleaned:
                        if not cleaned.endswith("\n"):
                            cleaned += "\n"
                        yield cleaned
                        total += len(cleaned)
                    still_active.append(i)
                # If the only remaining handles are small looping files,
                # we've already traversed all large sources — stop instead
                # of over-weighting the small files indefinitely.
                if not any(not loop_file[j] for j in still_active):
                    active = []
                    break
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


def encode_corpus_to_file(
    path: Path,
    tokenizer: BPETokenizer,
    max_chars: int,
    output_path: Path,
    progress_every_chars: int = 100_000_000,
) -> tuple[int, int]:
    """Encode corpus to a binary int32 file on disk, returning (cleaned_chars, total_tokens).

    The output file is meant to be read back via numpy memmap, so the encoded
    corpus never needs to fit fully in RAM.
    """
    import time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_chars = 0
    total_tokens = 0
    started = time.time()
    last_report_chars = 0
    with tmp_path.open("wb") as f:
        for chunk in iter_cleaned_chunks(path, max_chars):
            ids = tokenizer.encode(chunk)
            f.write(ids.tobytes())
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


def load_corpus(path: Path, max_chars: int) -> CleanedCorpus:
    if path.is_dir():
        raw = stream_text_directory(path, max_chars)
    elif path.suffix.lower() == ".txt":
        raw = stream_text_file(path, max_chars * 2)
    else:
        raw = stream_jsonl_conversations(path, max_chars * 2)
    cleaned = clean(raw)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return CleanedCorpus(
        text=cleaned,
        raw_chars=len(raw),
        cleaned_chars=len(cleaned),
        removed=len(raw) - len(cleaned),
        signature=corpus_signature(path),
    )


@dataclass
class TokenDataset:
    ids: np.ndarray
    rng: np.random.Generator

    @classmethod
    def from_text(cls, text: str, tokenizer: BPETokenizer, seed: int) -> "TokenDataset":
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            raise ValueError("Dataset must contain at least two tokens")
        return cls(ids=ids, rng=np.random.default_rng(seed))

    @classmethod
    def from_ids(cls, ids: np.ndarray, seed: int) -> "TokenDataset":
        if len(ids) < 2:
            raise ValueError("Dataset must contain at least two tokens")
        return cls(ids=ids, rng=np.random.default_rng(seed))

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
