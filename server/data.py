"""Memory-mapped corpora and deterministic token-weighted source sampling."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_json


class Corpus:
    def __init__(self, path, split, context, mode):
        self.path, self.context, self.mode = path, context, mode
        self.meta = read_json(path / "meta.json")
        self.ids = np.memmap(path / f"{split}.bin", dtype="<u4", mode="r")
        self.offsets = np.memmap(path / f"{split}.idx", dtype="<u8", mode="r")
        if self.offsets[-1] != len(self.ids):
            raise ValueError(f"Corrupt document index: {path}")
        self.labels = None
        if mode == "sft":
            self.labels = np.memmap(path / f"{split}.labels", dtype="<i4", mode="r")
            if len(self.labels) != len(self.ids):
                raise ValueError("Corrupt SFT labels")
            # Do not truncate the answer or silently train only a user prompt.
            lengths = np.diff(self.offsets)
            self.eligible = np.flatnonzero((lengths > 1) & (lengths <= context))
            self.size = len(self.eligible)
        elif mode == "long":
            # Full, contiguous windows inside real documents; unrelated short
            # documents packed together are not evidence of long-context learning.
            lengths = np.diff(self.offsets)
            self.windows = (lengths // context).astype(np.int64)
            self.cumulative = np.cumsum(self.windows)
            self.size = int(self.cumulative[-1])
        else:
            self.size = len(self.ids) // context
        if self.size < 1:
            raise ValueError(f"{path}/{split}: no usable {mode} examples at context={context}")

    def get(self, index):
        if self.mode == "sft":
            doc = self.eligible[index % self.size]
            start, end = map(int, self.offsets[doc:doc + 2])
        elif self.mode == "long":
            index %= self.size
            doc = int(np.searchsorted(self.cumulative, index, side="right"))
            previous = int(self.cumulative[doc - 1]) if doc else 0
            start = int(self.offsets[doc]) + (index - previous) * self.context
            end = start + self.context
        else:
            start = (index % self.size) * self.context
            end = start + self.context
        ids = torch.tensor(np.array(self.ids[start:end], dtype=np.int64))
        labels = ids.clone() if self.labels is None else torch.tensor(np.array(self.labels[start:end], dtype=np.int64))
        return {"input_ids": ids, "labels": labels}


class Mixture(Dataset):
    def __init__(self, root, split, context, mode, fingerprint, seed=42, names=None, eval_limit=256):
        self.corpora, weights = [], []
        for path in sorted(Path(root).glob("*/meta.json")):
            meta = read_json(path)
            s = meta["source"]
            if (s["kind"] == "sft") != (mode == "sft") or (names and s["name"] not in names):
                continue
            if meta["tokenizer_sha256"] != fingerprint:
                raise ValueError(f"Tokenizer mismatch: {path}")
            self.corpora.append(Corpus(path.parent, split, context, mode))
            weights.append(s["weight"])
        if not self.corpora or any(w <= 0 for w in weights):
            raise ValueError("No corpora or nonpositive mixture weight")
        if names and set(names) != {c.meta["source"]["name"] for c in self.corpora}:
            raise ValueError("Some requested sources are missing or have the wrong training mode")
        self.probabilities = np.asarray(weights) / sum(weights)
        self.seed, self.split = seed, split
        self.size = sum(c.size for c in self.corpora)
        if split == "val":
            self.size = min(self.size, eval_limit)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Stateless mapping: Trainer's seeded sampler and checkpoint data skip
        # reproduce the same examples on resume, independent of worker count.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        source = int(rng.choice(len(self.corpora), p=self.probabilities))
        corpus = self.corpora[source]
        return corpus.get(int(rng.integers(corpus.size)))


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, records):
        length = max(len(r["input_ids"]) for r in records)
        ids = torch.full((len(records), length), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(records):
            n = len(r["input_ids"])
            ids[i, :n], labels[i, :n], mask[i, :n] = r["input_ids"], r["labels"], 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
