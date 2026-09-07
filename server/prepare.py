"""Encode immutable, fingerprinted token stores with document boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import SPECIAL, file_digest, read_json, write_json


def encode_record(record, tok):
    if "text" in record:
        ids = tok.encode(record["text"], add_special_tokens=False) + [tok.eos_token_id]
        return ids, None
    ids, labels = [], []
    for message in record["messages"]:
        role, content = message["role"], message["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role: {role}")
        if any(s in content for s in SPECIAL):
            raise ValueError("Reserved chat marker in conversation content")
        prefix = tok.encode(f"<|{role}|>\n", add_special_tokens=False)
        body = tok.encode(content + "<|end|>\n", add_special_tokens=False)
        ids.extend(prefix + body)
        labels.extend([-100] * len(prefix) + (body if role == "assistant" else [-100] * len(body)))
    return ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("server/data/raw"))
    p.add_argument("--tokenizer", default="server/data/tokenizer")
    p.add_argument("--out", type=Path, default=Path("server/data/tokens"))
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    fingerprint = file_digest(Path(a.tokenizer) / "tokenizer.json")
    sources = sorted(a.raw.glob("*/meta.json"))
    if not sources:
        p.error("Download sources first")
    a.out.mkdir(parents=True, exist_ok=True)
    for source_meta in sources:
        meta = read_json(source_meta)
        source = meta["source"]
        dest = a.out / source["name"]
        if (dest / "meta.json").exists():
            old = read_json(dest / "meta.json")
            if old["tokenizer_sha256"] != fingerprint or old["download"] != meta:
                raise ValueError(f"Stale token cache at {dest}; use a new output directory")
            continue
        dest.mkdir(exist_ok=True)
        counts = {}
        for split in ("train", "val"):
            n, docs = 0, 0
            with (source_meta.parent / f"{split}.jsonl").open(encoding="utf-8") as src, \
                 (dest / f"{split}.bin").open("wb") as ids_file, \
                 (dest / f"{split}.labels").open("wb") as label_file, \
                 (dest / f"{split}.idx").open("wb") as idx:
                np.asarray([0], dtype="<u8").tofile(idx)
                for line in src:
                    ids, labels = encode_record(json.loads(line), tok)
                    if labels is not None and not any(x != -100 for x in labels[1:]):
                        continue
                    np.asarray(ids, dtype="<u4").tofile(ids_file)
                    if labels is not None:
                        np.asarray(labels, dtype="<i4").tofile(label_file)
                    n += len(ids)
                    docs += 1
                    np.asarray([n], dtype="<u8").tofile(idx)
            counts[split] = {"tokens": n, "documents": docs}
        write_json(dest / "meta.json", {"download": meta, "source": source,
            "tokenizer_sha256": fingerprint, "vocab_size": len(tok), "counts": counts})
        print(source["name"], counts, flush=True)


if __name__ == "__main__":
    main()
