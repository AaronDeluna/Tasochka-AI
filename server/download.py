"""Bounded-memory download, strict schemas, disk-backed exact deduplication."""
import argparse
import json
import sqlite3
from contextlib import ExitStack
from pathlib import Path

from .common import digest, extract, partition, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="server/configs/datasets.json")
    p.add_argument("--out", type=Path, default=Path("server/data/raw"))
    p.add_argument("--sources", nargs="+")
    p.add_argument("--max-records", type=int, help="Per source; for download smoke checks only")
    a = p.parse_args()
    if a.max_records is not None and a.max_records < 1:
        p.error("--max-records must be positive")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    manifest = read_json(a.manifest)
    sources = manifest["sources"]
    if a.sources:
        unknown = set(a.sources) - {s["name"] for s in sources}
        if unknown:
            p.error(f"Unknown sources: {unknown}")
        sources = [s for s in sources if s["name"] in a.sources]
    a.out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(a.out / "dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, source TEXT)")
    for s in sources:
        target = a.out / s["name"]
        if (target / "meta.json").exists():
            old = read_json(target / "meta.json")
            if old["source"] != s or old["max_records"] != a.max_records or old["validation_fraction"] != manifest["validation_fraction"]:
                raise ValueError(f"{target}: settings changed; use a fresh --out directory")
            print(f"Already complete: {target}", flush=True)
            continue
        target.mkdir(exist_ok=True)
        # Interrupted source is rebuilt; completed sources remain intact.
        db.execute("DELETE FROM seen WHERE source = ?", (s["name"],))
        db.commit()
        revision = HfApi().dataset_info(s["id"], revision=s.get("revision", "main")).sha
        kwargs = {k: s[k] for k in ("data_dir", "data_files") if s.get(k)}
        if s.get("loader"):
            files = [f"hf://datasets/{s['id']}@{revision}/{f}" for f in s["files"]]
            stream = load_dataset(s["loader"], data_files={s["split"]: files}, split=s["split"], streaming=True)
        else:
            stream = load_dataset(s["id"], name=s.get("config"), split=s["split"], revision=revision,
                                  streaming=True, **kwargs)
        counts = {"train": 0, "val": 0, "duplicate": 0, "filtered": 0, "chars": 0}
        with ExitStack() as stack:
            handles = {split: stack.enter_context((target / f"{split}.jsonl.tmp").open("w", encoding="utf-8")) for split in ("train", "val")}
            for i, row in enumerate(stream):
                if a.max_records is not None and i >= a.max_records:
                    break
                item = extract(row, s)
                if item is None:
                    counts["filtered"] += 1
                    continue
                record, key = item
                if db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (digest(key), s["name"])).rowcount == 0:
                    counts["duplicate"] += 1
                    continue
                split = partition(key, manifest["validation_fraction"])
                record["source"] = s["name"]
                record["provenance"] = {k: row[k] for k in ("id", "url", "repository_name", "repo_name", "path", "license", "licenses") if k in row}
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                counts["chars"] += len(json.dumps(record, ensure_ascii=False))
                if i % 10000 == 0:
                    db.commit()
                    print(s["name"], counts, flush=True)
                if counts["chars"] >= s["max_chars"]:
                    break
        if not counts["train"]:
            raise ValueError(f"No usable training data: {s['name']}")
        db.commit()
        for split in handles:
            (target / f"{split}.jsonl.tmp").replace(target / f"{split}.jsonl")
        write_json(target / "meta.json", {"source": s, "revision": revision, "counts": counts,
                                         "max_records": a.max_records, "validation_fraction": manifest["validation_fraction"]})
        print(s["name"], counts, flush=True)
    db.close()


if __name__ == "__main__":
    main()
