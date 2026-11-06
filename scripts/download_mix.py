"""Download the English + Russian pretraining mix for the 500M run.

Run it on the training server (fast network, big disk), not on the Mac:
the Russian web part alone means reading a few hundred GB from HuggingFace.

Sources, each into its own subfolder of datasets/mix/ (the trainer mixes
files in proportion to their size in characters, so disk share = training share):

  English (~70%)
    en_web/      HuggingFaceFW/fineweb-edu (sample-100BT) — educational web pages
    en_chat/     HuggingFaceTB/smol-smoltalk — assistant dialogs
  Russian (~30%)
    ru_web/      epfml/FineWeb2-HQ rus_Cyrl, quality_score >= --ru-min-quality.
                 Even this "HQ" subset is full of hotel listings, app stores and
                 machine translation below ~0.5; above it — WHO, history,
                 medicine. ru.wikipedia pages are dropped (ru_wiki has them).
    ru_wiki/     wikimedia/wikipedia 20231101.ru
    ru_chat/     IlyaGusev/saiga_scored: Russian only, opus_score >= 8, not
                 flagged by regex, and no dialog that names another assistant
                 (Claude, ChatGPT, …) — otherwise the model learns their identity.

Every source is resumable: rerun the same command after a network error.

    python scripts/download_mix.py                 # full 500M preset (~68B chars, ~90 GB)
    python scripts/download_mix.py --scale 0.001   # tiny smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from download_en import dump, render_chat, render_fineweb

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mix"

# Characters per source. A 48k EN+RU tokenizer gives ~4.4 chars/token for both
# languages, so ~68B chars ≈ 15.4B tokens: the 15B-token run plus ~3% headroom.
TARGETS = {
    "en_web": 45_000_000_000,
    "en_chat": 2_000_000_000,
    "ru_web": 15_500_000_000,
    "ru_wiki": 5_000_000_000,
    "ru_chat": 500_000_000,
}

# Word boundaries matter: without them "anthropic" hits every "philanthropic".
OTHER_ASSISTANT_RE = re.compile(
    r"\b(claude|anthropic|chatgpt|openai|gpt-?[34]\w*|gemini|llama|mistral|yandexgpt|gigachat"
    r"|сайга|saiga)\b",
    re.IGNORECASE)


def render_wiki(item: dict) -> Optional[str]:
    text = (item.get("text") or "").strip()
    if len(text) < 300:  # заглушки и списки
        return None
    return f"{item.get('title', '').strip()}\n\n{text}\n\n"


def render_saiga(item: dict) -> Optional[str]:
    if item.get("language") != "Russian" or item.get("is_bad_by_regex"):
        return None
    if (item.get("opus_score") or 0) < 8:
        return None
    if any(OTHER_ASSISTANT_RE.search(m.get("content") or "") for m in item.get("messages") or []):
        return None
    return render_chat(item)


def dump_fineweb2_hq(out_path: Path, max_chars: int, min_quality: float, workers: int) -> int:
    """FineWeb2-HQ ships 768-dim embeddings with every row, so whole-row streaming
    downloads ~3x more than the text. Read only the text/url/score columns with
    pyarrow range requests, several files in parallel; resumable by file index."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress")
    files_done, chars, size = 0, 0, 0
    if progress_path.exists() and out_path.exists():
        state = json.loads(progress_path.read_text())
        files_done, chars, size = state["files"], state["chars"], state["bytes"]
        if state.get("done"):
            print(f"ru_web: already complete ({chars / 1e6:.0f}M chars), skipping")
            return chars
        with out_path.open("r+b") as f:
            f.truncate(size)
        print(f"ru_web: resuming at {chars / 1e6:.0f}M chars ({files_done} files)")
    elif out_path.exists():
        out_path.unlink()

    files = sorted(f for f in HfApi().list_repo_files("epfml/FineWeb2-HQ", repo_type="dataset")
                   if f.startswith("rus_Cyrl/") and f.endswith(".parquet"))
    fs = HfFileSystem()

    def read_file(name: str) -> str:
        pf = pq.ParquetFile(fs.open(f"datasets/epfml/FineWeb2-HQ/{name}", block_size=16 << 20))
        parts = []
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["text", "url", "quality_score"]).to_pydict()
            for text, url, q in zip(t["text"], t["url"], t["quality_score"]):
                if q is None or q < min_quality or "wikipedia.org" in (url or ""):
                    continue
                block = render_fineweb({"text": text})
                if block:
                    parts.append(block)
        return "".join(parts)

    print(f"ru_web: {len(files)} parquet files, keeping quality >= {min_quality} "
          f"→ {out_path} (target {max_chars / 1e6:.0f}M chars)")
    todo = files[files_done:]
    with ThreadPoolExecutor(workers) as pool, out_path.open("a", encoding="utf-8") as f:
        # map() keeps file order, so the progress counter stays exact on resume.
        for text in pool.map(read_file, todo):
            if chars >= max_chars:
                break
            f.write(text)
            f.flush()
            chars += len(text)
            files_done += 1
            size = out_path.stat().st_size
            progress_path.write_text(json.dumps(
                {"files": files_done, "chars": chars, "bytes": size, "done": False}))
            print(f"  ru_web: {chars / 1e6:.0f}M / {max_chars / 1e6:.0f}M chars "
                  f"({files_done}/{len(files)} files)")
        pool.shutdown(cancel_futures=True)
    progress_path.write_text(json.dumps(
        {"files": files_done, "chars": chars, "bytes": size, "done": True}))
    print(f"ru_web: done, {chars / 1e6:.0f}M chars")
    return chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiply every target (0.001 for a quick test).")
    parser.add_argument("--only", nargs="*", choices=sorted(TARGETS), default=None)
    parser.add_argument("--ru-min-quality", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel parquet files for ru_web.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    want = set(args.only or TARGETS)
    target = {k: int(v * args.scale) for k, v in TARGETS.items()}
    out = args.out_dir
    total = 0
    if "en_web" in want:
        total += dump("en_web", lambda: load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True),
            render_fineweb, out / "en_web" / "fineweb_edu.txt", target["en_web"])
    if "en_chat" in want:
        total += dump("en_chat", lambda: load_dataset(
            "HuggingFaceTB/smol-smoltalk", split="train", streaming=True),
            render_chat, out / "en_chat" / "smoltalk_chat.txt", target["en_chat"])
    if "ru_web" in want:
        total += dump_fineweb2_hq(out / "ru_web" / "fineweb2_hq.txt", target["ru_web"],
                                  args.ru_min_quality, args.workers)
    if "ru_wiki" in want:
        total += dump("ru_wiki", lambda: load_dataset(
            "wikimedia/wikipedia", name="20231101.ru", split="train", streaming=True),
            render_wiki, out / "ru_wiki" / "wikipedia_ru.txt", target["ru_wiki"])
    if "ru_chat" in want:
        total += dump("ru_chat", lambda: load_dataset(
            "IlyaGusev/saiga_scored", split="train", streaming=True),
            render_saiga, out / "ru_chat" / "saiga_chat.txt", target["ru_chat"])
    print(f"total: {total / 1e6:.0f}M chars in {out}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
