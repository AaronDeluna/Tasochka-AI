"""Download an English pretraining + chat mix from HuggingFace.

Two sources, each into its own subfolder of datasets/en/:

  * fineweb_edu/  — HuggingFaceFW/fineweb-edu (sample-10BT): web pages filtered
                    for educational value. The bulk of the corpus: language,
                    world knowledge, explanations. Plain text, no chat tags.
  * smoltalk/     — HuggingFaceTB/smol-smoltalk: multi-turn assistant dialogs
                    (rewriting, math, code, Q&A) built for models under 1B.
                    Rendered as <user>/<assistant> blocks — the same format
                    `tasochka.generate --chat` and the server prompt with.

The trainer interleaves files in proportion to their size, so the share of
chat in the training stream equals its share on disk (mac preset: ~20%).

Presets:
  mac     ~0.5 GB  (≈120M tokens) — enough for a 1-hour check on a MacBook
  server  ~13 GB   (≈3B tokens)   — a ~125M model trained Chinchilla-style

Downloads are resumable: rerun the same command after a network error and it
continues where it stopped.

    python scripts/download_en.py --preset mac
    python scripts/download_en.py --preset server
    python scripts/download_en.py --web-chars 2e9 --chat-chars 2e8   # custom
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "en"

PRESETS = {
    "mac": {"web_chars": 400_000_000, "chat_chars": 100_000_000},
    "server": {"web_chars": 12_000_000_000, "chat_chars": 2_000_000_000},
}


def format_turn(role: str, content: str) -> str:
    tag = "user" if role.lower() in ("user", "human") else "assistant"
    return f"<{tag}>\n{content.strip()}\n</{tag}>\n"


def render_fineweb(item: dict) -> Optional[str]:
    text = (item.get("text") or "").strip()
    if len(text) < 200:  # обрывки меню/футеров
        return None
    return text + "\n\n"


def render_chat(item: dict) -> Optional[str]:
    parts = []
    for msg in item.get("messages") or []:
        role = (msg.get("role") or "").lower()
        content = (msg.get("content") or "").strip()
        if not content or role == "system":
            continue
        parts.append(format_turn(role, content))
    # Нужен хотя бы один обмен user → assistant
    if len(parts) < 2:
        return None
    return "".join(parts) + "\n"


def dump(name: str, rows: Callable[[], Iterable[dict]], render: Callable[[dict], Optional[str]],
         out_path: Path, max_chars: int) -> int:
    """Stream `rows` into `out_path` until `max_chars`, resumable via a .progress file.

    The progress file records how many source rows were consumed and the byte
    length of the output at that point. On restart the output is truncated back
    to that length (dropping a half-written tail) and the stream skips the rows
    already used.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress")
    rows_done, chars, size = 0, 0, 0
    if progress_path.exists() and out_path.exists():
        state = json.loads(progress_path.read_text())
        rows_done, chars, size = state["rows"], state["chars"], state["bytes"]
        if state.get("done") and chars >= max_chars:
            print(f"{name}: already complete ({chars / 1e6:.0f}M chars), skipping")
            return chars
        with out_path.open("r+b") as f:
            f.truncate(size)
        print(f"{name}: resuming at {chars / 1e6:.0f}M chars ({rows_done} rows)")
    elif out_path.exists():
        out_path.unlink()

    def save_progress(done: bool = False) -> None:
        progress_path.write_text(json.dumps(
            {"rows": rows_done, "chars": chars, "bytes": size, "done": done}))

    print(f"{name}: streaming → {out_path} (target {max_chars / 1e6:.0f}M chars)")
    stream = iter(rows())
    for _ in range(rows_done):  # skip what is already on disk
        next(stream, None)
    next_report = chars + max(1, max_chars // 20)
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for item in stream:
                if chars >= max_chars:
                    break
                rows_done += 1
                block = render(item)
                if block is None:
                    continue
                f.write(block)
                chars += len(block)
                if chars >= next_report:
                    f.flush()
                    size = out_path.stat().st_size
                    save_progress()
                    print(f"  {name}: {chars / 1e6:.0f}M / {max_chars / 1e6:.0f}M chars")
                    next_report += max(1, max_chars // 20)
            f.flush()
            size = out_path.stat().st_size
    except Exception as exc:  # SSL/сеть рвутся на долгих закачках
        save_progress()
        print(f"  ⚠️  {name}: connection lost at {chars / 1e6:.0f}M chars "
              f"({type(exc).__name__}: {exc}). Rerun the same command to resume.")
        return chars
    save_progress(done=True)
    print(f"{name}: done, {chars / 1e6:.0f}M chars")
    return chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mac")
    parser.add_argument("--web-chars", type=float, default=None,
                        help="Override FineWeb-Edu size in characters (e.g. 2e9).")
    parser.add_argument("--chat-chars", type=float, default=None,
                        help="Override smol-smoltalk size in characters.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    web_chars = int(args.web_chars if args.web_chars is not None else preset["web_chars"])
    chat_chars = int(args.chat_chars if args.chat_chars is not None else preset["chat_chars"])

    total = 0
    if web_chars > 0:
        total += dump(
            "fineweb-edu",
            lambda: load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                 split="train", streaming=True),
            render_fineweb, args.out_dir / "fineweb_edu" / "fineweb_edu.txt", web_chars)
    if chat_chars > 0:
        total += dump(
            "smol-smoltalk",
            lambda: load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True),
            render_chat, args.out_dir / "smoltalk" / "smoltalk_chat.txt", chat_chars)
    print(f"total: {total / 1e6:.0f}M chars in {args.out_dir}")
    # HF streaming leaves non-daemon threads behind that keep the interpreter
    # alive after main() returns; files are closed, so flush stdout and exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
