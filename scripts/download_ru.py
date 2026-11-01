"""Download high-quality Russian dialogue datasets from HuggingFace
and convert to the <user>/<assistant> format used by the existing training data.

Each dataset is written into its own subfolder: datasets/ru/<source>/<name>.txt.
The trainer reads datasets/ recursively, so the next `python -m tasochka.train`
run with `--data datasets/ru` picks up everything automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "ru"


def format_turn(role: str, content: str) -> str:
    tag = "user" if role.lower() in ("user", "human") else "assistant"
    return f"<{tag}>\n{content.strip()}\n</{tag}>\n"


def dump_saiga(out_path: Path, min_score: int = 7, max_items: int | None = None) -> int:
    """Saiga is multi-turn, opus_score-filtered. Quality is the best on HF for Russian."""
    print(f"streaming IlyaGusev/saiga_scored → {out_path.name} (min_score={min_score})")
    ds = load_dataset("IlyaGusev/saiga_scored", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            if item.get("language") != "Russian":
                continue
            score = item.get("opus_score")
            if score is not None and score < min_score:
                continue
            messages = item.get("messages") or []
            block_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                block_parts.append(format_turn(role, content))
            if not block_parts:
                continue
            block = "".join(block_parts) + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 1000 == 0:
                print(f"  {written} conversations, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"saiga done: {written} conversations, {chars / 1e6:.1f} MB")
    return chars


def dump_grandmaster(out_path: Path, max_items: int | None = None) -> int:
    """Vikhrmodels/GrandMaster-PRO-MAX — newer high-quality Russian multi-turn corpus."""
    print(f"streaming Vikhrmodels/GrandMaster-PRO-MAX → {out_path.name}")
    ds = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            # Filter: both prompt and answer in Russian
            if item.get("prompt_lang") != "ru" or item.get("answer_lang") != "ru":
                continue
            conversation = item.get("conversation") or []
            block_parts = []
            for msg in conversation:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                block_parts.append(format_turn(role, content))
            if not block_parts:
                continue
            block = "".join(block_parts) + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 1000 == 0:
                print(f"  {written} conversations, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"GrandMaster done: {written} conversations, {chars / 1e6:.1f} MB")
    return chars


def dump_gsm8k_ru(out_path: Path, max_items: int | None = None) -> int:
    """d0rj/gsm8k-ru — math word problems with step-by-step solutions."""
    print(f"streaming d0rj/gsm8k-ru → {out_path.name}")
    ds = load_dataset("d0rj/gsm8k-ru", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q or not a:
                continue
            block = format_turn("user", q) + format_turn("assistant", a) + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 1000 == 0:
                print(f"  {written} examples, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"gsm8k done: {written} examples, {chars / 1e6:.1f} MB")
    return chars


def dump_alpaca_ru(out_path: Path, max_items: int | None = None) -> int:
    """d0rj/alpaca-cleaned-ru — instruction/input/output dataset translated to Russian."""
    print(f"streaming d0rj/alpaca-cleaned-ru → {out_path.name}")
    ds = load_dataset("d0rj/alpaca-cleaned-ru", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            instr = (item.get("instruction") or "").strip()
            inp = (item.get("input") or "").strip()
            out = (item.get("output") or "").strip()
            if not instr or not out:
                continue
            user = instr if not inp else f"{instr}\n\n{inp}"
            block = format_turn("user", user) + format_turn("assistant", out) + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 2000 == 0:
                print(f"  {written} examples, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"alpaca-ru done: {written} examples, {chars / 1e6:.1f} MB")
    return chars


def _render_messages(messages, drop_system: bool = True) -> str:
    """Convert HF-style list of {role, content} into <user>/<assistant> blocks."""
    parts = []
    for msg in messages or []:
        role = (msg.get("role") or "").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system" and drop_system:
            continue
        parts.append(format_turn(role, content))
    return "".join(parts)


def dump_saiga_prefs(out_path: Path, max_items: int | None = None) -> int:
    """IlyaGusev/saiga_preferences — keep only the chosen (high-quality) response.

    Both `prompt` and `chosen` are lists of {role, content}. System messages
    are dropped — they leak persona shaping that we don't want in pretraining.
    """
    print(f"streaming IlyaGusev/saiga_preferences → {out_path.name}")
    ds = load_dataset("IlyaGusev/saiga_preferences", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            prompt_block = _render_messages(item.get("prompt"))
            chosen_block = _render_messages(item.get("chosen"))
            if not prompt_block or not chosen_block:
                continue
            block = prompt_block + chosen_block + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 1000 == 0:
                print(f"  {written} examples, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"saiga_prefs done: {written} examples, {chars / 1e6:.1f} MB")
    return chars


def dump_wikipedia_ru(out_path: Path, max_chars: int = 3_000_000_000) -> int:
    """wikimedia/wikipedia 20231101.ru — raw encyclopedia text. NOT chat format.

    Pretraining-style data: full Russian articles. Gives the model knowledge
    of language and basic facts. We do NOT wrap in <user>/<assistant> tags —
    the model learns chat format from the chat files separately.
    """
    print(f"streaming wikimedia/wikipedia 20231101.ru → {out_path.name} "
          f"(cap {max_chars / 1e9:.1f} GB)")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True)
    written_articles = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            if chars >= max_chars:
                break
            title = (item.get("title") or "").strip()
            text = (item.get("text") or "").strip()
            if not text:
                continue
            block = f"# {title}\n\n{text}\n\n"
            f.write(block)
            chars += len(block)
            written_articles += 1
            if written_articles % 5000 == 0:
                print(f"  {written_articles} articles, {chars / 1e6:.1f} MB")
    print(f"wikipedia done: {written_articles} articles, {chars / 1e6:.1f} MB")
    return chars


def dump_web_ru(out_path: Path, max_chars: int = 5_000_000_000) -> int:
    """HuggingFaceFW/fineweb-2 (rus_Cyrl) — большой чистый русский веб-корпус.

    Это ГЛАВНЫЙ источник объёма для крупной модели (1B и выше): современный,
    дедуплицированный, качественно отфильтрованный текст из веба. Стримится,
    поэтому качаем ровно `max_chars` символов и ни байтом больше — удобно
    подгонять под свободное место на диске.

    Как и Wikipedia — это pretraining-текст, БЕЗ <user>/<assistant> тегов.

    Ориентир по объёму (сколько текста нужно правилу Chinchilla ~20 ток/парам):
      * ~200-300M модель  → 5-8 GB  (~1.5-2 B токенов)
      * ~1B модель        → 60-90 GB (~15-20 B токенов)
    На Mac (мало диска) ставь --web-max-chars 3_000_000_000 (~3 GB) для теста,
    на сервере — большой лимит под полный прогон.
    """
    print(f"streaming HuggingFaceFW/fineweb-2 rus_Cyrl → {out_path.name} "
          f"(cap {max_chars / 1e9:.1f} GB)")
    print("  (первое разрешение шардов может занять несколько минут — это норма)")
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl",
                      split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        try:
            for item in ds:
                if chars >= max_chars:
                    break
                text = (item.get("text") or "").strip()
                if len(text) < 200:  # выкидываем совсем короткие обрывки
                    continue
                block = f"{text}\n\n"
                f.write(block)
                chars += len(block)
                written += 1
                if written % 20000 == 0:
                    print(f"  {written} docs, {chars / 1e9:.2f} GB")
        except Exception as e:  # SSL/сеть могут оборваться на долгой закачке
            f.flush()
            print(f"  ⚠️  обрыв на {chars / 1e9:.2f} GB ({type(e).__name__}: {e}). "
                  f"Уже скачанное сохранено — перезапусти скрипт, чтобы докачать.")
    print(f"web_ru done: {written} docs, {chars / 1e9:.2f} GB")
    return chars


def dump_veles(out_path: Path, max_items: int | None = None) -> int:
    """Vikhrmodels/Veles-2.5 — high-quality multi-turn Russian (translated)."""
    print(f"streaming Vikhrmodels/Veles-2.5 → {out_path.name}")
    ds = load_dataset("Vikhrmodels/Veles-2.5", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            conv = item.get("conversations") or []
            parts = []
            for msg in conv:
                role_raw = (msg.get("from") or msg.get("role") or "").lower()
                content = (msg.get("value") or msg.get("content") or "").strip()
                if not content:
                    continue
                if role_raw in ("system", "instruction"):
                    continue
                role = "user" if role_raw in ("human", "user") else "assistant"
                parts.append(format_turn(role, content))
            if not parts:
                continue
            block = "".join(parts) + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 1000 == 0:
                print(f"  {written} dialogs, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"veles done: {written} dialogs, {chars / 1e6:.1f} MB")
    return chars


def dump_russian_dialogues(out_path: Path, max_items: int | None = None) -> int:
    """Den4ikAI/russian_dialogues — short Q&A pairs."""
    print(f"streaming Den4ikAI/russian_dialogues → {out_path.name}")
    ds = load_dataset("Den4ikAI/russian_dialogues", split="train", streaming=True)
    written = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q or not a:
                continue
            block = format_turn("user", q) + format_turn("assistant", a) + "\n"
            f.write(block)
            chars += len(block)
            written += 1
            if written % 5000 == 0:
                print(f"  {written} pairs, {chars / 1e6:.1f} MB")
            if max_items and written >= max_items:
                break
    print(f"russian_dialogues done: {written} pairs, {chars / 1e6:.1f} MB")
    return chars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-saiga", action="store_true")
    parser.add_argument("--skip-grandmaster", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--skip-alpaca", action="store_true")
    parser.add_argument("--skip-prefs", action="store_true")
    parser.add_argument("--skip-wiki", action="store_true")
    parser.add_argument("--skip-web", action="store_true",
                       help="Пропустить большой веб-корпус FineWeb-2 (по умолчанию качается)")
    parser.add_argument("--skip-veles", action="store_true")
    parser.add_argument("--skip-dialogues", action="store_true")
    parser.add_argument("--wiki-max-chars", type=int, default=3_000_000_000,
                       help="Cap Wikipedia output size (bytes/chars). 3 GB default.")
    parser.add_argument("--web-max-chars", type=int, default=5_000_000_000,
                       help="Лимит веб-корпуса FineWeb-2 в символах. 5 GB по умолч.; "
                            "на Mac ставь 3_000_000_000, на сервере — 60-90 млрд для 1B.")
    parser.add_argument("--saiga-min-score", type=int, default=7,
                       help="Saiga conversations with opus_score below this are skipped")
    parser.add_argument("--max-items", type=int, default=None,
                       help="Cap per-dataset for testing")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def dst(subdir: str, name: str) -> Path:
        """Каждый датасет — в свою подпапку datasets/<subdir>/<name>."""
        d = args.out_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    total = 0
    if not args.skip_saiga:
        total += dump_saiga(dst("saiga", "saiga_chat.txt"), args.saiga_min_score, args.max_items)
    if not args.skip_grandmaster:
        total += dump_grandmaster(dst("grandmaster", "grandmaster_chat.txt"), args.max_items)
    if not args.skip_gsm8k:
        total += dump_gsm8k_ru(dst("gsm8k", "gsm8k_chat.txt"), args.max_items)
    if not args.skip_alpaca:
        total += dump_alpaca_ru(dst("alpaca_ru", "alpaca_ru_chat.txt"), args.max_items)
    if not args.skip_prefs:
        total += dump_saiga_prefs(dst("saiga_prefs", "saiga_prefs_chat.txt"), args.max_items)
    if not args.skip_wiki:
        total += dump_wikipedia_ru(dst("wikipedia", "wikipedia_ru.txt"), args.wiki_max_chars)
    if not args.skip_web:
        total += dump_web_ru(dst("web_ru", "web_ru.txt"), args.web_max_chars)
    if not args.skip_veles:
        total += dump_veles(dst("veles", "veles_chat.txt"), args.max_items)
    if not args.skip_dialogues:
        total += dump_russian_dialogues(dst("russian_dialogues", "russian_dialogues_chat.txt"), args.max_items)
    print(f"total written: {total / 1e6:.1f} MB across new files")


if __name__ == "__main__":
    main()
