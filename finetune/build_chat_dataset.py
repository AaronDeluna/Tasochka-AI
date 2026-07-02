"""Собирает разговорный датасет для QLoRA-дообучения.

Смешивает:
  1. Реальные русские диалоги из russian-train-dataset/*.txt
     (формат <user>...</user><assistant>...</assistant>, multi-turn поддержан)
  2. Persona-примеры Тасочки из finetune/data/ (генерятся build_persona.py)

Фильтры качества (практика Qwen/DeepSeek — маленький чистый SFT-датасет
работает лучше большого грязного):
  - только полные пары user/assistant, диалог начинается с user
  - длина диалога 40..2700 символов (влезает в max_seq_length=1024 токенов без обрезки)
  - ответ ассистента 5..2500 символов
  - доля кириллицы в ответах >= 35%
  - выбрасываются диалоги, где ассистент называет себя чужой моделью
    (Сайга, GPT, YandexGPT, ...) — конфликт с персоной
  - дедупликация по первому вопросу+ответу

Запуск:
    ./.venv/bin/python finetune/build_chat_dataset.py
Создаёт:
    finetune/data_chat/train.jsonl
    finetune/data_chat/valid.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "russian-train-dataset"
PERSONA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "data_chat"

# Сколько диалогов брать из каждого источника (reservoir sampling).
SOURCES = {
    "saiga_chat.txt": 3500,             # отборные (score>=7) GPT-4 диалоги
    "grandmaster_chat.txt": 1500,       # сложные инструкции
    "russian_dialogues_chat.txt": 500,  # живая разговорная речь
}

# Ассистент не должен называть себя чужой моделью.
IDENTITY_BLACKLIST = re.compile(
    r"(?i)\b(сайга|saiga|gpt-?[345o]?|chatgpt|openai|яндекс ?gpt|yandexgpt|"
    r"gigachat|гигачат|llama|лама от|claude|anthropic|gemma|gemini|"
    r"разработан[аы]? (компанией )?(google|сбер|яндекс|openai|meta))\b"
)

CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")


def iter_conversations(path: Path, max_bytes: int = 400_000_000):
    """Стримит диалоги из txt-файла с разметкой <user>/<assistant>.

    Граница диалога — пустая строка вне тегов (multi-turn пары идут подряд
    без пустой строки, см. download_datasets.py).
    """
    conv: list[dict] = []
    role = None
    buf: list[str] = []
    read = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            read += len(line)
            s = line.rstrip("\n")
            if s == "<user>":
                role, buf = "user", []
            elif s == "</user>":
                if role == "user":
                    conv.append({"role": "user", "content": "\n".join(buf).strip()})
                role = None
            elif s == "<assistant>":
                role, buf = "assistant", []
            elif s == "</assistant>":
                if role == "assistant":
                    conv.append({"role": "assistant", "content": "\n".join(buf).strip()})
                role = None
            elif role is not None:
                buf.append(s)
            elif not s.strip() and conv:
                yield conv
                conv = []
                if read > max_bytes:
                    return
    if conv:
        yield conv


def valid_conversation(conv: list[dict]) -> bool:
    if len(conv) < 2 or len(conv) % 2 != 0:
        return False
    total = 0
    for i, msg in enumerate(conv):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected or not msg["content"]:
            return False
        total += len(msg["content"])
    # ~2.7 символа на токен у Gemma-токенизатора на русском: 2700 символов
    # надёжно влезают в max_seq_length=1024 без обрезки.
    if not 40 <= total <= 2700:
        return False
    for msg in conv[1::2]:  # ответы ассистента
        text = msg["content"]
        if not 5 <= len(text) <= 2500:
            return False
        letters = sum(1 for ch in text if ch.isalpha())
        if letters and len(CYRILLIC_RE.findall(text)) / letters < 0.35:
            return False
        if IDENTITY_BLACKLIST.search(text):
            return False
    return True


def conv_key(conv: list[dict]) -> str:
    norm = (conv[0]["content"] + "|" + conv[1]["content"]).lower()
    norm = re.sub(r"\s+", " ", norm)[:400]
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def sample_source(path: Path, cap: int, rng: random.Random, seen: set) -> list[list[dict]]:
    """Reservoir sampling: равномерная выборка cap диалогов из всего файла."""
    reservoir: list[list[dict]] = []
    n_seen = 0
    for conv in iter_conversations(path):
        if not valid_conversation(conv):
            continue
        key = conv_key(conv)
        if key in seen:
            continue
        seen.add(key)
        n_seen += 1
        if len(reservoir) < cap:
            reservoir.append(conv)
        else:
            j = rng.randrange(n_seen)
            if j < cap:
                reservoir[j] = conv
    print(f"  {path.name}: прошло фильтры {n_seen}, взято {len(reservoir)}")
    return reservoir


def load_persona(rng: random.Random) -> tuple[list, list]:
    """Persona-примеры из build_persona.py (генерятся, если ещё нет)."""
    train_path = PERSONA_DIR / "train.jsonl"
    valid_path = PERSONA_DIR / "valid.jsonl"
    if not train_path.exists():
        print("persona-датасет не найден, генерирую через build_persona.py ...")
        runpy.run_path(str(Path(__file__).resolve().parent / "build_persona.py"),
                       run_name="__main__")

    def read(p: Path) -> list:
        out = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line)["messages"])
        return out

    return read(train_path), read(valid_path)


# Пресеты: доля балласта (реальные диалоги) и повторов персоны.
#   full    — большой смешанный датасет (учит и стиль, и персону)
#   persona — фокус на личности: мало балласта (чтобы не забыть речь) + больше
#             повторов персоны. Балласт тут НЕ учит QA (модель и так умеет),
#             он только не даёт переобучиться на persona-фразах.
PRESETS = {
    "full":    {"ballast_scale": 1.0, "persona_mult": 2, "out": "data_chat"},
    "persona": {"ballast_scale": 0.28, "persona_mult": 3, "out": "data_persona"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="full",
                       help="full — большой смешанный; persona — фокус на личности (рекомендуется).")
    parser.add_argument("--persona-mult", type=int, default=None,
                       help="Сколько раз повторить persona-примеры (по умолчанию из пресета).")
    parser.add_argument("--ballast-scale", type=float, default=None,
                       help="Множитель числа реальных диалогов (по умолчанию из пресета).")
    parser.add_argument("--out", type=str, default=None,
                       help="Папка вывода (по умолчанию из пресета).")
    parser.add_argument("--n-valid", type=int, default=250,
                       help="Сколько реальных диалогов отложить в valid.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    persona_mult = args.persona_mult if args.persona_mult is not None else preset["persona_mult"]
    ballast_scale = args.ballast_scale if args.ballast_scale is not None else preset["ballast_scale"]
    out_dir = Path(__file__).resolve().parent / (args.out or preset["out"])

    rng = random.Random(args.seed)
    seen: set = set()

    print(f"Пресет: {args.preset} (балласт x{ballast_scale}, persona x{persona_mult})")
    print("Собираю диалоги из источников:")
    dialogs: list[list[dict]] = []
    for name, cap in SOURCES.items():
        path = DATASET_DIR / name
        if not path.exists():
            print(f"  [пропуск] {name} не найден в {DATASET_DIR}")
            continue
        dialogs.extend(sample_source(path, max(1, int(cap * ballast_scale)), rng, seen))

    if not dialogs:
        raise SystemExit(f"Не найдено ни одного источника в {DATASET_DIR} — "
                         f"сначала запусти download_datasets.py")

    rng.shuffle(dialogs)
    n_valid = min(args.n_valid, len(dialogs) // 10)
    valid_real = dialogs[:n_valid]
    train_real = dialogs[n_valid:]

    persona_train, persona_valid = load_persona(rng)
    train = train_real + persona_train * persona_mult
    valid = valid_real + persona_valid
    rng.shuffle(train)
    rng.shuffle(valid)

    out_dir.mkdir(exist_ok=True)

    def dump(path: Path, items: list) -> None:
        with path.open("w", encoding="utf-8") as f:
            for messages in items:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    dump(out_dir / "train.jsonl", train)
    dump(out_dir / "valid.jsonl", valid)

    persona_share = 100.0 * len(persona_train) * persona_mult / max(1, len(train))
    print(f"\nTrain: {len(train)} диалогов ({len(train_real)} реальных балласт + "
          f"{len(persona_train)}x{persona_mult} persona, {persona_share:.0f}% persona)")
    print(f"Valid: {len(valid)} диалогов")
    print(f"→ {out_dir / 'train.jsonl'}")
    print(f"→ {out_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
