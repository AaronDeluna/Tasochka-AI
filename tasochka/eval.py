"""Honest evaluation: every prompt is chosen to NOT appear verbatim in any
seed file (persona_seed.txt, self_awareness.txt). The goal is to measure
generalization from the main corpus, not memorization of hand-written Q&A.

For comparison: any answer that exactly matches a seed phrase means the
model is doing template lookup. Answers that look related but differently
worded mean the model actually learned something.

Usage:
    python -m tasochka.eval > eval_output.txt 2>&1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .generate import CHAT_STOPS, build_chat_prompt, generate_stream, load

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka"

# (category, label, prompt) — none of these phrasings or facts are in seed files.
PROMPTS = [
    # ── CHAT (greetings/openers NOT in seed) ──────────────────────
    ("CHAT",      "greet slang",                 "Здравствуй!"),
    ("CHAT",      "greet informal",              "Йоу, ты тут?"),
    ("CHAT",      "small-talk weekend",          "Сегодня воскресенье, ничего не хочется делать."),
    ("CHAT",      "ask-to-chat",                 "Поболтаем?"),

    # ── IDENTITY (paraphrased — tests generalization) ─────────────
    ("IDENTITY",  "identity slang",              "А ты что за зверь?"),
    ("IDENTITY",  "identity one-line",           "Опиши себя в одном предложении."),
    ("IDENTITY",  "identity third-person",       "Если бы я рассказывал друзьям о тебе, что бы я им сказал?"),
    ("IDENTITY",  "self-doubt",                  "Откуда ты знаешь что ты не человек?"),

    # ── FACT — capitals NOT in seed ────────────────────────────────
    ("FACT",      "capital poland",              "Какая столица Польши?"),
    ("FACT",      "capital brazil",              "Какая столица Бразилии?"),
    ("FACT",      "capital canada",              "Какая столица Канады?"),
    ("FACT",      "capital turkey",              "Какая столица Турции?"),

    # ── FACT — literature NOT in seed ─────────────────────────────
    ("FACT",      "author bulgakov",             "Кто написал «Мастера и Маргариту»?"),
    ("FACT",      "author tolstoy karenina",     "Кто написал «Анну Каренину»?"),
    ("FACT",      "writer chekhov",              "Расскажи кратко кто такой Антон Павлович Чехов."),
    ("FACT",      "writer gogol",                "Кто такой Николай Васильевич Гоголь?"),

    # ── FACT — science NOT in seed ────────────────────────────────
    ("FACT",      "ice melting",                 "При какой температуре плавится лёд?"),
    ("FACT",      "photosynthesis",              "Что такое фотосинтез?"),
    ("FACT",      "human chromosomes",           "Сколько хромосом у человека?"),
    ("FACT",      "speed of light",              "С какой скоростью движется свет?"),

    # ── REASONING — math above 12×12, problems not in seed ───────
    ("REASONING", "math two-digit add",          "Сколько будет 23 + 47?"),
    ("REASONING", "math two-digit sub",          "Сколько будет 100 - 37?"),
    ("REASONING", "math 13x8",                   "Сколько будет 13 умножить на 8?"),
    ("REASONING", "money word problem",          "У Игоря было 100 рублей, он потратил 37. Сколько у него осталось?"),
    ("REASONING", "day-of-week",                 "Если сегодня среда, какой день недели будет через 5 дней?"),
    ("REASONING", "age problem",                 "Брат старше сестры на 3 года. Брату 10 лет. Сколько лет сестре?"),

    # ── INSTRUCTION — new formats and topics ──────────────────────
    ("INSTR",     "five fruits",                 "Перечисли пять фруктов."),
    ("INSTR",     "birthday greeting",           "Напиши короткое поздравление с днём рождения."),
    ("INSTR",     "describe sea one word",       "Опиши море одним словом."),
    ("INSTR",     "list seasons",                "Маркированным списком назови четыре времени года."),

    # ── CREATIVE — novel genres/topics ────────────────────────────
    ("CREATIVE",  "haiku autumn",                "Напиши хокку про осень."),
    ("CREATIVE",  "cafe name",                   "Придумай название для уютной кофейни."),
    ("CREATIVE",  "ideal day",                   "Опиши свой идеальный день."),
    ("CREATIVE",  "couplet about cat",           "Сочини двустишие про кота."),

    # ── EMOTION — new situations ──────────────────────────────────
    ("EMOTION",   "interview nerves",            "Я очень нервничаю перед собеседованием. Что делать?"),
    ("EMOTION",   "headache",                    "У меня болит голова уже целый день."),
    ("EMOTION",   "loneliness",                  "Поговори со мной, мне одиноко."),

    # ── EDGE — new absurdities / unknowns ─────────────────────────
    ("EDGE",      "unknown person",              "Кто такой Тимофей Анатольевич Балалайкин из Урюпинска?"),
    ("EDGE",      "made-up term",                "Что такое квантовая интерпретанта Зюзина-Хрюндера?"),
    ("EDGE",      "impossible event",            "Когда люди впервые приземлились на Юпитер?"),
    ("EDGE",      "future fact",                 "Какие главные новости были в 5089 году?"),

    # ── CONTEXT — quick test of in-prompt memory ──────────────────
    ("CONTEXT",   "remember in prompt",          "Меня зовут Иван. Запомни моё имя. Как меня зовут?"),
    ("CONTEXT",   "word meaning",                "Что значит слово «парадигма»?"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--rep-penalty", type=float, default=1.18)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    model, tokenizer, cfg = load(args.checkpoint)
    print(f"# Tasochka AI honest evaluation ({len(PROMPTS)} prompts, none verbatim in seed)")
    print(f"# checkpoint: {args.checkpoint}")
    print(f"# vocab={tokenizer.vocab_size} ctx={cfg.context_length} "
          f"layers={cfg.num_layers} dim={cfg.embedding_dim}")
    print(f"# sampling: temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
          f"rep_penalty={args.rep_penalty} no_repeat_ngram={args.no_repeat_ngram} "
          f"seed={args.seed} max_new={args.max_new}")
    print()

    for i, (category, label, prompt) in enumerate(PROMPTS, 1):
        print(f"=========================================================")
        print(f"[{i:02d}/{len(PROMPTS)}] {category:<10} | {label}")
        print(f"USER: {prompt}")
        print(f"---------------------------------------------------------")
        full_prompt = build_chat_prompt(prompt)
        for piece in generate_stream(
            model, tokenizer, cfg, full_prompt,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p if args.top_p > 0 else None,
            rep_penalty=args.rep_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
            stop_strings=CHAT_STOPS,
        ):
            print(piece, end="", flush=True)
        print()
        print()


if __name__ == "__main__":
    main()
