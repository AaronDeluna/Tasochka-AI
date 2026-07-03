"""Дистилляция: гонит Java-вопросы через OpenAI API и собирает датасет.

Учитель (GPT) отвечает по-русски, код — только на Java. Ответы складываются в
JSONL формата {"messages": [{user}, {assistant}]}, готовый для train_persona.

Ключ берётся ТОЛЬКО из окружения (в код не хардкодим):
    export OPENAI_API_KEY="sk-..."
    ./.venv/bin/python finetune/distill_gpt.py

Настройки через окружение:
    OPENAI_MODEL       модель учителя (по умолчанию gpt-4o)
    DISTILL_WORKERS    параллельных запросов (по умолчанию 6)
    DISTILL_MAX_TOKENS максимум токенов ответа (по умолчанию 900)
    DISTILL_TEMP       температура (по умолчанию 0.3)

Особенности:
  - Resume: уже отвеченные вопросы пропускаются (можно прервать и продолжить).
  - Ретраи с backoff на 429/5xx/таймаут.
  - Пишет инкрементально — прогресс не теряется при обрыве.

Зависимостей нет — только стандартная библиотека (urllib).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "distill" / "questions.txt"
OUT = HERE / "distill" / "java_qa.jsonl"

API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
WORKERS = int(os.environ.get("DISTILL_WORKERS", "6"))
MAX_TOKENS = int(os.environ.get("DISTILL_MAX_TOKENS", "900"))
TEMP = float(os.environ.get("DISTILL_TEMP", "0.3"))
ENDPOINT = "https://api.openai.com/v1/chat/completions"

SYSTEM = (
    "Ты — опытный Java-разработчик и наставник. Отвечай на русском языке. "
    "Весь код пиши ТОЛЬКО на Java — никогда не на Python и не на других языках. "
    "Отвечай по существу, корректно и без воды. Если просят написать метод или "
    "класс — дай компилируемый Java-код и короткое пояснение. Не выдумывай факты; "
    "если не уверен — скажи об этом честно."
)

_write_lock = threading.Lock()


def load_done() -> set[str]:
    """Вопросы, на которые ответ уже есть в выходном файле (для resume)."""
    done: set[str] = set()
    if OUT.exists():
        with OUT.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["messages"][0]["content"])
                except Exception:
                    continue
    return done


def ask(question: str, retries: int = 5) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            msg = e.read().decode("utf-8", "replace")[:200]
            print(f"\n[HTTP {e.code}] {question[:50]}... → {msg}")
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"\n[network] {question[:50]}... → {e}")
            return None
    return None


def main() -> None:
    if not API_KEY:
        sys.exit("Нет ключа. Сначала: export OPENAI_API_KEY=\"sk-...\"  затем запусти снова.")
    if not QUESTIONS.exists():
        sys.exit(f"Нет файла вопросов {QUESTIONS} — сначала запусти distill_questions.py")

    questions = [q.strip() for q in QUESTIONS.read_text(encoding="utf-8").splitlines() if q.strip()]
    done = load_done()
    todo = [q for q in questions if q not in done]
    print(f"Модель-учитель: {MODEL} | воркеров: {WORKERS}")
    print(f"Всего вопросов: {len(questions)} | уже готово: {len(done)} | осталось: {len(todo)}")
    if not todo:
        print("Всё уже собрано. Датасет:", OUT)
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = 0
    fail = 0

    def work(q: str):
        ans = ask(q)
        return q, ans

    with OUT.open("a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(work, q) for q in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            q, ans = fut.result()
            if ans:
                rec = {"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ]}
                with _write_lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                ok += 1
            else:
                fail += 1
            if i % 20 == 0 or i == len(todo):
                el = time.time() - started
                eta = el / i * (len(todo) - i)
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}  "
                      f"({el/60:.1f} мин, осталось ~{eta/60:.0f} мин)")

    print(f"\nГотово: {ok} ответов записано, {fail} ошибок → {OUT}")
    print(f"Дальше: bash finetune/train_persona_code.sh")


if __name__ == "__main__":
    main()
