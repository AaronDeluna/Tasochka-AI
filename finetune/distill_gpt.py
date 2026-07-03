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


# Новые модели (gpt-5.x, o1/o3/o4) хотят max_completion_tokens и часто фиксированную
# температуру. Стартуем с этого варианта для них, а если API ругнётся на параметр —
# на лету переключаемся. Флаги общие на весь прогон, чтобы не долбить 400 повторно.
_NEWGEN = MODEL.startswith(("gpt-5", "o1", "o3", "o4"))
_params = {
    "token_key": "max_completion_tokens" if _NEWGEN else "max_tokens",
    "send_temp": not _NEWGEN,
}


def _payload(question: str) -> bytes:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        _params["token_key"]: MAX_TOKENS,
    }
    if _params["send_temp"]:
        body["temperature"] = TEMP
    return json.dumps(body).encode("utf-8")


def ask(question: str, retries: int = 5) -> str | None:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=_payload(question), headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            msg = e.read().decode("utf-8", "replace")
            # Авто-подстройка под требования модели по параметрам.
            low = msg.lower()
            fixed = False
            if "max_tokens" in low and "max_completion_tokens" in low:
                _params["token_key"] = "max_completion_tokens"; fixed = True
            if "temperature" in low and ("unsupported" in low or "does not support" in low
                                          or "only the default" in low):
                _params["send_temp"] = False; fixed = True
            if fixed and attempt < retries - 1:
                continue
            print(f"\n[HTTP {e.code}] {question[:50]}... → {msg[:200]}")
            return None
        except Exception as e:  # noqa: BLE001 — SSL/сеть/парсинг: ретраим, не роняем прогон
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"\n[network] {question[:50]}... → {type(e).__name__}: {e}")
            return None
    return None


def list_models() -> None:
    """Печатает доступные ключу модели (для выбора правильного имени)."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    ids = sorted(m["id"] for m in data.get("data", []))
    chat = [m for m in ids if any(k in m for k in ("gpt", "o1", "o3", "o4", "chat"))]
    print("Доступные chat-модели твоего ключа:")
    for m in chat:
        print("  ", m)
    print(f"\nВыбери нужную и запусти:  export OPENAI_MODEL=\"<имя>\"  затем distill_gpt.py")


def main() -> None:
    if not API_KEY:
        sys.exit("Нет ключа. Сначала: export OPENAI_API_KEY=\"sk-...\"  затем запусти снова.")
    if "--list-models" in sys.argv:
        list_models()
        return
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
            try:
                q, ans = fut.result()
            except Exception as e:  # noqa: BLE001 — не роняем весь прогон из-за одного вопроса
                print(f"\n[skip] {type(e).__name__}: {e}")
                fail += 1
                continue
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
