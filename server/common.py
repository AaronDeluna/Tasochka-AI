"""Shared, dependency-light validation and chat format."""
import hashlib
import json
from pathlib import Path

SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
CHAT_TEMPLATE = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    # Never strip punctuation, tabs, repeated spaces or code indentation.
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def extract(row, source):
    if source["kind"] == "sft":
        parts = [row.get(k, "") for k in source["prompt_fields"]]
        if not all(isinstance(x, str) for x in parts):
            raise ValueError(f"Invalid prompt schema in {source['name']}")
        answer = row[source["answer_field"]]
        if not isinstance(answer, str):
            raise ValueError("Answer must be a string")
        prompt = normalize("\n\n".join(x for x in parts if x and x.strip() != "<noinput>"))
        answer = normalize(answer)
        if not prompt.strip() or not answer.strip():
            return None
        # Partition by prompt, so alternate answers cannot leak to validation.
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}, prompt
    value = row[source["field"]]
    if not isinstance(value, str):
        raise ValueError(f"Invalid text schema in {source['name']}")
    value = normalize(value)
    if len(value.strip()) < 200 or value.count("\ufffd") > len(value) * 0.001:
        return None
    if any(s in value for s in SPECIAL):
        return None
    return {"text": value}, value


def partition(key, fraction):
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    return "val" if int(digest(key)[:16], 16) / 2**64 < fraction else "train"
