"""Train our own byte-level BPE on a balanced, train-only corpus sample."""
import argparse
import json
from itertools import zip_longest
from pathlib import Path
from .common import CHAT_TEMPLATE, SPECIAL, read_json


def samples(path, max_chars):
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r.get("text") or "\n".join(m["content"] for m in r["messages"])
            # Bound each Rust trainer sample without damaging the actual corpus.
            for start in range(0, len(text), 16384):
                value = text[start:start + min(16384, max_chars - count)]
                if value:
                    yield value
                    count += len(value)
                if count >= max_chars:
                    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("server/data/raw"))
    p.add_argument("--out", type=Path, default=Path("server/data/tokenizer"))
    p.add_argument("--vocab-size", type=int, default=48000)
    p.add_argument("--sample-chars", type=int, default=500_000_000)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("Tokenizer output exists; changing it invalidates all tokens and weights")
    paths = sorted(a.raw.glob("*/train.jsonl"))
    if not paths or a.vocab_size < 263 or a.sample_chars <= 0:
        p.error("Need training files, vocab >= 263 and a positive character budget")
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    metadata = [read_json(path.parent / "meta.json")["source"] for path in paths]
    groups = ["sft" if s["kind"] == "sft" else "pretrain" for s in metadata]
    totals = {group: sum(s["weight"] for s, g in zip(metadata, groups) if g == group) for group in set(groups)}
    budgets = [max(1, int(a.sample_chars * (0.2 if g == "sft" else 0.8) * s["weight"] / totals[g]))
               for s, g in zip(metadata, groups)]
    iterator = (x for group in zip_longest(*(samples(path, budget) for path, budget in zip(paths, budgets)))
                for x in group if x is not None)
    backend.train_from_iterator(iterator, trainers.BpeTrainer(vocab_size=a.vocab_size, min_frequency=2,
        special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token=SPECIAL[0], bos_token=SPECIAL[1],
        eos_token=SPECIAL[2], additional_special_tokens=SPECIAL[3:], model_max_length=131072,
        clean_up_tokenization_spaces=False)
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(a.out)
    print(f"Saved tokenizer: {len(tok)} tokens")


if __name__ == "__main__":
    main()
