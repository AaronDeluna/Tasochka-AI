"""Offline end-to-end CLI smoke on synthetic data; never a quality benchmark."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('server/smoke-output'))
    a = p.parse_args()
    if a.out.exists():
        p.error('Choose a fresh output directory')
    raw = a.out / 'raw'
    for name, kind in [('text', 'text'), ('dialogue', 'sft')]:
        d = raw / name; d.mkdir(parents=True)
        source = dict(name=name, kind=kind, weight=1.0)
        write_json(d / 'meta.json', dict(source=source, revision='synthetic-fixture'))
        for split in ['train', 'val']:
            with (d / f'{split}.jsonl').open('w') as f:
                for i in range(20):
                    row = {'text': (f'{split} документ {i}. Русский текст. Код: def f(x): return x + 1\n' * 600)} if kind == 'text' else {'messages': [
                        {'role': 'user', 'content': f'{split}: Сколько будет {i}+1?'},
                        {'role': 'assistant', 'content': f'Ответ: {i + 1}.'}]}
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tok = a.out / 'tokenizer'; tokens = a.out / 'tokens'
    def run(module, *args):
        subprocess.run([sys.executable, '-m', f'server.{module}', *map(str, args)], check=True)
    run('tokenizer', '--raw', raw, '--out', tok, '--vocab-size', 512)
    run('prepare', '--raw', raw, '--out', tokens, '--tokenizer', tok)
    config = a.out / 'tiny.json'
    write_json(config, dict(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0))
    common = ['--cpu', '--config', config, '--tokenizer', tok, '--data', tokens, '--context', 64,
              '--max-steps', 2, '--warmup-ratio', 0, '--accumulation', 2, '--save-steps', 1, '--eval-steps', 1, '--eval-limit', 2]
    run('train', *common, '--output', a.out / 'pretrain')
    run('train', *common, '--output', a.out / 'long', '--mode', 'long', '--context', 8192, '--max-steps', 1, '--accumulation', 1, '--init-from', a.out / 'pretrain/final')
    run('train', *common, '--output', a.out / 'sft', '--mode', 'sft', '--init-from', a.out / 'long/final')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.out / 'sft/final')
    tokenizer = AutoTokenizer.from_pretrained(a.out / 'sft/final')
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Привет'}], add_generation_prompt=True, return_tensors='pt')
    model.generate(ids, max_new_tokens=2, pad_token_id=0)
    print('PASS: tokenizer -> tokens -> pretrain -> 8k YaRN continuation -> SFT -> standard HF export -> generation')


if __name__ == '__main__':
    main()
