"""Inspect architecture/data and measure hardware availability before allocation."""
import argparse
from pathlib import Path
from .common import file_digest, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('server/data/tokens'))
    p.add_argument('--tokenizer', type=Path, default=Path('server/data/tokenizer'))
    p.add_argument('--manifest', default='server/configs/datasets.json')
    p.add_argument('--config', default='server/configs/model_3b.json')
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    import torch
    from transformers import LlamaConfig
    from .model import TasochkaForCausalLM
    from .data import Corpus
    with torch.device('meta'):
        model = TasochkaForCausalLM(LlamaConfig(**read_json(a.config)))
    count = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {count:,}; BF16 weights: {count * 2 / 2**30:.2f} GiB')
    print(f'Approximate mixed-precision Adam model state before sharding: {count * 16 / 2**30:.1f} GiB (activations extra)')
    if not a.cpu:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise SystemExit('CUDA/BF16 unavailable')
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {prop.name}, {prop.total_memory / 2**30:.1f} GiB')
    fingerprint = file_digest(a.tokenizer / 'tokenizer.json')
    for s in read_json(a.manifest)['sources']:
        path = a.data / s['name']
        meta = read_json(path / 'meta.json')
        if meta['tokenizer_sha256'] != fingerprint:
            raise SystemExit(f'Tokenizer mismatch: {s["name"]}')
        if meta['download'].get('max_records') is not None:
            raise SystemExit(f'Download smoke limit still set for {s["name"]}')
        if meta['source'] != s:
            raise SystemExit(f'Manifest changed: {s["name"]}')
        mode = 'sft' if s['kind'] == 'sft' else 'pretrain'
        for split in ('train', 'val'):
            corpus = Corpus(path, split, 4096, mode)
            print(s['name'], split, 'tokens=', len(corpus.ids), 'usable examples=', corpus.size)
        if s['name'] == 'long_books':
            for context in (16384, 32768, 131072):
                for split in ('train', 'val'):
                    c = Corpus(path, split, context, 'long')
                    print('long_books', split, context, 'contiguous windows=', c.size)
    print('Data checks passed. This does NOT certify GPU memory fit or model quality. Benchmark each context on the target GPUs.')


if __name__ == '__main__':
    main()
