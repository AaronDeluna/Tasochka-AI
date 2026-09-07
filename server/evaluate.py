"""Save Russian/code generations and optional exact-budget needle retrieval tests."""
import argparse
import json
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', default='server/configs/eval_prompts.jsonl')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--needle-contexts', nargs='*', type=int, default=[])
    p.add_argument('--haystack', type=Path, help='Held-out UTF-8 text; required for needle tests')
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists; select a new report name')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    device = 'cpu' if a.cpu else 'cuda'
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        torch_dtype=torch.float32 if a.cpu else torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    if not a.cpu:
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    def generate(ids, budget):
        if len(ids) + budget > model.config.max_position_embeddings:
            raise ValueError('Prompt plus output exceeds checkpoint context')
        inputs = torch.tensor([ids], device=device)
        with torch.inference_mode():
            out = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=budget,
                do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id,
                eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids('<|end|>')])
        return tok.decode(out[0, len(ids):], skip_special_tokens=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', encoding='utf-8') as f:
        for line in Path(a.prompts).read_text().splitlines():
            r = json.loads(line)
            ids = tok.apply_chat_template(r['messages'], add_generation_prompt=True)
            r['answer'] = generate(ids, 256)
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
        if a.needle_contexts:
            if not a.haystack:
                p.error('--haystack is required')
            haystack = tok.encode(a.haystack.read_text(encoding='utf-8'), add_special_tokens=False)
            rng = random.Random(917)
            for context in a.needle_contexts:
                for depth in (0.1, 0.5, 0.9):
                    secret = str(rng.randrange(10000000, 99999999))
                    prefix = tok.encode('<|user|>\nПрочитай документ и найди секретный номер.\n', add_special_tokens=False)
                    suffix = tok.encode('\nКакой секретный номер указан? Верни только цифры.<|end|>\n<|assistant|>\n', add_special_tokens=False)
                    needle = tok.encode('\nСекретный номер: ' + secret + '.\n', add_special_tokens=False)
                    n = context - 32 - len(prefix) - len(suffix) - len(needle)
                    if n <= 0 or len(haystack) < n:
                        raise ValueError('Need a longer held-out haystack or a valid context budget')
                    offset = int(n * depth)
                    ids = prefix + haystack[:offset] + needle + haystack[offset:n] + suffix
                    answer = generate(ids, 32)
                    f.write(json.dumps(dict(category='needle', context=context, prompt_tokens=len(ids),
                        depth=depth, expected=secret, answer=answer, exact_match=answer.strip() == secret), ensure_ascii=False) + '\n')
                    f.flush()
    print(f'Saved {a.out}. Manually review Russian/code answers; needle retrieval alone does not certify 128k reasoning.')


if __name__ == '__main__':
    main()
