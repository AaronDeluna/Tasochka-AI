"""Interactive generation with KV cache and an explicit context budget."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    device = "cpu" if a.cpu else "cuda"
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        torch_dtype=torch.float32 if a.cpu else torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    messages = [{"role": "system", "content": "Ты — Тасочка, полезный русскоязычный ассистент."}]
    while True:
        try:
            prompt = input("Вы: ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() in ("exit", "quit"):
            break
        if prompt.strip() == "/reset":
            messages = messages[:1]
            continue
        messages.append({"role": "user", "content": prompt})
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        if ids.shape[-1] + a.max_new_tokens > model.config.max_position_embeddings:
            messages.pop()
            print("Контекст заполнен. /reset — начать новый диалог.")
            continue
        with torch.inference_mode():
            result = model.generate(ids, attention_mask=torch.ones_like(ids), use_cache=True,
                max_new_tokens=a.max_new_tokens, do_sample=a.temperature > 0,
                **({"temperature": a.temperature, "top_p": 0.9} if a.temperature > 0 else {}),
                eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|end|>")], pad_token_id=tok.pad_token_id)
        answer = tok.decode(result[0, ids.shape[-1]:], skip_special_tokens=True)
        print("Тасочка:", answer)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
