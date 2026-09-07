"""From-scratch 3B training, progressive context extension and masked SFT."""
import argparse
import math
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, LlamaConfig, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from .common import file_digest, read_json, write_json
from .data import Collator, Mixture
from .model import TasochkaForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pretrain", "long", "sft"], default="pretrain")
    p.add_argument("--config", default="server/configs/model_3b.json")
    p.add_argument("--tokenizer", default="server/data/tokenizer")
    p.add_argument("--data", default="server/data/tokens")
    p.add_argument("--output", required=True)
    p.add_argument("--init-from", help="Local completed previous stage; never downloads pretrained weights")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=60_000_000_000)
    p.add_argument("--max-steps", type=int, help="Overrides token budget, useful for smoke tests/SFT")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.01)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--eval-limit", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", nargs="+")
    p.add_argument("--deepspeed")
    p.add_argument("--cpu", action="store_true", help="Tiny-model tests only")
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    a = p.parse_args()
    if a.context < 2 or a.context > 131072 or min(a.tokens, a.batch_size, a.accumulation) < 1:
        p.error("Invalid context, token budget or batch size")
    if a.max_steps is not None and a.max_steps < 1:
        p.error("max-steps must be positive")
    if not a.cpu and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        p.error("CUDA GPU with BF16 support required; --cpu is only for tiny tests")
    if a.context > 4096 and a.mode == "pretrain":
        p.error("Start at <=4096 tokens; use --mode long with a previous stage")
    if a.mode != "pretrain" and not a.init_from:
        p.error("Long-context/SFT stages require --init-from")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    batch_tokens = a.context * a.batch_size * a.accumulation * world
    steps = a.max_steps or math.ceil(a.tokens / batch_tokens)
    output = Path(a.output)
    fingerprint = file_digest(Path(a.tokenizer) / "tokenizer.json")
    data_fingerprints = {str(f): file_digest(f) for f in sorted(Path(a.data).glob("*/meta.json"))}
    signature = {k: v for k, v in vars(a).items() if k not in ("resume", "local_rank")}
    signature.update(world_size=world, tokenizer_sha256=fingerprint,
                     data=data_fingerprints, model_config=read_json(a.config))
    checkpoint_path = None
    if a.resume:
        if not (output / "run.json").exists() or read_json(output / "run.json") != signature:
            p.error("Resume configuration/data/world size differs from saved run")
        checkpoint_path = get_last_checkpoint(str(output))
        if not checkpoint_path:
            p.error("No checkpoint to resume")
    elif output.exists() and any(output.iterdir()):
        p.error("Output exists: use --resume or choose a new directory")
    set_seed(a.seed)
    # Must construct TrainingArguments before loading/initializing a ZeRO-3 model.
    training = TrainingArguments(output_dir=str(output), max_steps=steps,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.accumulation, learning_rate=a.lr,
        lr_scheduler_type="cosine", lr_scheduler_kwargs={"num_cycles": 0.5},
        warmup_ratio=a.warmup_ratio, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95,
        max_grad_norm=1.0, bf16=not a.cpu, tf32=not a.cpu, use_cpu=a.cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", logging_steps=1 if a.cpu else 10,
        eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=3, save_safetensors=True, prediction_loss_only=True,
        dataloader_num_workers=0, dataloader_pin_memory=not a.cpu,
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        seed=a.seed, data_seed=a.seed, report_to="none", deepspeed=a.deepspeed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    if a.init_from:
        previous = Path(a.init_from)
        if not (previous / "stage_complete.json").exists():
            p.error("Previous stage must be a completed server/train.py export")
        if file_digest(previous / "tokenizer.json") != fingerprint:
            p.error("Previous-stage tokenizer mismatch")
        cfg = LlamaConfig.from_pretrained(previous, local_files_only=True)
    else:
        cfg = LlamaConfig(**read_json(a.config))
        if len(tok) > cfg.vocab_size:
            p.error("Tokenizer exceeds model vocabulary")
    if a.mode == "long":
        if a.context <= cfg.max_position_embeddings:
            p.error("Long stage must extend the previous context")
        cfg.rope_scaling = {"rope_type": "yarn", "factor": a.context / 4096.0,
                            "original_max_position_embeddings": 4096}
        cfg.max_position_embeddings = a.context
    elif a.context > cfg.max_position_embeddings:
        p.error("Sequence exceeds model context")
    cfg.use_cache = False
    cfg._attn_implementation = "sdpa"
    # Never permit the quadratic-memory math attention fallback on a server.
    if not a.cpu:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    if a.init_from:
        model = TasochkaForCausalLM.from_pretrained(a.init_from, config=cfg,
                    torch_dtype=torch.bfloat16 if not a.cpu else torch.float32, local_files_only=True)
    else:
        model = TasochkaForCausalLM(cfg)
    train_data = Mixture(a.data, "train", a.context, a.mode, fingerprint, a.seed, a.sources)
    val_data = Mixture(a.data, "val", a.context, a.mode, fingerprint, a.seed + 1, a.sources, a.eval_limit)
    trainer = Trainer(model=model, args=training, processing_class=tok,
                      train_dataset=train_data, eval_dataset=val_data, data_collator=Collator(tok.pad_token_id))
    # Chunked loss normalizes each microbatch itself. Let Trainer apply accumulation scaling.
    trainer.model_accepts_loss_kwargs = False
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run.json", signature)
        print({"parameters": sum(getattr(x, "ds_numel", x.numel()) for x in model.parameters()),
               "optimizer_steps": steps, "nominal_tokens_per_step": batch_tokens,
               "nominal_tokens": steps * batch_tokens,
               "source_windows": {c.meta["source"]["name"]: c.size for c in train_data.corpora}}, flush=True)
    trainer.train(resume_from_checkpoint=checkpoint_path)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    final = output / "final"
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tok.save_pretrained(final)
        write_json(final / "stage_complete.json", {"mode": a.mode, "context": cfg.max_position_embeddings,
            "trained_sequence_length": a.context, "global_step": trainer.state.global_step,
            "metrics": metrics, "quality_certified": False})


if __name__ == "__main__":
    main()
