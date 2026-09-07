"""Standard Llama weights with checkpointed, chunked training cross entropy."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class TasochkaForCausalLM(LlamaForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("return_dict", None)
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True, **kwargs).last_hidden_state
        hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        target = labels[:, 1:].reshape(-1)
        count = (target != -100).sum()
        if count.item() == 0:
            raise ValueError("Batch has no supervised target tokens")
        total = hidden.new_zeros((), dtype=torch.float32)

        def chunk_loss(h, y):
            return F.cross_entropy(self.lm_head(h).float(), y, ignore_index=-100, reduction="sum")

        # Avoid materializing [batch, 128k, 48k] logits. Checkpointing is required:
        # simply splitting CE retains every chunk's logits until backward.
        for start in range(0, target.numel(), 256):
            h, y = hidden[start:start + 256], target[start:start + 256]
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, h, y, use_reentrant=False)
            else:
                total = total + chunk_loss(h, y)
        return CausalLMOutputWithPast(loss=total / count)
