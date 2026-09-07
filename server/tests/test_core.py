import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from server.common import CHAT_TEMPLATE, SPECIAL, extract, normalize, partition
from server.prepare import encode_record
from server.model import TasochkaForCausalLM
from server.data import Collator, Corpus


@pytest.fixture
def tok():
    b = Tokenizer(models.BPE())
    b.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    b.decoder = decoders.ByteLevel()
    b.train_from_iterator(['Привет мир! def test():\n    return "ok"'] * 20,
        trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    t = PreTrainedTokenizerFast(tokenizer_object=b, pad_token=SPECIAL[0], bos_token=SPECIAL[1],
        eos_token=SPECIAL[2], additional_special_tokens=SPECIAL[3:])
    t.chat_template = CHAT_TEMPLATE
    return t


def test_code_roundtrip(tok):
    text = 'def f(x):\n\treturn x != "ёж" and x ** 2 >= 5  # комментарий 🐈\n'
    assert normalize(text) == text
    assert tok.decode(tok.encode(text)) == text


def test_sft_mask_and_template(tok):
    r = {"messages": [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Привет мир!"}]}
    ids, labels = encode_record(r, tok)
    assert ids == tok.apply_chat_template(r['messages'], tokenize=True)
    supervised = tok.decode([x for x in labels if x != -100])
    assert supervised == "Привет мир!<|end|>\n"
    batch = Collator(tok.pad_token_id)([{"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)},
        {"input_ids": torch.tensor(ids[:-2]), "labels": torch.tensor(labels[:-2])}])
    assert batch['labels'][1, -1] == -100
    assert batch['attention_mask'][1, -1] == 0


def test_partition_groups_answers():
    source = dict(kind='sft', name='x', prompt_fields=['instruction'], answer_field='output')
    _, first = extract(dict(instruction='question', output='a'), source)
    _, second = extract(dict(instruction='question', output='b'), source)
    assert first == second
    assert partition(first, .01) == partition(second, .01)


def test_chunked_loss_and_gradients():
    torch.manual_seed(42)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True)
    reference = LlamaForCausalLM(cfg)
    model = TasochkaForCausalLM(cfg)
    model.load_state_dict(reference.state_dict())
    x = torch.randint(0, 64, (2, 150))  # Cross the loss chunk boundary.
    labels = x.clone(); labels[:, :15] = -100
    a = model(input_ids=x, labels=labels).loss
    b = reference(input_ids=x, labels=labels).loss
    torch.testing.assert_close(a, b)
    a.backward(); b.backward()
    for (n, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4, msg=n)


def test_long_windows_do_not_cross_documents(tmp_path):
    (tmp_path / 'meta.json').write_text('{}')
    np.arange(24, dtype='<u4').tofile(tmp_path / 'train.bin')
    np.array([0, 3, 24], dtype='<u8').tofile(tmp_path / 'train.idx')
    c = Corpus(tmp_path, 'train', 8, 'long')
    assert c.size == 2
    assert c.get(0)['input_ids'].tolist() == list(range(3, 11))
    assert c.get(1)['input_ids'].tolist() == list(range(11, 19))


def test_3b_parameter_count():
    cfg = LlamaConfig(**json.loads(Path('server/configs/model_3b.json').read_text()))
    with torch.device('meta'):
        model = TasochkaForCausalLM(cfg)
    count = sum(p.numel() for p in model.parameters())
    assert 2.9e9 < count < 3.1e9
    print('parameters:', count)


def test_yarn_128k_configuration():
    cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072,
        rope_scaling={'rope_type': 'yarn', 'factor': 32.0, 'original_max_position_embeddings': 4096})
    model = TasochkaForCausalLM(cfg)
    # Test actual rotary positions near 128k without quadratic full attention.
    x = torch.zeros(1, 2, 32)
    cos, sin = model.model.rotary_emb(x, torch.tensor([[131070, 131071]]))
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_interrupted_resume_matches_uninterrupted(tmp_path):
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    class StopAtTwo(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True
                control.should_save = True
            return control
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    data = [{'input_ids': torch.arange(12) % 32, 'labels': torch.arange(12) % 32} for _ in range(20)]
    def trainer(directory, callbacks=None):
        set_seed(42)
        args = TrainingArguments(output_dir=str(directory), max_steps=4, use_cpu=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=2, save_steps=2,
            report_to='none', disable_tqdm=True, learning_rate=.001, lr_scheduler_type='cosine')
        t = Trainer(model=TasochkaForCausalLM(cfg), args=args, train_dataset=data,
                    data_collator=Collator(0), callbacks=callbacks or [])
        t.model_accepts_loss_kwargs = False
        return t
    full = trainer(tmp_path / 'full'); full.train()
    interrupted = trainer(tmp_path / 'resume', [StopAtTwo()]); interrupted.train()
    resumed = trainer(tmp_path / 'resume'); resumed.train(resume_from_checkpoint=str(tmp_path / 'resume/checkpoint-2'))
    assert resumed.state.global_step == 4
    for p, q in zip(full.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
