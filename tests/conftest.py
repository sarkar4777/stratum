"""
Shared fixtures. The important one builds a complete tiny base model from
scratch - a byte-level tokenizer trained on the example data and a random
two-layer Llama of a few hundred KB - so the full train/merge/eval pipeline
runs on CPU in seconds, with no downloads and no GPU.
"""
import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


def _corpus_lines():
    lines = ["User:", "Assistant:", "System:", "Extract the total", "Classify"]
    for name in ("extract.jsonl", "classify.jsonl"):
        p = EXAMPLES / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    lines.append(row["prompt"])
                    lines.append(row["response"])
    return lines


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """Build and save a tiny random base model plus tokenizer. Returns its path."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-base")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400,
                                  special_tokens=["[UNK]", "</s>", "<pad>"])
    tok.train_from_iterator(_corpus_lines(), trainer)

    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="</s>",
                                   pad_token="<pad>", unk_token="[UNK]")
    fast.save_pretrained(str(out))

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        pad_token_id=fast.pad_token_id,
        eos_token_id=fast.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(str(out))
    return str(out)


@pytest.fixture()
def skill_file(tmp_path):
    """A small training file in the standard prompt/response shape."""
    rows = [
        {"prompt": "Extract the total from this invoice: 'Total: $88'",
         "response": '{"total": 88}'},
        {"prompt": "Extract the total from this invoice: 'Amount due: $250'",
         "response": '{"total": 250}'},
        {"prompt": "Extract the total from this invoice: 'You owe: $5'",
         "response": '{"total": 5}'},
        {"prompt": "Extract the total from this invoice: 'Sum: 12 dollars'",
         "response": '{"total": 12}'},
    ]
    p = tmp_path / "skill.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)
