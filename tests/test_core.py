"""Core correctness tests for STRATUM. Run: python -m pytest tests/ -v"""
import json, tempfile, os
import torch
from safetensors.torch import save_file
from stratum.muon import newton_schulz, Muon, split_params_for_muon
from stratum.merge import extract_deltas, merge
from stratum.data import build_example, load_jsonl
from stratum.evaluate import score_contains, score_exact, score_json_field


def test_orthogonalization_flattens():
    torch.manual_seed(0)
    s = torch.linalg.svdvals(newton_schulz(torch.randn(8, 8)).float())
    assert s.max() < 1.6 and s.min() > 0.3


def test_newton_schulz_handles_zero():
    z = torch.zeros(4, 4)
    out = newton_schulz(z)
    assert torch.isfinite(out).all()


def test_muon_reduces_loss():
    torch.manual_seed(1)
    W = torch.nn.Parameter(torch.randn(8, 8)); tgt = torch.randn(8, 8)
    opt = Muon([W], lr=0.1); first = None
    for i in range(60):
        opt.zero_grad(); loss = ((W - tgt) ** 2).mean(); loss.backward(); opt.step()
        if i == 0: first = loss.item()
    assert loss.item() < first * 0.5


def test_muon_guards_nan_gradient():
    W = torch.nn.Parameter(torch.randn(4, 4))
    opt = Muon([W]); before = W.detach().clone()
    W.grad = torch.full((4, 4), float("nan"))
    opt.step() # must not poison W
    assert torch.isfinite(W).all()
    assert torch.allclose(W.detach(), before) # skipped, unchanged


def test_muon_rejects_non_2d():
    v = torch.nn.Parameter(torch.randn(4)); v.grad = torch.randn(4)
    try:
        Muon([v]).step(); assert False
    except ValueError:
        pass


def test_extract_deltas_math():
    d = tempfile.mkdtemp()
    r, alpha = 8, 16
    A, B = torch.randn(r, 16), torch.randn(16, r)
    save_file({
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B,
    }, os.path.join(d, "adapter_model.safetensors"))
    json.dump({"r": r, "lora_alpha": alpha}, open(os.path.join(d, "adapter_config.json"), "w"))
    deltas = extract_deltas(d)
    key = "model.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(deltas[key], (B @ A) * (alpha / r), atol=1e-5)


def test_linear_merge_is_addition():
    a = {"w": torch.ones(4, 4)}; b = {"w": torch.ones(4, 4) * 2}
    assert torch.allclose(merge("linear", [a, b], [1.0, 1.0])["w"], torch.ones(4, 4) * 3)


def test_all_merge_methods_run():
    a = {"w": torch.randn(6, 6)}; b = {"w": torch.randn(6, 6)}
    for method in ["linear", "ties", "dare"]:
        assert "w" in merge(method, [a, b])


def test_merge_rejects_bad_method():
    try:
        merge("nope", [{"w": torch.randn(4, 4)}]); assert False
    except ValueError:
        pass


def test_loss_mask_covers_prompt_only():
    class FakeTok:
        pad_token_id = 0; eos_token = "</s>"
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            t = "".join(m["content"] for m in msgs)
            return t + ("" if add_generation_prompt else "")
        def __call__(self, text, **kw):
            return {"input_ids": [ord(c) % 50 for c in text][:kw.get("max_length", 999)]}
    tok = FakeTok()
    ids, labels = build_example(tok, "AAAA", "BBB", None, 64)
    masked = sum(1 for l in labels if l == -100)
    assert masked >= 4 # at least the prompt is masked
    assert masked < len(labels) # but not everything


def test_load_jsonl_validates():
    d = tempfile.mkdtemp(); p = os.path.join(d, "x.jsonl")
    open(p, "w").write('{"prompt": "a", "response": "b"}\n')
    rows = load_jsonl(p, ("prompt", "response"))
    assert len(rows) == 1
    open(p, "w").write('{"prompt": "a"}\n') # missing response
    try:
        load_jsonl(p, ("prompt", "response")); assert False
    except ValueError:
        pass


def test_scorers():
    assert score_contains("the total is 88", "88") == 1.0
    assert score_exact("account_access", "account_access") == 1.0
    assert score_exact("account_access extra", "account_access") == 0.0
    assert score_json_field('{"total": 44}', {"total": 44}) == 1.0
    assert score_json_field('{"total": 44}', {"total": 44, "tax": 4}) == 0.5


def test_distillation_loss_flows_and_zeroes():
    from stratum.distill import distillation_loss
    torch.manual_seed(0)
    B, T, V = 2, 6, 40
    s = torch.randn(B, T, V, requires_grad=True)
    t = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T)); labels[:, :3] = -100
    loss = distillation_loss(s, t, labels)
    assert torch.isfinite(loss)
    loss.backward(); assert s.grad is not None
    # identical distributions, pure soft -> ~0
    logits = torch.randn(B, T, V)
    s2 = logits.clone().requires_grad_(True)
    z = distillation_loss(s2, logits.clone(), torch.randint(0, V, (B, T)), alpha=1.0)
    assert z.item() < 0.01


def test_distillation_handles_all_masked():
    from stratum.distill import distillation_loss
    B, T, V = 2, 5, 30
    s = torch.randn(B, T, V, requires_grad=True)
    loss = distillation_loss(s, torch.randn(B, T, V), torch.full((B, T), -100))
    assert torch.isfinite(loss)


def test_data_distillation_generator(tmp_path=None):
    import tempfile, os, json
    from stratum.distill import generate_dataset_from_teacher
    out = os.path.join(tempfile.mkdtemp(), "d.jsonl")
    generate_dataset_from_teacher(["a", "b"], "Do X.", lambda p: "answer", out)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 2 and rows[0]["response"] == "answer"


def test_teacher_backends():
    from stratum.teachers import get_teacher
    assert get_teacher("echo")("x").startswith("(echo")
    try:
        get_teacher("bogus"); assert False
    except ValueError:
        pass


def test_model_hint():
    from stratum.hf_utils import resolve_model_hint
    assert resolve_model_hint("qwen3") # bad id -> nonempty hint
    assert not resolve_model_hint("Qwen/Qwen3-1.7B") # good id -> empty


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
