"""Core correctness tests for STRATUM. Run: python -m pytest tests/ -v"""
import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from stratum.muon import newton_schulz, Muon, split_params_for_muon
from stratum.merge import extract_deltas, merge, load_stratum_factors
from stratum.data import build_example, load_jsonl, make_batches, strip_think
from stratum.evaluate import (score_contains, score_exact, score_json_field,
                              _values_match)



def test_orthogonalization_flattens():
    torch.manual_seed(0)
    s = torch.linalg.svdvals(newton_schulz(torch.randn(8, 8)).float())
    assert s.max() < 1.6 and s.min() > 0.3


def test_newton_schulz_handles_zero():
    out = newton_schulz(torch.zeros(4, 4))
    assert torch.isfinite(out).all()


def test_muon_reduces_loss():
    torch.manual_seed(1)
    W = torch.nn.Parameter(torch.randn(8, 8))
    tgt = torch.randn(8, 8)
    opt = Muon([W], lr=0.1)
    first = None
    for i in range(60):
        opt.zero_grad()
        loss = ((W - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first * 0.5


def test_muon_guards_nan_gradient():
    W = torch.nn.Parameter(torch.randn(4, 4))
    opt = Muon([W])
    before = W.detach().clone()
    W.grad = torch.full((4, 4), float("nan"))
    opt.step()  # must not poison W
    assert torch.isfinite(W).all()
    assert torch.allclose(W.detach(), before)  # skipped, unchanged


def test_muon_rejects_non_2d():
    v = torch.nn.Parameter(torch.randn(4))
    v.grad = torch.randn(4)
    with pytest.raises(ValueError):
        Muon([v]).step()


def test_param_split_routes_embeddings_to_adamw():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 4)
            self.proj = torch.nn.Linear(4, 4)

    muon_p, adamw_p = split_params_for_muon(M())
    assert len(muon_p) == 1  # proj.weight only
    assert len(adamw_p) == 2  # embedding table and proj.bias



def _write_fake_stratum(d, extra_cfg=None, extra_tensors=None):
    r, alpha = 8, 16
    A = torch.randn(r, 16)
    B = torch.randn(16, r)
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B,
    }
    tensors.update(extra_tensors or {})
    save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
    cfg = {"r": r, "lora_alpha": alpha}
    cfg.update(extra_cfg or {})
    with open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return A, B, r, alpha


def test_extract_deltas_math():
    d = tempfile.mkdtemp()
    A, B, r, alpha = _write_fake_stratum(d)
    deltas = extract_deltas(d)
    key = "model.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(deltas[key], (B @ A) * (alpha / r), atol=1e-5)


def test_extract_deltas_respects_rslora_scaling():
    d = tempfile.mkdtemp()
    A, B, r, alpha = _write_fake_stratum(d, extra_cfg={"use_rslora": True})
    deltas = extract_deltas(d)
    key = "model.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(deltas[key], (B @ A) * (alpha / r ** 0.5), atol=1e-5)


def test_load_factors_rejects_dora():
    d = tempfile.mkdtemp()
    _write_fake_stratum(d, extra_cfg={"use_dora": True})
    with pytest.raises(ValueError, match="DoRA"):
        load_stratum_factors(d)


def test_load_factors_rejects_modules_to_save():
    d = tempfile.mkdtemp()
    _write_fake_stratum(d, extra_tensors={
        "base_model.model.lm_head.modules_to_save.weight": torch.randn(4, 4)})
    with pytest.raises(ValueError, match="modules_to_save"):
        load_stratum_factors(d)


def test_linear_merge_is_addition():
    a = {"w": torch.ones(4, 4)}
    b = {"w": torch.ones(4, 4) * 2}
    merged = merge("linear", [a, b], [1.0, 1.0])
    assert torch.allclose(merged["w"], torch.ones(4, 4) * 3)


def test_all_merge_methods_run():
    a = {"w": torch.randn(6, 6)}
    b = {"w": torch.randn(6, 6)}
    for method in ["linear", "ties", "dare"]:
        assert "w" in merge(method, [a, b])


def test_dare_is_deterministic_with_seed():
    a = {"w": torch.randn(16, 16)}
    b = {"w": torch.randn(16, 16)}
    m1 = merge("dare", [a, b], seed=5)
    m2 = merge("dare", [a, b], seed=5)
    m3 = merge("dare", [a, b], seed=6)
    assert torch.equal(m1["w"], m2["w"])
    assert not torch.equal(m1["w"], m3["w"])


def test_merge_rejects_bad_method():
    with pytest.raises(ValueError):
        merge("nope", [{"w": torch.randn(4, 4)}])



class FakeTok:
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        raise TypeError("no chat template on this fake")

    def __call__(self, text, **kw):
        ids = [ord(c) % 50 for c in text]
        limit = kw.get("max_length")
        if limit:
            ids = ids[:limit]
        return {"input_ids": ids}


def test_loss_mask_covers_prompt_only():
    tok = FakeTok()
    ids, labels = build_example(tok, "AAAA", "BBB", None, 64)
    masked = sum(1 for l in labels if l == -100)
    assert masked >= 4  # at least the prompt is masked
    assert masked < len(labels)  # but not everything
    assert len(ids) == len(labels)


def test_build_example_rejects_fully_masked_row():
    tok = FakeTok()
    with pytest.raises(ValueError, match="no response tokens"):
        build_example(tok, "A" * 100, "BBB", None, 40)


def test_make_batches_pads_and_masks():
    tok = FakeTok()
    rows = [{"prompt": "AA", "response": "BBBB"},
            {"prompt": "AAAAAAAA", "response": "BB"}]
    batches = list(make_batches(tok, rows, None, 64, 2))
    assert len(batches) == 1
    input_ids, attn, labels = batches[0]
    assert input_ids.shape == attn.shape == labels.shape
    # The shorter example is padded and its padding is masked out everywhere.
    lengths = attn.sum(dim=1)
    short = int(lengths.argmin())
    pad_region = attn[short] == 0
    assert (labels[short][pad_region] == -100).all()
    assert (input_ids[short][pad_region] == tok.pad_token_id).all()


def test_load_jsonl_validates(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"prompt": "a", "response": "b"}\n', encoding="utf-8")
    rows = load_jsonl(str(p), ("prompt", "response"))
    assert len(rows) == 1
    p.write_text('{"prompt": "a"}\n', encoding="utf-8")  # missing response
    with pytest.raises(ValueError):
        load_jsonl(str(p), ("prompt", "response"))


def test_load_jsonl_reads_utf8(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text('{"prompt": "Total: 88 €", "response": "ok"}\n', encoding="utf-8")
    rows = load_jsonl(str(p), ("prompt", "response"))
    assert "€" in rows[0]["prompt"]


def test_strip_think():
    assert strip_think("<think>reasoning here</think>42") == "42"
    assert strip_think("42") == "42"
    assert strip_think("<think>ran out of tok") == ""
    assert strip_think("a<think>x</think>b<think>y</think>c") == "abc"



def test_scorers():
    assert score_contains("the total is 88", "88") == 1.0
    assert score_exact("account_access", "account_access") == 1.0
    assert score_exact("account_access extra", "account_access") == 0.0
    assert score_json_field('{"total": 44}', {"total": 44}) == 1.0
    assert score_json_field('{"total": 44}', {"total": 44, "tax": 4}) == 0.5


def test_json_field_matches_number_formats():
    assert score_json_field('{"total": "1,499"}', {"total": 1499}) == 1.0
    assert score_json_field('{"total": "$88"}', {"total": 88}) == 1.0
    assert score_json_field('{"total": 88.0}', {"total": 88}) == 1.0
    assert _values_match("abc", "abc")
    assert not _values_match("abc", "abd")



def _valid_recipe(tmp_path):
    return {
        "base_model": "b", "output_model": "m",
        "strata": [{"name": "x", "skill": "s.jsonl", "out": "strata/x"}],
        "merge": {"method": "linear"},
    }


def test_recipe_valid_passes(tmp_path):
    import yaml
    from stratum.recipe import load_recipe
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(_valid_recipe(tmp_path)), encoding="utf-8")
    recipe = load_recipe(str(p))
    assert recipe["base_model"] == "b"


def test_recipe_rejects_unknown_key(tmp_path):
    import yaml
    from stratum.recipe import load_recipe
    bad = _valid_recipe(tmp_path)
    bad["strata"][0]["epoch"] = 5  # typo for epochs
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="epoch"):
        load_recipe(str(p))


def test_recipe_rejects_missing_required(tmp_path):
    import yaml
    from stratum.recipe import load_recipe
    bad = _valid_recipe(tmp_path)
    del bad["base_model"]
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="base_model"):
        load_recipe(str(p))


def test_recipe_rejects_weight_count_mismatch(tmp_path):
    import yaml
    from stratum.recipe import load_recipe
    bad = _valid_recipe(tmp_path)
    bad["merge"]["weights"] = [1.0, 1.0]
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="weights"):
        load_recipe(str(p))


def test_recipe_setting_fallback():
    from stratum.recipe import stratum_setting
    recipe = {"batch_size": 8}
    assert stratum_setting(recipe, {"batch_size": 2}, "batch_size", 4) == 2
    assert stratum_setting(recipe, {}, "batch_size", 4) == 8
    assert stratum_setting({}, {}, "batch_size", 4) == 4


def test_example_recipes_validate():
    from stratum.recipe import load_recipe
    root = os.path.join(os.path.dirname(__file__), "..", "examples")
    for name in ("recipe.yaml", "recipe-distill.yaml"):
        recipe = load_recipe(os.path.join(root, name))
        assert recipe["strata"]



def test_distillation_loss_flows_and_zeroes():
    from stratum.distill import distillation_loss
    torch.manual_seed(0)
    B, T, V = 2, 6, 40
    s = torch.randn(B, T, V, requires_grad=True)
    t = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = -100
    loss = distillation_loss(s, t, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert s.grad is not None
    # identical distributions, pure soft loss, should be ~0
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


def test_data_distillation_generator(tmp_path):
    from stratum.distill import generate_dataset_from_teacher
    out = str(tmp_path / "d.jsonl")
    generate_dataset_from_teacher(["a", "b"], "Do X.", lambda p: "answer", out)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 2
    assert rows[0]["response"] == "answer"


def test_data_distillation_resumes(tmp_path):
    from stratum.distill import generate_dataset_from_teacher
    out = str(tmp_path / "d.jsonl")
    calls = []

    def teacher(prompt):
        calls.append(prompt)
        return "answer"

    generate_dataset_from_teacher(["a", "b"], "Do X.", teacher, out)
    assert len(calls) == 2
    # Running again asks the teacher nothing - both pairs already exist.
    generate_dataset_from_teacher(["a", "b", "c"], "Do X.", teacher, out)
    assert len(calls) == 3
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 3


def test_data_distillation_retries_flaky_teacher(tmp_path, monkeypatch):
    import time
    from stratum.distill import generate_dataset_from_teacher
    monkeypatch.setattr(time, "sleep", lambda s: None)
    out = str(tmp_path / "d.jsonl")
    state = {"failed": False}

    def flaky(prompt):
        if not state["failed"]:
            state["failed"] = True
            raise ConnectionError("api hiccup")
        return "answer"

    generate_dataset_from_teacher(["a"], "Do X.", flaky, out)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert rows[0]["response"] == "answer"


def test_teacher_backends():
    from stratum.teachers import get_teacher
    assert get_teacher("echo")("x").startswith("(echo")
    with pytest.raises(ValueError, match="claude-cli"):
        get_teacher("bogus")


def test_claude_cli_teacher_needs_the_cli(monkeypatch):
    import shutil
    from stratum.teachers import get_teacher
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(EnvironmentError, match="Claude Code CLI"):
        get_teacher("claude-cli")


def test_gemini_teacher_needs_a_key(monkeypatch):
    from stratum.teachers import get_teacher
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        get_teacher("gemini")


def test_gemini_vision_teacher_needs_a_key(monkeypatch):
    from stratum.vision import get_vision_teacher
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        get_vision_teacher("gemini")


def test_model_hint():
    from stratum.hf_utils import resolve_model_hint
    assert resolve_model_hint("qwen3")  # bad id gives a nonempty hint
    assert not resolve_model_hint("Qwen/Qwen3-1.7B")  # good id gives nothing
