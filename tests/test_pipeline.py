"""
End-to-end pipeline tests on a tiny self-built model (see conftest.py).

These are the tests the docs point at: train two strata, extract their deltas,
merge with every method, apply onto the base, evaluate, and check the merged
weights are exactly base + merged deltas. Everything runs on CPU in seconds.
"""
import json
from pathlib import Path

import pytest
import torch

from stratum.merge import (extract_deltas, load_stratum_factors, merge,
                           merge_strata, read_stratum_card)
from stratum.train import train_tile


@pytest.fixture(scope="session")
def two_strata(tiny_base, tmp_path_factory):
    """Train two strata off the tiny base, once per test session."""
    root = tmp_path_factory.mktemp("strata")
    examples = Path(__file__).parent.parent / "examples"

    dirs = []
    for name in ("extract", "classify"):
        out = root / name
        loss = train_tile(
            skill_path=str(examples / f"{name}.jsonl"),
            out_dir=str(out),
            base_model=tiny_base,
            rank=4, epochs=2, batch_size=2, grad_accum=2, max_len=96,
            load_4bit=False, seed=7,
        )
        assert loss is not None and loss == loss  # finished with a real number
        dirs.append(str(out))
    return dirs


def test_train_writes_adapter_and_card(two_strata, tiny_base):
    for d in two_strata:
        assert (Path(d) / "adapter_model.safetensors").exists()
        card = read_stratum_card(d)
        assert card["base_model"] == tiny_base
        assert card["rank"] == 4
        assert card["load_4bit"] is False
        assert card["num_pairs"] == 8


def test_extracted_delta_matches_factors(two_strata):
    factors = load_stratum_factors(two_strata[0])
    deltas = extract_deltas(two_strata[0])
    assert set(factors) == set(deltas)
    key = next(iter(factors))
    A, B, scaling = factors[key]
    assert torch.allclose(deltas[key], (B @ A) * scaling, atol=1e-6)


def test_merge_applies_exact_deltas(two_strata, tiny_base, tmp_path):
    from transformers import AutoModelForCausalLM

    out = tmp_path / "merged"
    summary = merge_strata(two_strata, str(out), method="linear",
                           weights=[1.0, 0.5])
    assert summary["deltas_applied"] > 0
    assert summary["deltas_unmatched"] == 0
    assert (out / "stratum_merge.json").exists()

    base_sd = AutoModelForCausalLM.from_pretrained(
        tiny_base, torch_dtype=torch.bfloat16).state_dict()
    merged_sd = AutoModelForCausalLM.from_pretrained(
        str(out), torch_dtype=torch.bfloat16).state_dict()

    d0 = extract_deltas(two_strata[0])
    d1 = extract_deltas(two_strata[1])
    key = next(iter(d0))
    expected = base_sd[key].float() + d0[key] + 0.5 * d1[key]
    # The merged model was saved in bf16, so compare at bf16 resolution.
    assert torch.allclose(merged_sd[key].float(), expected, atol=2e-2, rtol=2e-2)
    # And it must actually differ from the base.
    assert not torch.equal(merged_sd[key], base_sd[key])


def test_merge_all_methods_produce_models(two_strata, tmp_path):
    for method in ("ties", "dare"):
        out = tmp_path / f"merged-{method}"
        summary = merge_strata(two_strata, str(out), method=method)
        assert summary["deltas_applied"] > 0
        assert (out / "config.json").exists()


def test_merge_refuses_mismatched_bases(two_strata, tmp_path):
    import shutil

    other = tmp_path / "other-base-stratum"
    shutil.copytree(two_strata[0], other)
    card_path = other / "stratum_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["base_model"] = "somewhere/else"
    card_path.write_text(json.dumps(card), encoding="utf-8")

    with pytest.raises(ValueError, match="different base models"):
        merge_strata([two_strata[0], str(other)], str(tmp_path / "nope"))


def test_merge_refuses_missing_card(tmp_path):
    empty = tmp_path / "not-a-stratum"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="stratum_card"):
        merge_strata([str(empty)], str(tmp_path / "nope"))


def test_eval_runs_on_merged_model(two_strata, tmp_path):
    from stratum.evaluate import run_eval

    out = tmp_path / "merged-for-eval"
    merge_strata(two_strata, str(out))
    test_file = Path(__file__).parent.parent / "examples" / "test.jsonl"
    report = run_eval(str(out), str(test_file), scorer="contains",
                      max_new_tokens=8, verbose=False,
                      json_out=str(tmp_path / "report.json"))
    assert 0.0 <= report["mean"] <= 1.0
    assert report["n"] == 5
    assert set(report["per_skill"]) == {"extract", "classify"}
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["mean"] == report["mean"]


def test_eval_runs_on_single_stratum(two_strata, tmp_path):
    """A stratum dir has only an adapter - eval must attach it to its base."""
    from stratum.evaluate import run_eval

    test_file = Path(__file__).parent.parent / "examples" / "test-extract.jsonl"
    report = run_eval(two_strata[0], str(test_file), scorer="json_field",
                      max_new_tokens=8, verbose=False)
    assert 0.0 <= report["mean"] <= 1.0


def test_dare_same_seed_same_model(two_strata, tmp_path):
    from transformers import AutoModelForCausalLM

    sds = []
    for run in ("a", "b"):
        out = tmp_path / f"dare-{run}"
        merge_strata(two_strata, str(out), method="dare", drop=0.5, seed=123)
        sds.append(AutoModelForCausalLM.from_pretrained(
            str(out), torch_dtype=torch.bfloat16).state_dict())
    for key in sds[0]:
        assert torch.equal(sds[0][key], sds[1][key])


def test_train_rejects_prompt_longer_than_max_len(tiny_base, tmp_path):
    long_row = {"prompt": "count " * 400, "response": "done"}
    p = tmp_path / "long.jsonl"
    p.write_text(json.dumps(long_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no response tokens"):
        train_tile(skill_path=str(p), out_dir=str(tmp_path / "out"),
                   base_model=tiny_base, rank=2, epochs=1, max_len=32,
                   load_4bit=False)


def test_stack_runs_a_whole_recipe(tiny_base, tmp_path, monkeypatch):
    """The one-command build: recipe in, trained strata and merged model out."""
    import sys
    import yaml
    from stratum.__main__ import main

    examples = Path(__file__).parent.parent / "examples"
    recipe = {
        "base_model": tiny_base,
        "output_model": str(tmp_path / "model"),
        "load_4bit": False,
        "max_len": 96,
        "batch_size": 2,
        "strata": [
            {"name": "extract", "skill": str(examples / "extract.jsonl"),
             "out": str(tmp_path / "s1"), "rank": 2, "epochs": 1},
            {"name": "classify", "skill": str(examples / "classify.jsonl"),
             "out": str(tmp_path / "s2"), "rank": 2, "epochs": 1},
        ],
        "merge": {"method": "linear", "weights": [1.0, 1.0]},
        # A gate the tiny random model can always clear, to prove gates run.
        "evals": [{"test": str(examples / "test.jsonl"),
                   "scorer": "contains", "min_score": 0.0}],
    }
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml.safe_dump(recipe), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["stratum", "stack", str(rp)])
    main()

    assert (tmp_path / "model" / "config.json").exists()
    assert (tmp_path / "model" / "stratum_merge.json").exists()
    card = read_stratum_card(str(tmp_path / "s1"))
    assert card["max_len"] == 96  # recipe-wide setting reached the training run
    assert card["batch_size"] == 2

    # A gate the tiny model cannot clear must fail the build.
    recipe["evals"][0]["min_score"] = 1.01
    rp.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["stratum", "stack", str(rp)])
    with pytest.raises(SystemExit, match="eval gates"):
        main()


def test_train_handles_utf8_data(tiny_base, tmp_path, skill_file):
    """Non-ascii training data (euro signs, accents) must survive the trip."""
    rows = [{"prompt": "Extraire le total: 'Total: 88 €'",
             "response": '{"total": 88}'}] * 4
    p = tmp_path / "utf8.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
    out = tmp_path / "utf8-stratum"
    loss = train_tile(skill_path=str(p), out_dir=str(out), base_model=tiny_base,
                      rank=2, epochs=1, batch_size=2, max_len=96,
                      load_4bit=False, seed=3)
    assert loss == loss
