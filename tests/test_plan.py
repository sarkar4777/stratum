"""Tests for build planning: size parsing, memory estimates, verdicts,
eval-gate validation, and the remote bundle."""
import pytest

from stratum.plan import (FITS, NO_FIT, TIGHT, estimate_vram_gb,
                          model_params_b, plan_recipe, plan_stratum,
                          rental_advice, write_remote_bundle)

GPU_8 = {"gpu": "fake 8gb", "vram_gb": 8.0, "cuda": True, "bnb": True}
GPU_24 = {"gpu": "fake 24gb", "vram_gb": 24.0, "cuda": True, "bnb": True}
CPU = {"gpu": None, "vram_gb": 0.0, "cuda": False, "bnb": False}


def test_params_parsed_from_model_names():
    assert model_params_b("Qwen/Qwen3-1.7B") == 1.7
    assert model_params_b("Qwen/Qwen3-0.6B") == 0.6
    assert model_params_b("meta-llama/Llama-3.1-8B-Instruct") == 8.0
    assert model_params_b("mistralai/Mistral-7b-v0.3") == 7.0
    assert model_params_b("some/local-path") is None
    assert model_params_b("C:/models/tiny-base") is None


def test_estimate_moves_the_right_way():
    small = estimate_vram_gb(1.7, load_4bit=True)
    big = estimate_vram_gb(8.0, load_4bit=True)
    assert big > small
    assert estimate_vram_gb(1.7, load_4bit=False) > small
    assert estimate_vram_gb(1.7, batch_size=8) > small
    assert estimate_vram_gb(1.7, teacher_params_b=4.0) > small
    # The anchors the docs quote: 1.7B QLoRA fits 8 GB, 8B bf16 does not.
    assert estimate_vram_gb(1.7, load_4bit=True) < 8
    assert estimate_vram_gb(8.0, load_4bit=False) > 16


def test_stratum_verdicts():
    fits = plan_stratum("x", 1.7, GPU_8, True, 4, 1024)
    assert fits["verdict"] == FITS
    nope = plan_stratum("x", 8.0, GPU_8, False, 4, 1024)
    assert nope["verdict"] == NO_FIT
    assert nope["suggestions"]  # it must offer a way out
    ok_on_big = plan_stratum("x", 8.0, GPU_24, True, 4, 1024)
    assert ok_on_big["verdict"] in (FITS, TIGHT)


def test_unknown_size_passes_with_note():
    p = plan_stratum("x", None, GPU_8, True, 4, 1024)
    assert p["verdict"] == FITS
    assert p["suggestions"]


def test_cpu_verdicts():
    assert plan_stratum("x", 0.6, CPU, True, 4, 1024)["verdict"] == FITS
    assert plan_stratum("x", 4.0, CPU, True, 4, 1024)["verdict"] == NO_FIT


def test_plan_recipe_takes_worst_verdict():
    recipe = {
        "base_model": "Qwen/Qwen3-8B", "output_model": "m",
        "load_4bit": False,
        "strata": [
            {"name": "a", "skill": "s", "out": "o"},
            {"name": "b", "skill": "s", "out": "o", "load_4bit": True},
        ],
    }
    plan = plan_recipe(recipe, GPU_8)
    assert plan["verdict"] == NO_FIT  # the bf16 stratum drags it down
    assert len(plan["strata"]) == 2


def test_distill_stratum_counts_the_teacher():
    recipe = {
        "base_model": "Qwen/Qwen3-1.7B", "output_model": "m",
        "strata": [
            {"name": "d", "skill": "s", "out": "o",
             "distill": {"teacher": "Qwen/Qwen3-8B"}},
        ],
    }
    with_teacher = plan_recipe(recipe, GPU_8)
    assert with_teacher["strata"][0]["need_gb"] > estimate_vram_gb(1.7)


def test_rental_advice_scales_with_size():
    assert "24 GB" in rental_advice(1.7)
    assert "48 GB" in rental_advice(8.0)
    assert "80 GB" in rental_advice(32.0)


def test_remote_bundle_contents(tmp_path):
    recipe = {"base_model": "b", "output_model": "models/my-slm",
              "strata": [], "evals": [{"test": "t.jsonl"}]}
    path = write_remote_bundle(str(tmp_path / "remote"), "recipe.yaml", recipe)
    text = open(path, encoding="utf-8").read()
    assert "stratum stack recipe.yaml" in text
    assert "models/my-slm" in text
    assert "tar -czf" in text


def test_recipe_validates_evals(tmp_path):
    import yaml
    from stratum.recipe import load_recipe

    good = {"base_model": "b", "output_model": "m",
            "strata": [{"name": "x", "skill": "s", "out": "o"}],
            "evals": [{"test": "t.jsonl", "scorer": "exact", "min_score": 0.5}]}
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(good), encoding="utf-8")
    assert load_recipe(str(p))["evals"]

    bad = dict(good)
    bad["evals"] = [{"test": "t.jsonl", "minscore": 0.5}]  # typo
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="minscore"):
        load_recipe(str(p))

    bad["evals"] = [{"scorer": "exact"}]  # no test file
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        load_recipe(str(p))
