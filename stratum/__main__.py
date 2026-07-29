"""
STRATUM command-line interface.

    stratum doctor check hardware, recommend size
    stratum train --skill S.jsonl --out strata/x train one stratum
    stratum merge strata/a strata/b --out model fuse strata into a model
    stratum eval model --test T.jsonl score a model
    stratum chat model talk to a model
    stratum stack recipe.yaml run a whole build from a recipe

Run `stratum <command> -h` for per-command options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_doctor(args):
    import torch
    print("STRATUM doctor\n" + "-" * 42)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}\nVRAM: {vram:.1f} GB")
        if vram >= 40:
            rec = "Qwen3-8B comfortably, or 4B in full precision."
        elif vram >= 24:
            rec = "Qwen3-4B comfortably; 8B with 4-bit."
        elif vram >= 12:
            rec = "Qwen3-4B with 4-bit; 1.7B in bf16."
        elif vram >= 8:
            rec = "Qwen3-1.7B; or 4B with 4-bit + short sequences."
        else:
            rec = "Qwen3-0.6B in bf16; 1.7B with 4-bit."
        print(f"\nRecommended base: {rec}")
        try:
            import bitsandbytes # noqa
            print("4-bit (QLoRA): available.")
        except Exception:
            print("4-bit (QLoRA): NOT installed. `pip install bitsandbytes` to fit bigger models.")
    else:
        print("No CUDA GPU. CPU training works but is slow.")
        print("\nRecommended base: Qwen3-0.6B, small datasets, patience.")

    print()
    from .hf_utils import check_hf_ready
    check_hf_ready(verbose=True)


def cmd_train(args):
    from .train import train_tile
    train_tile(
        skill_path=args.skill, out_dir=args.out, base_model=args.base,
        rank=args.rank, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, max_len=args.max_len, optimizer=args.optimizer,
        system=args.system, load_4bit=not args.no_4bit, seed=args.seed,
    )


def cmd_merge(args):
    _do_merge(args.strata, args.out, args.method, args.weights, args.density, args.drop)


def _do_merge(strata, out_dir, method, weights, density, drop):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .merge import extract_deltas, merge as merge_deltas

    # Verify common base.
    cards = []
    for s in strata:
        card_path = Path(s) / "stratum_card.json"
        if not card_path.exists():
            sys.exit(f"{s} has no stratum_card.json - is it a STRATUM stratum?")
        cards.append(json.loads(card_path.read_text()))
    bases = {c["base_model"] for c in cards}
    if len(bases) != 1:
        sys.exit(f"Strata have different base models: {bases}. "
                 f"Only strata from the same base can be merged.")
    base_model = bases.pop()

    if weights is None:
        weights = [1.0] * len(strata)
    if len(weights) != len(strata):
        sys.exit("Number of --weights must match number of strata.")

    print(f"Merging {len(strata)} strata from {base_model}")
    print(f" method={method} weights={weights}")

    deltas = [extract_deltas(s) for s in strata]
    kw = {}
    if method == "ties":
        kw["density"] = density
    if method == "dare":
        kw["drop"] = drop
    merged = merge_deltas(method, deltas, weights, **kw)

    # Apply onto a fresh base and save a standalone model.
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    sd = model.state_dict()
    applied, missing = 0, 0
    for name, delta in merged.items():
        if name in sd:
            sd[name] = sd[name] + delta.to(sd[name].dtype)
            applied += 1
        else:
            missing += 1
    model.load_state_dict(sd)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outp))
    tokenizer.save_pretrained(str(outp))
    (outp / "stratum_merge.json").write_text(json.dumps({
        "base_model": base_model, "method": method, "weights": weights,
        "strata": [c["stratum_name"] for c in cards], "deltas_applied": applied,
    }, indent=2))
    print(f"Applied {applied} weight deltas ({missing} unmatched). Model -> {out_dir}")


def cmd_eval(args):
    from .evaluate import run_eval
    run_eval(args.model, args.test, scorer=args.scorer, system=args.system)


def cmd_chat(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import format_chat

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("Chat with your STRATUM model. Ctrl-C to quit.\n")
    try:
        while True:
            q = input("you: ").strip()
            if not q:
                continue
            text = format_chat(tokenizer, q, response=None, system=args.system)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=400, do_sample=False,
                                     temperature=None, top_p=None,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            print("stratum:", tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


def cmd_distill(args):
    """Train a student stratum by logit-distilling from a teacher model."""
    from .distill import distill_tile
    distill_tile(
        skill_path=args.skill, out_dir=args.out,
        student_model=args.student, teacher_model=args.teacher,
        rank=args.rank, epochs=args.epochs, batch_size=args.batch_size,
        max_len=args.max_len, temperature=args.temperature, alpha=args.alpha,
        system=args.system, seed=args.seed,
    )


def cmd_teacher_gen(args):
    """Data distillation: have a teacher WRITE training pairs from seed inputs."""
    from .distill import generate_dataset_from_teacher
    from .teachers import get_teacher

    seeds = [l.strip() for l in Path(args.seeds).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(seeds)} seed inputs from {args.seeds}")
    teacher_fn = get_teacher(args.teacher, model=args.model)
    generate_dataset_from_teacher(seeds, args.instruction, teacher_fn, args.out)


def cmd_stack(args):
    """Run a whole build from a YAML recipe: train listed strata, then merge them."""
    import yaml
    from .train import train_tile

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    base = recipe["base_model"]
    strata_dirs = []
    for st in recipe["strata"]:
        out = st["out"]
        if "distill" in st:
            # This stratum is distilled from a teacher.
            from .distill import distill_tile
            dcfg = st["distill"]
            print(f"\n=== Distilling stratum: {st['name']} (teacher {dcfg['teacher']}) ===")
            distill_tile(
                skill_path=st["skill"], out_dir=out,
                student_model=base, teacher_model=dcfg["teacher"],
                rank=st.get("rank", 16), epochs=st.get("epochs", 3),
                temperature=dcfg.get("temperature", 2.0), alpha=dcfg.get("alpha", 0.5),
                system=recipe.get("system"),
            )
        else:
            print(f"\n=== Training stratum: {st['name']} ===")
            train_tile(
                skill_path=st["skill"], out_dir=out, base_model=base,
                rank=st.get("rank", 16), epochs=st.get("epochs", 3),
                optimizer=recipe.get("optimizer", "muon"),
                system=recipe.get("system"),
                load_4bit=recipe.get("load_4bit", True),
            )
        strata_dirs.append(out)

    m = recipe.get("merge", {})
    print("\n=== Merging strata ===")
    _do_merge(strata_dirs, recipe["output_model"], m.get("method", "linear"),
              m.get("weights"), m.get("density", 0.2), m.get("drop", 0.9))
    print(f"\nBuild complete -> {recipe['output_model']}")


def main():
    p = argparse.ArgumentParser(prog="stratum",
                                description="Build specialized SLMs from mergeable skill strata.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check hardware, recommend model size")
    d.set_defaults(func=cmd_doctor)

    t = sub.add_parser("train", help="train one stratum")
    t.add_argument("--skill", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base", default="Qwen/Qwen3-1.7B")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-2, help="Muon learning rate")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--max-len", type=int, default=1024)
    t.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    t.add_argument("--system", default=None, help="optional system prompt")
    t.add_argument("--no-4bit", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("merge", help="fuse strata into one model")
    m.add_argument("strata", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--method", choices=["linear", "ties", "dare"], default="linear")
    m.add_argument("--weights", nargs="+", type=float)
    m.add_argument("--density", type=float, default=0.2, help="TIES: fraction kept")
    m.add_argument("--drop", type=float, default=0.9, help="DARE: fraction dropped")
    m.set_defaults(func=cmd_merge)

    e = sub.add_parser("eval", help="score a model on a test set")
    e.add_argument("model")
    e.add_argument("--test", required=True)
    e.add_argument("--scorer", choices=["contains", "exact", "json_field"], default="contains")
    e.add_argument("--system", default=None)
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("chat", help="talk to a model")
    c.add_argument("model")
    c.add_argument("--system", default=None)
    c.set_defaults(func=cmd_chat)

    di = sub.add_parser("distill", help="train a student stratum by imitating a teacher (logit distillation)")
    di.add_argument("--skill", required=True)
    di.add_argument("--out", required=True)
    di.add_argument("--student", default="Qwen/Qwen3-1.7B", help="small model that learns")
    di.add_argument("--teacher", default="Qwen/Qwen3-4B", help="big model to imitate (must share tokenizer)")
    di.add_argument("--rank", type=int, default=16)
    di.add_argument("--epochs", type=int, default=3)
    di.add_argument("--batch-size", type=int, default=2)
    di.add_argument("--max-len", type=int, default=1024)
    di.add_argument("--temperature", type=float, default=2.0, help="soften distributions")
    di.add_argument("--alpha", type=float, default=0.5, help="soft vs hard loss weight")
    di.add_argument("--system", default=None)
    di.add_argument("--seed", type=int, default=42)
    di.set_defaults(func=cmd_distill)

    tg = sub.add_parser("teacher-gen", help="data distillation: a teacher writes training pairs from seed inputs")
    tg.add_argument("--seeds", required=True, help="text file, one seed input per line")
    tg.add_argument("--instruction", required=True, help="what the skill should do, one line")
    tg.add_argument("--out", required=True, help="output training JSONL")
    tg.add_argument("--teacher", default="hf", choices=["hf", "openai", "anthropic", "echo"],
                    help="teacher backend")
    tg.add_argument("--model", default=None, help="teacher model name/id for the chosen backend")
    tg.set_defaults(func=cmd_teacher_gen)

    s = sub.add_parser("stack", help="run a full build from a YAML recipe")
    s.add_argument("recipe")
    s.set_defaults(func=cmd_stack)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
