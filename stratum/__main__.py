"""
STRATUM command-line interface.

    stratum doctor                                   check hardware, recommend size
    stratum plan recipe.yaml                         will this build fit here?
    stratum train --skill S.jsonl --out strata/x     train one stratum
    stratum merge strata/a strata/b --out model      fuse strata into a model
    stratum eval model --test T.jsonl                score a model (or one stratum)
    stratum chat model                               talk to a model
    stratum distill ...                              student imitates a teacher
    stratum teacher-gen ...                          teacher writes training pairs
    stratum stack recipe.yaml                        run a whole build from a recipe

Run `stratum <command> -h` for per-command options.
"""
from __future__ import annotations

import argparse
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
            rec = "Qwen3-4B comfortably, 8B with 4-bit."
        elif vram >= 12:
            rec = "Qwen3-4B with 4-bit, 1.7B in bf16."
        elif vram >= 8:
            rec = "Qwen3-1.7B, or 4B with 4-bit + short sequences."
        else:
            rec = "Qwen3-0.6B in bf16, 1.7B with 4-bit."
        print(f"\nRecommended base: {rec}")
        try:
            import bitsandbytes  # noqa
            print("4-bit (QLoRA): available.")
        except Exception:
            print("4-bit (QLoRA): NOT installed. `pip install bitsandbytes` to fit bigger models.")
    else:
        print("No CUDA GPU. CPU training works but is slow.")
        print("\nRecommended base: Qwen3-0.6B, small datasets, patience.")

    print()
    from .hf_utils import check_hf_ready
    check_hf_ready(verbose=True)
    print("\nTo check a specific build against this machine: "
          "`stratum plan recipe.yaml`")


def cmd_train(args):
    from .train import train_tile
    train_tile(
        skill_path=args.skill, out_dir=args.out, base_model=args.base,
        rank=args.rank, lr=args.lr, adamw_lr=args.adamw_lr, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_len=args.max_len, optimizer=args.optimizer,
        system=args.system, load_4bit=not args.no_4bit, seed=args.seed,
    )


def cmd_merge(args):
    from .merge import merge_strata
    try:
        merge_strata(args.strata, args.out, method=args.method,
                     weights=args.weights, density=args.density,
                     drop=args.drop, seed=args.seed)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_eval(args):
    from .evaluate import run_eval
    report = run_eval(args.model, args.test, scorer=args.scorer,
                      system=args.system, json_out=args.json_out,
                      baseline=args.baseline)
    if args.min_score is not None and report["mean"] < args.min_score:
        sys.exit(f"FAIL: score {report['mean']:.1%} is below --min-score "
                 f"{args.min_score:.1%}.")


def cmd_chat(args):
    import torch
    from .data import format_messages, strip_think
    from .hf_utils import load_for_inference

    model, tokenizer = load_for_inference(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # The conversation so far. Each turn is fed back in, so the model sees
    # its own history - without this, "chat" would be amnesiac one-shots.
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Chat with your STRATUM model. Ctrl-C to quit.\n")
    try:
        while True:
            q = input("you: ").strip()
            if not q:
                continue
            messages.append({"role": "user", "content": q})
            text = format_messages(tokenizer, messages, add_generation_prompt=True)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=400, do_sample=False,
                                     temperature=None, top_p=None,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            answer = strip_think(tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
            messages.append({"role": "assistant", "content": answer})
            print("stratum:", answer, "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


def cmd_distill(args):
    from .distill import distill_tile
    distill_tile(
        skill_path=args.skill, out_dir=args.out,
        student_model=args.student, teacher_model=args.teacher,
        rank=args.rank, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_len=args.max_len, temperature=args.temperature, alpha=args.alpha,
        system=args.system, teacher_4bit=args.teacher_4bit, seed=args.seed,
    )


def cmd_teacher_gen(args):
    from .distill import generate_dataset_from_teacher
    from .teachers import get_teacher

    seeds = [l.strip() for l in
             Path(args.seeds).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(seeds)} seed inputs from {args.seeds}")
    teacher_fn = get_teacher(args.teacher, model=args.model)
    generate_dataset_from_teacher(seeds, args.instruction, teacher_fn, args.out)


def cmd_plan(args):
    """Check a recipe against this machine before spending hours on it."""
    from .plan import (plan_recipe, print_plan, probe_hardware,
                       write_remote_bundle)
    from .recipe import load_recipe

    try:
        recipe = load_recipe(args.recipe)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    hw = probe_hardware()
    plan = plan_recipe(recipe, hw)
    print_plan(plan, hw, args.recipe)
    if args.emit_remote:
        print()
        write_remote_bundle(args.emit_remote, args.recipe, recipe)


def cmd_stack(args):
    """Run a whole build from a YAML recipe: train listed strata, then merge them."""
    from .recipe import load_recipe, stratum_setting
    from .merge import merge_strata

    try:
        recipe = load_recipe(args.recipe)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    # Preflight: refuse a build the hardware clearly cannot run, before it
    # trains for an hour and dies. The estimate is rough, so only a clear
    # no-fit stops the build, and --force overrides even that.
    from .plan import NO_FIT, plan_recipe, probe_hardware, rental_advice
    hw = probe_hardware()
    plan = plan_recipe(recipe, hw)
    if plan["verdict"] == NO_FIT and not args.force:
        sys.exit(
            f"This build likely will not fit on this machine "
            f"(run `stratum plan {args.recipe}` for the details and fixes).\n"
            f"For the training burst, {rental_advice(plan['base_params_b'])}\n"
            f"Use --force to try anyway."
        )

    base = recipe["base_model"]
    strata_dirs = []
    for st in recipe["strata"]:
        out = st["out"]
        common = dict(
            rank=st.get("rank", 16),
            epochs=st.get("epochs", 3),
            lr=stratum_setting(recipe, st, "lr", 2e-2),
            batch_size=stratum_setting(recipe, st, "batch_size", 4),
            max_len=stratum_setting(recipe, st, "max_len", 1024),
            system=stratum_setting(recipe, st, "system", None),
            seed=stratum_setting(recipe, st, "seed", 42),
        )
        if "distill" in st:
            from .distill import distill_tile
            dcfg = st["distill"]
            print(f"\n=== Distilling stratum: {st['name']} (teacher {dcfg['teacher']}) ===")
            distill_tile(
                skill_path=st["skill"], out_dir=out,
                student_model=base, teacher_model=dcfg["teacher"],
                temperature=dcfg.get("temperature", 2.0),
                alpha=dcfg.get("alpha", 0.5),
                teacher_4bit=dcfg.get("teacher_4bit", False),
                grad_accum=stratum_setting(recipe, st, "grad_accum", 4),
                **{**common, "batch_size": dcfg.get("batch_size",
                                                    common["batch_size"])},
            )
        else:
            from .train import train_tile
            print(f"\n=== Training stratum: {st['name']} ===")
            train_tile(
                skill_path=st["skill"], out_dir=out, base_model=base,
                optimizer=stratum_setting(recipe, st, "optimizer", "muon"),
                adamw_lr=stratum_setting(recipe, st, "adamw_lr", 1e-3),
                grad_accum=stratum_setting(recipe, st, "grad_accum", 4),
                load_4bit=stratum_setting(recipe, st, "load_4bit", True),
                **common,
            )
        strata_dirs.append(out)

    m = recipe.get("merge", {})
    print("\n=== Merging strata ===")
    try:
        merge_strata(strata_dirs, recipe["output_model"],
                     method=m.get("method", "linear"), weights=m.get("weights"),
                     density=m.get("density", 0.2), drop=m.get("drop", 0.9),
                     seed=m.get("seed", 42))
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    # Eval gates: the recipe tests what it built. A failed gate fails the
    # build, which is what lets CI or a rented box run this unattended.
    failures = []
    if recipe.get("evals"):
        from .evaluate import run_eval
        print("\n=== Evaluating ===")
        for ev in recipe["evals"]:
            report = run_eval(recipe["output_model"], ev["test"],
                              scorer=ev.get("scorer", "contains"),
                              system=ev.get("system", recipe.get("system")))
            bar = ev.get("min_score")
            if bar is not None and report["mean"] < bar:
                failures.append(f"{ev['test']}: {report['mean']:.1%} "
                                f"is below min_score {bar:.1%}")
    if failures:
        sys.exit("Build FAILED its eval gates:\n  " + "\n  ".join(failures))
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
    t.add_argument("--adamw-lr", type=float, default=1e-3,
                   help="learning rate for the AdamW side (non-matrix params, or everything with --optimizer adamw)")
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
    m.add_argument("--seed", type=int, default=42,
                   help="makes DARE's random dropping reproducible")
    m.set_defaults(func=cmd_merge)

    e = sub.add_parser("eval", help="score a model (or a single stratum) on a test set")
    e.add_argument("model")
    e.add_argument("--test", required=True)
    e.add_argument("--scorer", choices=["contains", "exact", "json_field"], default="contains")
    e.add_argument("--system", default=None)
    e.add_argument("--json-out", default=None,
                   help="write the full report as JSON here (for CI)")
    e.add_argument("--min-score", type=float, default=None,
                   help="exit non-zero if the mean score is below this (for CI gating)")
    e.add_argument("--baseline", default=None,
                   help="also score this model (usually the base) for comparison")
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
    di.add_argument("--lr", type=float, default=2e-2, help="Muon learning rate")
    di.add_argument("--epochs", type=int, default=3)
    di.add_argument("--batch-size", type=int, default=2)
    di.add_argument("--grad-accum", type=int, default=4)
    di.add_argument("--max-len", type=int, default=1024)
    di.add_argument("--temperature", type=float, default=2.0, help="soften distributions")
    di.add_argument("--alpha", type=float, default=0.5, help="soft vs hard loss weight")
    di.add_argument("--system", default=None)
    di.add_argument("--teacher-4bit", action="store_true",
                    help="load the frozen teacher in 4-bit to fit both models (NVIDIA GPU)")
    di.add_argument("--seed", type=int, default=42)
    di.set_defaults(func=cmd_distill)

    tg = sub.add_parser("teacher-gen", help="data distillation: a teacher writes training pairs from seed inputs")
    tg.add_argument("--seeds", required=True, help="text file, one seed input per line")
    tg.add_argument("--instruction", required=True, help="what the skill should do, one line")
    tg.add_argument("--out", required=True, help="output training JSONL (re-run to resume)")
    tg.add_argument("--teacher", default="hf", choices=["hf", "openai", "anthropic", "echo"],
                    help="teacher backend")
    tg.add_argument("--model", default=None, help="teacher model name/id for the chosen backend")
    tg.set_defaults(func=cmd_teacher_gen)

    pl = sub.add_parser("plan", help="check a recipe against this machine, or plan a remote build")
    pl.add_argument("recipe")
    pl.add_argument("--emit-remote", default=None, metavar="DIR",
                    help="write a build-and-test script for a rented GPU box here")
    pl.set_defaults(func=cmd_plan)

    s = sub.add_parser("stack", help="run a full build from a YAML recipe")
    s.add_argument("recipe")
    s.add_argument("--force", action="store_true",
                   help="run even when the preflight says it will not fit")
    s.set_defaults(func=cmd_stack)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
