# Contributing to STRATUM

Welcome. STRATUM aims to make "train your own industry model" doable by a
curious developer on the hardware they own. Contributions that serve that goal:

- **Documented skill examples** - a well-made `examples/*.jsonl` teaching a
  useful skill, with a note on what it does and how many pairs it needs.
- **Hardware reports** - run `stratum doctor` and open an issue with your GPU,
  VRAM, and which base model size actually worked. Builds a community table.
- **Merge recipes** - good `--weights` / `--method` combinations for common
  skill pairings.
- **Doc improvements** - every doc assumes zero prior knowledge; tell us where
  it slips or leaves a gap.

## Ground rules

- Docs stay beginner-first. New terms get a plain definition and a glossary entry.
- Code stays readable over clever - this is a teaching project.
- Run `python -m pytest tests/ -v` before opening a PR; add tests for new logic.

MIT licensed - by contributing you release your work under it.
