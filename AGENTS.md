# Solver development

Build the user's paper and target examples with
[paper-to-newton-solver](.claude/skills/paper-to-newton-solver/SKILL.md).
The same skill is exposed to Codex through `.agents/skills/`.
If `tools/new_solver.py` exists, initialize this fresh clone in place
with `python3 tools/new_solver.py --in-place --name newton-solver-<method>`.
Setup removes that script; continue an initialized solver without reinitializing.

Keep the solver self-contained and use Newton's public API. Never patch installed
Newton or import `newton._src`. The starter has only a particle integrator and
plane contact; add runtime support only when the chosen method needs it. Keep
global matrices and solves sparse; small element blocks may be dense. Preserve
reference equations, resolution and numerical budgets when reproducing examples.

Keep mechanics in `_src/solvers/template/` (renamed to the method by setup), with
explicit exports through the family, `_src/solvers`, and the package root.
Examples use the public package API, not solver internals. Keep examples and
rendering helpers directly in `examples/`; static meshes belong in its `assets/`
folder. Separate coupling from mechanics when coupling is requested. Prefer
ordinary modules and direct calls over registries, facades or unused backends.

Use `uv`, Warp kernels and `unittest`. Keep dependencies and `uv.lock` synchronized.
Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked python -m unittest discover -s tests -v`, and `uv build`.
GPU tests use `NEWTON_TEST_DEVICE=cuda:0`. Validate advertised gradients and
measure warmed physics separately from rendering or playback.

The launcher discovers `example_*.py` automatically. Give each an accurate
docstring and `__main__` guard; `HEADLESS = True` marks terminal-only examples.
Keep equations in `docs/method.md` and paper-to-function links in `docs/papers.md`.
Generate README.md with `python3 tools/build_readme.py`; record actual example
clips with `tools/record_examples.py`. Preserve attribution and licenses. The
starter demonstrates integration and autodiff; it does not reproduce a paper.
