---
name: paper-to-newton-solver
description: Build a self-contained Warp/Newton solver from a research paper and requested examples, including numerical validation and paper-to-code documentation. Use for new solver implementations or ports, not routine example edits.
---

# Paper to Newton solver

Turn the user's paper and target examples into a working solver in this repository.
Keep the chosen physical method and scope; add only code the solver actually uses.

## Start from the requested outcome

Read the repository's `AGENTS.md`, `pyproject.toml`, existing code, and the supplied
paper and reference examples. Only ask for missing information that blocks the
work: an identifiable paper, the intended demonstrations, or an unresolved
physical interpretation. Reuse answers already given. Infer routine naming and
setup choices; state assumptions about resolution, device and performance goals.

If `tools/new_solver.py` still exists, choose the requested name
(or derive one from the method) and run:

```bash
python3 tools/new_solver.py --in-place --name newton-solver-<method>
```

This initializes the cloned repository in place and keeps its Git history. Setup
removes the script; skip initialization when it is absent. Continue in the current
repository; a sibling package or another template checkout is unnecessary.

## Implement the actual mathematics

Read the relevant paper equations, algorithm order, supplements and experiment
settings. Inspect figures when they define geometry or results. Use released
reference code to resolve details, preserving its tested discretization,
substeps, contact laws and convergence settings when reproducing an example.
Explain conflicts with the paper or consequential deviations; do not silently
change physics or tolerances to make a run pass.

Keep `docs/method.md` concise and current: state variables and units, energy or
forces, discrete update/residual, boundary/contact assumptions, stopping criteria
and supported derivatives. Keep global matrices and linear solves sparse when
needed; small element blocks may be dense. Separate the physical residual from
preconditioners or approximate Hessians. Share an integrator only when multiple
paths actually use it; the paper determines the integration method.

Use installed Newton's public APIs and native Warp kernels. Never patch the
installed Newton package or import `newton._src`. Follow `SolverBase` for standard
particle/body state; an explicitly owned generalized state is appropriate when
the method requires it. Keep solver state, temporary workspaces and saved
backward data separate. Add dependencies explicitly and update `uv.lock`.

Keep mechanics in `src/<package>/_src/solvers/<method>/`, with explicit exports
through the family, `_src/solvers`, and package `__init__.py`. Examples use that
public API. Put required runtime geometry under `_src/geometry/`, reusable
integration under `_src/solvers/implicit/`, and coupling in its own folder only
when requested. Bundle static geometry in `examples/assets/`; retain procedural
geometry when it is part of the experiment or explicitly requested. Preserve
third-party licenses and attribution when adapting code or assets.

## Validate forward and backward behavior

Choose checks appropriate to the method: rest equilibrium, rigid motion, a known
loaded element, conservation or a reference trajectory. Keep requested reference
resolution and numerical budgets in the examples. If a performance target is
unmet, measure and report it rather than lowering fidelity without explanation.

For differentiation, state the supported inputs and loss. Use unrolled Warp
reverse mode for the executed update sequence, or an implicit adjoint of a
converged residual when appropriate. An implicit adjoint solves
`R_z.T @ a = loss_z` and accumulates `loss_x - R_x.T @ a`; its Jacobian must match
the physical residual, including parameter-dependent constraints. Verify forward
and transpose-solve residuals. A cached iteration matrix need not be that Jacobian.
Keep saved states alive and unchanged through backward. Leave tape recording and
adjoint storage out of ordinary forward runs when gradients are not requested.

Check central finite differences at several scaled step sizes, including a short
multistep loss; report absolute and relative errors. Separate smooth contact
branches from changes of topology, contact features or stick/slip state. Do not
claim derivatives through branches that are frozen. Add an optimization example
when requested; otherwise a focused gradient test can establish the contract.

## Finish the requested examples

Keep runnable examples and shared rendering helpers directly in `examples/`.
The launcher discovers `example_*.py`; give each an accurate docstring and a
`__main__` guard. Use `HEADLESS = True` for terminal-only examples. Replace starter
demos as the requested examples become available; avoid redundant showcases.

Run the repository's lint, tests, package build and relevant CPU/CUDA examples.
Measure warmed physics separately from compilation, rendering, backward and
optimization. Synchronize asynchronous GPU timings; report device, resolution,
precision, timestep/substeps and iteration budgets. Video playback is not a
real-time benchmark.

In `docs/papers.md`, link each implemented paper and map its equations or
pseudocode steps to actual functions using relative source links. Clearly label
partial implementations. Use `docs/usage.md` only for example-specific needs.
Generate the README with `python3 tools/build_readme.py` and record one actual
feature demonstration per example with `tools/record_examples.py`. Keep scripted
presentation in `tools/video_scenarios.py`. Report unavailable assets or rendering
capabilities; never substitute fabricated clips or unverified reproduction claims.
