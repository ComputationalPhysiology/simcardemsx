# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`simcardemsx` (distribution name `simcardemsx`, import name `simcardemsx`) is a next-generation cardiac electro-mechanics solver built on FEniCSx. It couples an electrophysiology (EP) solver from [fenicsx-beat](https://github.com/finsberg/fenicsx-beat) (`beat`) with a mechanics solver from [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) (`pulse`), driven by cellular ODE models that are code-generated from `.ode` files via [gotranx](https://github.com/finsberg/gotranx).

This package requires FEniCSx/dolfinx, which is not pip-installable on its own — it must come from the `ghcr.io/fenics/dolfinx/dolfinx` container image (see `.devcontainer/`). Assume dolfinx, mpi4py, petsc4py, ufl, and basix are already present in the environment rather than trying to pip install them.

`third-party/` contains local, untracked (gitignored) checkouts of sibling/dependency projects (`fenicsx-beat`, `fenicsx-pulse`, `crossbridge`, `cardiac-geometriesx`) kept around for reference when reading their source. They are not part of this repo's history. They are maintained by the same people, so a change that belongs upstream is made as a PR against that project's own repo, not patched around here.

## Common commands

Run from the repository root.

```bash
# Editable install with dev extras (inside the dolfinx container — see .devcontainer/setup.sh
# for the extra h5py/scifem build steps needed first)
python3 -m pip install -e .[dev]

# Run the full test suite (also computes coverage per pyproject.toml addopts)
python3 -m pytest

# Run a single test file / test
python3 -m pytest tests/test_ode_model.py -v
python3 -m pytest tests/test_ode_model.py::test_code_generator_smoke_test -v

# Lint / format (ruff is the source of truth; line length 100)
ruff check .
ruff format .

# Type check
mypy src

# Run all pre-commit hooks (ruff, mypy, cspell, formatting checks) on the whole repo
pre-commit run --all
```

`pytest-codspeed` benchmarks (the `benchmark` extra) are separate from the normal suite. cspell checks `src/`, `tests/`, `docs`, `README.md` against `.cspell_dict.txt` — add domain-specific terms there rather than rewording code/docs to dodge the spellchecker.

## Architecture

A simulation couples two independently-meshed physics solvers that exchange state every mechanics time step.

**1. Code generation from `.ode` files (`ode_model.generate_ode_code`)**

A single gotranx `.ode` file describes the full cellular model, with one component named `"mechanics"` (the active-stress state variables, e.g. `Zetas`/`Zetaw`/`XS`/`XW` for a Land model) and the rest being EP state. `generate_ode_code` splits it (`ode.get_component("mechanics")`, `ode - mechanics_comp`) and code-generates two Python modules into an output dir, both via gotranx's own stock generators — there is no custom code generator, printer or method-signature template in this package any more (`ode2mechanics.py`/`template.py` are gone):

- The EP remainder is compiled by `gotranx.cli.gotran2py.get_code` (generalized Rush-Larsen scheme) into `ep_model.py` — plain numpy code, consumed by `beat.odesolver.DolfinODESolver`.
- The `"mechanics"` component is compiled by `gotranx.cli.gotran2ufl.get_code`, the same GRL scheme, into `mechanics_model.py`. Its state update is **UFL expressions instead of numpy/math**, so it can be evaluated symbolically on the mechanics mesh — nothing here is hand-ported any more.
- Each call passes `missing_values=<other side>.missing_variables`, and each written module gets a `provides: dict[str, int]` dict appended after generation: the names its own `missing_values()` returns, by output index — i.e. what it hands to the *other* side. `load_ode_modules` generates and loads both, returning an `ODEModules(ep, mechanics)` pair; prefer it over calling `generate_ode_code` and loading only `ep_model` — deriving what crosses (§2, `transfer_plan.resolve`) needs both.
- `backends/` holds the **activation backends**, which is the main organising idea of the package. Each owns how active tension is produced *and* how it is coupled to mechanics, and each is a `pulse.ActiveModel` handed straight to `pulse.CardiacModel`. `backends/base.py` defines the `ActivationBackend` protocol and the `Transfer` record describing a variable crossing between EP and mechanics (name + unit + state/monitor, since a bare name carries neither direction nor unit). **Only `ZetaSplitUFL` and `CrossbridgeSegregated` implement that protocol.** `GeneratedActivation`, the primary backend now, does not: it has no `wants_from_ep`/`gives_to_ep`/`ep_inputs` — what crosses is derived from the two generated modules instead (§2) and moved by a `TransferPlan`, not declared on the backend.
- `backends/generated.py` holds `GeneratedActivation`: the `mechanics` component, generated as UFL, stepped once per mechanics solve. `scheme="monolithic"` (the default) sets the component's `lmbda`/`dLambda` parameters from the *current* displacement, so `S(C)` is a UFL expression of `C` and Newton differentiates through the whole contraction model — whatever the `.ode` file contains, not just the two Land distortion states `ZetaSplitUFL` hand-coded. `scheme="segregated"` instead freezes `lmbda`/`dLambda` at the last converged step: the naive scheme, kept only as the non-convergent comparison. `t`, `dt` and every parameter are `dolfinx.fem.Constant`s, read when the form is assembled — not Python floats — because `pulse.StaticProblem` compiles the form once, while `dt == 0`; a Python-float branch taken at that point is baked in forever (exactly the bug `ZetaSplitUFL`'s dt fix, below, corrects), and gotranx's UFL printer also emits `ufl.Or(x > a, x < b)`, which fails outright on a float `x`. States live on a quadrature space at the mechanics form's own `quadrature_degree` by default (an `element=("family", degree)` override exists, e.g. for comparison against DG1), so the stored states are exactly the ones the residual used. `active_tension` is a P1 `Function`, averaged from the quadrature tension via `averaging.make_averager`, always reported **in kPa** regardless of the model's own `tension_unit` (`ZetaSplitUFL` reports Pa, `CrossbridgeSegregated` kPa).
- **`S` is the primary contract, not `strain_energy`.** `pulse.StaticProblem._material_form` assembles `model.S(C)` and never touches `strain_energy`; neither `GeneratedActivation` nor `ZetaSplitUFL` has a closed-form potential to differentiate (`Ta` depends on the stretch *rate* through states advanced by GRL), so both raise `NotImplementedError` from `strain_energy` rather than returning something plausible.
- `ZetaSplitUFL.Ta(lmbda)` is a **method** returning a UFL expression; the output `Function` is `active_tension`. Do not merge those names — a `Function` called with a float does not raise, it hangs in point evaluation.
- `backends/zeta_split.py` holds `ZetaSplitUFL`, the hand-ported Land model, now **deprecated** (`DeprecationWarning` at construction, naming `GeneratedActivation`) and kept only because physcardems imports it. Its `dt` used to be a Python float, baked into the compiled UFL form while `t == t_prev == 0`; it is now a UFL expression of `Constant`s, like `GeneratedActivation`'s, so its zeta states do see the current stretch inside Newton once the clock moves. Don't describe this backend as "monolithic-in-Newton by construction" any more — that phrasing predates the dt fix, and in any case `GeneratedActivation` now does the same for any `.ode` split, so it is no longer this backend's distinguishing feature.
- `backends/segregated.py` holds `CrossbridgeSegregated`, the Ca_i split. Because crossbridge is NumPy it cannot be re-integrated inside Newton, so this interface is segregated — which is *not convergent* without stabilization once active stiffness exceeds passive stiffness (R&Q 2021). It therefore delegates its stress form to `pulse.StabilizedActiveStress`. Its `step()` captures the stretch, advances the ODE with it, and sets `lmbda_prev` to it **in one place**; `post_solve()` is deliberately empty so those two can never diverge.
- `units.py` holds the EP/crossbridge/mechanics conversions (ms↔s, mM↔µM, kPa, `J_TRPN`). None of these mismatches raises. The one that is not a plain scale factor is `active_stiffness_to_mechanics`: `Ka` is reported per unit of crossbridge's `Λ = SL/SL0` but consumed against the solver's `λ`, so it needs `× SL_ref/SL0`. With the default `SL_ref == SL0` the factor is 1, which is exactly why omitting it goes unnoticed.
- `RDQ18` defines no `SL0` (it uses `SL` directly in `Chi(SL)`), so `CrossbridgeSegregated` requires an explicit `SL_ref` for it rather than guessing one — a guess would silently move the model along its force-length curve.
- **Active-stress convention:** backends take a `formulation`. The default is `stretch`, the Regazzoni & Quarteroni normalization `P_a = Ta·F f0⊗f0/|F f0|`, under which `|P_a f0| = Ta` regardless of stretch. The historical simcardems form `invariant` (`P_a = Ta·F f0⊗f0`) is larger by a factor of λ and is retained only to reproduce pre-change results — results generated with it are not comparable to new ones. On a 1-element contraction the difference is ~31% in stress but only ~2.4% in peak shortening, since Holzapfel-Ogden stiffens exponentially; don't assume the discrepancy is negligible on other geometries.

- `zero_d.py` replaces the FEM mechanics with R&Q's 0D tissue model, so the same contraction models can be driven through all three coupling schemes (monolithic / segregated / stabilized) with no mesh. The monolithic run is a genuine reference solution — it root-finds the strain at which the ODE is advanced so the balance holds — and `tests/test_zero_d_coupling.py` uses it to assert the claim the whole design rests on: the naive scheme's error *grows* under refinement while the stabilized one converges at first order. **It works in seconds and pascals**, following the paper, unlike the rest of the package which uses ms.
- The instability only appears in the **quasistatic** regime (`M = sigma = 0`) and needs `Ka > Kp`. With R&Q's dynamic parameters at small `dt` the inertia term `M/dt^2` swamps `Kp` and hides it — so a test that fails to show oscillation may just be in the wrong regime, not fixed.

**2. What crosses between EP and activation (`transfer_plan.py`, `interpolation.py`, `averaging.py`)**

Nothing declares the crossings directly; they are read off three name→index dicts every generated module carries: `missing` (what it needs from the *other* side), `provides` (what its own `missing_values()` hands to the other side) and `parameter`.

- `transfer_plan.resolve(ep_module, activation_module)` is the pure half — no meshes, no function spaces. It pairs each side's `missing` names against the other side's `provides`, and raises `ValueError` naming every name either side needs that the other does not produce (checked in both directions before raising, so one error reports the whole mismatch) — the guard against mismatched splits or stale generated code. `lmbda` is not read from a `provides` dict — it comes from the fibre stretch of the mechanics solve, not a generated `missing_values()` — so whether it crosses back to EP is a capability question, answered by whether `lmbda` is in the EP module's `parameter` dict; `dLambda` never crosses back (no shipped EP remainder consumes it). The result is a `Crossings(forward, backward, stretch_to_ep)`.
- `transfer_plan.TransferPlan` is the runtime half, built from a `Crossings`, the EP module, beat's `DolfinODESolver`, and a `GeneratedActivation`. **Forward** (`.forward(t)`): calls the EP module's own `missing_values()` on beat's current state/parameter arrays and interpolates each row into the backend's `inputs` (a quadrature space, typically, on the other mesh) via `interpolation.TransferOperator`. **Backward** (`.backward()`): **averages, never point-interpolates** — λ and everything else crossing back are discontinuous across mechanics cells, so point interpolation into a continuous P1 space would take each node's value from whichever neighbouring cell was visited last. It averages each backend output onto a space on the mechanics mesh matching the EP ODE space (`averaging.make_averager`: lumped nodal average for a P1 target, exact cell average for DG0), interpolates *that* onto the EP ODE space, and writes the result **in place** into `ode.missing_variables` / `ode.parameters` — beat's `DolfinODESolver` holds those arrays by reference at construction, so rebinding them would leave EP integrating with stale data. Because of that, the EP ODE space must be P1 or DG0 (`NotImplementedError` otherwise — averaging is only defined onto those two), and whenever `lmbda` crosses back, `ode.parameters` must already be 2-D (one column per point): a λ that varies between points cannot land in a parameter array with one value for all of them.
- `interpolation.TransferOperator` is unchanged in mechanism (`dolfinx.fem.create_interpolation_data` / `interpolate_nonmatching`, precomputed once at construction) but now refuses a quadrature-space *source* with `ValueError` rather than letting dolfinx abort the whole process — quadrature values only exist at that element's own points and cannot be interpolated from.

**3. Mechanics problem**

There is no problem subclass: stock `pulse.StaticProblem` is used, unmodified, with `GeneratedActivation`, `ZetaSplitUFL` or `CrossbridgeSegregated` as `model.active`. The active stress comes from the backend's `S`; solver options belong to whoever builds the problem, not to any backend or the controller (the tests and the worked example set `snes_atol = 1e-9`, since at resting calcium the first residual is ~1e-9 and stalls at round-off below a pure relative tolerance). `backend.post_solve()` — called by the caller after each converged solve — does what used to be `MechanicsProblem.post_solve`: for `GeneratedActivation`, storing the states/λ/outputs of the step the converged residual used as the new "previous" values, evaluated before anything is overwritten; for `ZetaSplitUFL`, recomputing `lmbda`/`dLambda`/`Ta` from the solved displacement and advancing the Land state.

**4. Orchestration (`controller.py`)**

`SimulationController.step()` is the whole time-stepping algorithm, and it drives only `GeneratedActivation` — `ZetaSplitUFL`/`CrossbridgeSegregated` remain usable directly against `pulse` without it. In order: run `dt_mech / dt_ep` EP micro-steps (`ep_callback(t, ep_step_idx)` after each); `TransferPlan.forward(t)` at the new `t`, into the backend's `inputs`; set `backend.t` to the *old* `t` and `backend.dt` to `dt_mech`, and solve one mechanics Newton step (`RuntimeError` if it fails to converge); `backend.post_solve()`; `TransferPlan.backward()`; `mech_callback(t, mech_step_idx, newton_iterations)`. Construction guards that `backend is mechanics_problem.model.active` (`ValueError` otherwise — the controller steps the backend and the problem solves with its own, so they must be the same object), that `dt_mech` is a whole multiple of `dt_ep` (rtol `1e-9` — round-off, not a real remainder), and, when the backend stores states on a quadrature space, that its `quadrature_degree` equals `geometry.metadata["quadrature_degree"]` — otherwise the stored states would not be the ones the residual used. `TransferPlan` construction (`resolve` plus the spaces/operators) happens once, in `__init__`.

**5. Output (`datacollector.py`)**

`DataCollector` + `Timers` handle writing point/field EP and mechanics data (via dolfinx's function evaluation at points) and simulation timing, driven by callbacks passed into `SimulationController.step`.

`numerical_experiments/strong_coupling_zetasplit/main.py` is a full worked example wiring all of the above together end-to-end (mesh generation via `cardiac-geometriesx`, `beat` conductivities/stimulus/solver setup, `pulse` material/BCs, the controller loop, and `DataCollector`) — read it to see how the pieces connect in practice rather than inferring it from the library modules alone. `--odefile` selects which `.ode` file (and hence which split) runs; output goes to `output/<odefile stem>/`; the EP ODE space is P1; and it writes `timings.json` (`ep_ode_s`, `ep_pde_s`, `mech_s`, plus `loop_s`/`total_s`) — the wall-time baseline the ODE-performance sub-project (`.scratch/roadmap.md`, sub-project 7) will compare against.

`config.py` defines pydantic models (using `pydantic_pint` for unit-aware fields) for simulation configuration; most of it beyond `Conductivity` is currently commented out / unimplemented.

## Testing notes

Tests avoid needing a real EP/mechanics stack where possible, though the coupling gates for `GeneratedActivation` deliberately do use one on a single element, since that is what the claims being tested are about:

- `test_generated_activation.py` drives `GeneratedActivation` directly (`register`/`post_solve`, no `pulse.StaticProblem`) against the *same* `mechanics` component generated as numpy instead of UFL, as the reference — the isometric-clamp gate matches it to round-off (`rtol=1e-10`).
- `test_monolithic_coupling.py` runs the coupling gates through a real one-element `pulse.StaticProblem`, quasistatic with a soft passive material so active stiffness exceeds passive — the regime R&Q show the naive segregated scheme is not convergent in. It judges a run unstable by the **onset of instability** — the first step after which the spread of λ over the element's quadrature points exceeds `1e-3` (or Newton fails) — rather than by when Newton gives up, which depends on solver settings the physics doesn't; the naive scheme's unstable mode is spatial, so `mean(λ)` alone hides it. Marked `@pytest.mark.slow`.
- `test_transfer_plan.py` tests `resolve()` against the three shipped splits and `TransferPlan`'s construction guards and each direction on its own; `test_round_trip.py` (gate 5) is the one test that runs both directions together, through `SimulationController`, with real beat and pulse and nothing injected — a bug like the old MEF-zeros regression (mechanics → EP transferring nothing because the buffer it interpolated from was never written) shows up here as a flat array, not a false pass.
- `test_averaging.py` tests `make_averager` against hand-computed lumped/cell averages.
- `test_coupled_system.py` exercises only `SimulationController`'s construction guards (quadrature-degree mismatch, backend identity, the `dt_mech`/`dt_ep` ratio) against the same real `make_ep_solver`/`make_mechanics` fixtures as the gates above, on 1-cell meshes for speed — not a `DummyEPSolver`/`MockODEModel`, which no longer exist in this suite.
- `test_isolated_ep.py` builds a minimal synthetic `.ode` file on the fly (via `load_ode_modules`) to check the generated EP module's own mechano-electric feedback (`missing_variables` reaching the RHS), independent of any coupling machinery.
- `test_isolated_mechanics.py` / `test_stability.py` exercise `ZetaSplitUFL` directly against `pulse.StaticProblem`.

`test_backends.py` carries the equivalence record for the port: it writes out, by hand, the stress form that the deleted `MechanicsProblem._material_form` built, and requires `ZetaSplitUFL` to reproduce it. Since that form no longer exists in the package, the test is the only remaining copy of it — don't delete it when adding backends.

## Planning docs

Specs, implementation plans, ADRs, the domain glossary and handoff notes live in `.scratch/`, which is gitignored on purpose: they are working material, not part of the repo. `.scratch/roadmap.md` is the entry point and lists the sub-projects in order. `.scratch/CONTEXT.md` is the glossary and `.scratch/adr/` holds the decisions still in force. `.scratch/archive/feat-coupler/` preserves what the abandoned `feat/coupler` branch learned; read its `HANDOFF.md` before re-deriving anything about the EP/mechanics coupling.
