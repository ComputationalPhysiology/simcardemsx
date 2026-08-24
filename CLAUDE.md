# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`simcardemsx` (distribution name `simcardemsx`, import name `simcardemsx`) is a next-generation cardiac electro-mechanics solver built on FEniCSx. It couples an electrophysiology (EP) solver from [fenicsx-beat](https://github.com/finsberg/fenicsx-beat) (`beat`) with a mechanics solver from [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) (`pulse`), driven by cellular ODE models that are code-generated from `.ode` files via [gotranx](https://github.com/finsberg/gotranx).

This package requires FEniCSx/dolfinx, which is not pip-installable on its own — it must come from the `ghcr.io/fenics/dolfinx/dolfinx` container image (see `.devcontainer/`). Assume dolfinx, mpi4py, petsc4py, ufl, and basix are already present in the environment rather than trying to pip install them.

`third-party/` contains local, untracked (gitignored) checkouts of sibling/dependency projects (`fenicsx-beat`, `fenicsx-pulse`, `circulation`, `crossbrigde`) kept around for reference when reading their source — they are not part of this repo's history and shouldn't be edited as part of work here.

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
python3 -m pytest tests/test_ode_model.py::test_runtime_ode_model_with_mock_dict -v

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

**1. Code generation from `.ode` files (`ode_model.generate_ode_code`, `ode2mechanics.py`, `template.py`)**

A single gotranx `.ode` file describes the full cellular model, with one component named `"mechanics"` (the Land active-stress state variables, e.g. `Zetas`/`Zetaw`/`XS`/`XW`) and the rest being EP state. `generate_ode_code` splits it in two and code-generates two different Python modules into an output dir:

- The EP remainder (`ode - mechanics_component`) is compiled with gotranx's stock `PythonCodeGenerator` (generalized Rush-Larsen scheme) into `ep_model.py` — plain numpy code, consumed by `beat.odesolver.DolfinODESolver`.
- The `"mechanics"` component is compiled by the custom `SimcardemsCodeGenerator`/`SimcardemsPrinter` in `ode2mechanics.py` into `mechanics_model.py`. This generator emits **UFL expressions instead of numpy/math** (`ufl.conditional`, `ufl.lt`, `ufl.And`, …, via `rel_op_2_ufl` and the overridden `_print_*` methods) and wraps the generated rush-larsen update inside a hardcoded `LAND_MODEL` string that subclasses `pulse.active_model.ActiveModel`. The result is a dolfinx-native active model whose state update runs symbolically on the mesh. `template.py` overrides gotranx's default method-signature template (`method()`) used by both generators.
- `backends/` holds the **activation backends**, which is the main organising idea of the package. Each owns how active tension is produced *and* how it is coupled to mechanics, and each is a `pulse.ActiveModel` handed straight to `pulse.CardiacModel`. `backends/base.py` defines the `ActivationBackend` protocol and the `Transfer` record describing a variable crossing between EP and mechanics (name + unit + state/monitor, since a bare name carries neither direction nor unit). `backends/zeta_split.py` holds `ZetaSplitUFL`, the ported Land model.
- **`S` is the primary contract, not `strain_energy`.** `pulse.StaticProblem._material_form` assembles `model.S(C)` and never touches `strain_energy`, and most backends have no closed-form potential to differentiate — for the zeta split `Ta` depends on the stretch through `zeta_s(lambda_dot)`. `ZetaSplitUFL.strain_energy` raises rather than returning something plausible.
- `ZetaSplitUFL.Ta(lmbda)` is a **method** returning a UFL expression; the output `Function` is `active_tension`. Do not merge those names — a `Function` called with a float does not raise, it hangs in point evaluation.
- The zeta-split backend is monolithic-in-Newton by construction: `Ta` is a UFL expression of the current displacement, so Newton re-linearizes it every iteration. That is what makes it stable, and it is exactly what a NumPy contraction model cannot do.
- `backends/segregated.py` holds `CrossbridgeSegregated`, the Ca_i split. Because crossbridge is NumPy it cannot be re-integrated inside Newton, so this interface is segregated — which is *not convergent* without stabilization once active stiffness exceeds passive stiffness (R&Q 2021). It therefore delegates its stress form to `pulse.StabilizedActiveStress`. Its `step()` captures the stretch, advances the ODE with it, and sets `lmbda_prev` to it **in one place**; `post_solve()` is deliberately empty so those two can never diverge.
- `units.py` holds the EP/crossbridge/mechanics conversions (ms↔s, mM↔µM, kPa, `J_TRPN`). None of these mismatches raises. The one that is not a plain scale factor is `active_stiffness_to_mechanics`: `Ka` is reported per unit of crossbridge's `Λ = SL/SL0` but consumed against the solver's `λ`, so it needs `× SL_ref/SL0`. With the default `SL_ref == SL0` the factor is 1, which is exactly why omitting it goes unnoticed.
- `RDQ18` defines no `SL0` (it uses `SL` directly in `Chi(SL)`), so `CrossbridgeSegregated` requires an explicit `SL_ref` for it rather than guessing one — a guess would silently move the model along its force-length curve.
- **Active-stress convention:** backends take a `formulation`. The default is `stretch`, the Regazzoni & Quarteroni normalization `P_a = Ta·F f0⊗f0/|F f0|`, under which `|P_a f0| = Ta` regardless of stretch. The historical simcardems form `invariant` (`P_a = Ta·F f0⊗f0`) is larger by a factor of λ and is retained only to reproduce pre-change results — results generated with it are not comparable to new ones. On a 1-element contraction the difference is ~31% in stress but only ~2.4% in peak shortening, since Holzapfel-Ogden stiffens exponentially; don't assume the discrepancy is negligible on other geometries.
- `land.py` is now a deprecated shim re-exporting `ZetaSplitUFL` as `LandModel` with a `DeprecationWarning`. `mechanicsproblem.py` is gone: its `_material_form` is redundant now that backends supply `S`, and its `post_solve` moved onto the backend.

- `zero_d.py` replaces the FEM mechanics with R&Q's 0D tissue model, so the same contraction models can be driven through all three coupling schemes (monolithic / segregated / stabilized) with no mesh. The monolithic run is a genuine reference solution — it root-finds the strain at which the ODE is advanced so the balance holds — and `tests/test_zero_d_coupling.py` uses it to assert the claim the whole design rests on: the naive scheme's error *grows* under refinement while the stabilized one converges at first order. **It works in seconds and pascals**, following the paper, unlike the rest of the package which uses ms.
- The instability only appears in the **quasistatic** regime (`M = sigma = 0`) and needs `Ka > Kp`. With R&Q's dynamic parameters at small `dt` the inertia term `M/dt^2` swamps `Kp` and hides it — so a test that fails to show oscillation may just be in the wrong regime, not fixed.

**2. Runtime EP/mechanics coupling (`ode_model.RuntimeODEModel`, `interpolation.py`)**

`RuntimeODEModel` wraps the loaded `ep_model` module dict and owns two `MissingValue` instances (`missing_mech`, `missing_ep`) — one per direction of data flow between the EP mesh and the mechanics mesh. Each `MissingValue` holds dolfinx `Function`s on both meshes/function spaces plus a `TransferOperator` (`interpolation.py`) that precomputes non-matching-mesh interpolation data (`dolfinx.fem.create_interpolation_data` / `interpolate_nonmatching`) so values can be pushed between the (generally non-matching) EP and mechanics meshes every step.

**3. Mechanics problem**

There is no problem subclass: stock `pulse.StaticProblem` is used. The active stress comes from the backend's `S`, and what used to be `MechanicsProblem.post_solve` — recomputing `lmbda`/`dLambda`/`Ta` from the solved displacement and advancing the Land state — is now `backend.post_solve()`, called by the caller after each solve.

**4. Orchestration (`controller.py`)**

`SimulationController.step()` is the whole time-stepping algorithm: run `dt_mech / dt_ep` EP micro-steps (with an `ep_callback` hook per micro-step), transfer the resulting EP state to the mechanics missing-values, solve one mechanics Newton step, call `backend.post_solve()` (updates active-model kinematics + advances Land state), then transfer mechanics state (stretch etc.) back to the EP side for mechano-electric feedback next round.

**5. Output (`datacollector.py`)**

`DataCollector` + `Timers` handle writing point/field EP and mechanics data (via dolfinx's function evaluation at points) and simulation timing, driven by callbacks passed into `SimulationController.step`.

`numerical_experiments/strong_coupling_zetasplit/main.py` is a full worked example wiring all of the above together end-to-end (mesh generation via `cardiac-geometriesx`, `beat` conductivities/stimulus/solver setup, `pulse` material/BCs, the controller loop, and `DataCollector`) — read it to see how the pieces connect in practice rather than inferring it from the library modules alone.

`config.py` defines pydantic models (using `pydantic_pint` for unit-aware fields) for simulation configuration; most of it beyond `Conductivity` is currently commented out / unimplemented.

## Testing notes

Tests avoid needing a real EP/mechanics stack where possible: `test_coupled_system.py` drives `SimulationController` with a `DummyEPSolver`/`MockODEModel`, `test_isolated_ep.py` builds a minimal synthetic `.ode` file on the fly (via `generate_ode_code`) to test mechano-electric feedback, and `test_isolated_mechanics.py` / `test_stability.py` exercise `ZetaSplitUFL` directly against `pulse.StaticProblem`.

`test_backends.py` carries the equivalence record for the port: it writes out, by hand, the stress form that the deleted `MechanicsProblem._material_form` built, and requires the backend to reproduce it. Since that form no longer exists in the package, the test is the only remaining copy of it — don't delete it when adding backends.
