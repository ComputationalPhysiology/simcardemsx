"""One-element scheme comparison: monolithic, naive segregated and stabilized.

No EP: the cellular inputs are prescribed, so each run isolates the coupling scheme.
Three studies, each run under every scheme and time step, with a monolithic reference
at a much smaller step:

* ``static-caisplit``: ``pulse.StaticProblem`` on one element, the Ca_i split driven
  by a prescribed calcium transient (gate 1 of ``tests/test_monolithic_coupling.py``);
* ``static-zetasplit``: the same element, the zeta split driven by prescribed
  ``XS``/``XW`` (gate 4);
* ``dynamic-zetasplit``: the damped 1 cm element of ``pulse.DynamicProblem``, the zeta
  split at 10 times gate 4's inputs (gate D2).

The builders and inputs below mirror ``tests/conftest.py`` (``_mechanics``,
``_dynamic_mechanics``, ``_rollers``, ``calcium``, ``_twitch``, ``_zetasplit_inputs``)
and ``tests/test_dynamic_coupling.py`` (``D2_INPUT_SCALE``, ``_d2_inputs``), by value.
They are copies, not imports: keep them in step with the gates. The step loop is that
of ``_run`` in ``tests/test_monolithic_coupling.py``.

Each run goes to ``<output-dir>/element/<study>/<scheme>/dt<dt>/`` with ``steps.csv``,
``snapshots.npz`` (every step), ``run.json`` (which holds the SNES residual history
of the step ending nearest 20 ms as ``newton_residuals``) and ``timings.json``. A step
that fails, or leaves a non-finite displacement, ends that run; the next one starts.

Usage::

    python3 element.py [--output-dir DIR] [--study NAME ...]

Serial only.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from mpi4py import MPI

import dolfinx
import numpy as np
import pulse

from simcardemsx.backends import GeneratedActivation
from simcardemsx.ode_model import load_ode_modules

HERE = Path(__file__).resolve().parent
# scheme_comparison sits next to this script, not in the installed package.
sys.path.insert(0, str(HERE.parent))
from scheme_comparison.record import Recorder

ODEFILES_DIR = HERE.parent / "odefiles"
SCHEMES = ("monolithic", "segregated", "stabilized")

#: Absolute Newton tolerance, as ``conftest.SNES_ATOL``: at rest the first residual is
#: already at round-off, which a pure relative tolerance cannot converge.
SNES_ATOL = 1e-9

#: Factor on the zeta split's inputs in the dynamic study, as ``D2_INPUT_SCALE``.
D2_INPUT_SCALE = 10.0

#: The step whose SNES residual history is stored ends nearest this time, in ms.
RESIDUAL_T_MS = 20.0


def calcium(t: float) -> float:
    """Prescribed Ca_i transient in mM: 1e-4 at rest, peaking at 1e-3 at t = 25 ms."""
    tau = max(t - 5.0, 0.0)
    return 1e-4 + 9e-4 * (tau / 20.0) * np.exp(1.0 - tau / 20.0)


def _twitch(t: float) -> float:
    """Unit twitch shape: 0 until t = 5 ms, peaking at 1 at t = 25 ms."""
    tau = max(t - 5.0, 0.0)
    return (tau / 20.0) * np.exp(1.0 - tau / 20.0)


def _caisplit_inputs(t: float) -> dict[str, float]:
    return {"cai": calcium(t)}


def _zetasplit_inputs(t: float) -> dict[str, float]:
    b = _twitch(t)
    return {"XS": 0.01 * b, "XW": 0.005 * b}


def _d2_inputs(t: float) -> dict[str, float]:
    return {name: D2_INPUT_SCALE * value for name, value in _zetasplit_inputs(t).items()}


def _rollers(mesh: dolfinx.mesh.Mesh):
    """Roller conditions: ``u_i = 0`` on the face ``x_i = 0``, for i = 0, 1, 2."""

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        fdim = mesh.topology.dim - 1
        bcs = []
        for i in range(mesh.geometry.dim):
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, i=i: np.isclose(x[i], 0.0),
            )
            dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), fdim, facets)
            bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
        return bcs

    return dirichlet_bc


def _static_mechanics(
    mech_module: ModuleType,
    scheme: str,
    quadrature_degree: int = 2,
) -> tuple[pulse.StaticProblem, GeneratedActivation]:
    """The gate-1 element: a unit cube, Holzapfel-Ogden, incompressible, rollers."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    backend = GeneratedActivation(
        mech_module,
        mesh,
        f0,
        quadrature_degree=quadrature_degree,
        scheme=scheme,  # type: ignore[arg-type]
    )
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=pulse.Incompressible(),
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=pulse.BoundaryConditions(dirichlet=[_rollers(mesh)]),
        parameters={"base_bc": pulse.problem.BaseBC.free, "petsc_options": petsc_options},
    )
    return problem, backend


def _dynamic_mechanics(
    mech_module: ModuleType,
    scheme: str,
    dt_ms: float,
    quadrature_degree: int = 2,
) -> tuple[pulse.DynamicProblem, GeneratedActivation]:
    """The D2 element: a 1 cm cube, Compressible, Viscous (eta 100 Pa s), rho 1e3."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    mesh.geometry.x[:] *= 0.01
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    backend = GeneratedActivation(
        mech_module,
        mesh,
        f0,
        quadrature_degree=quadrature_degree,
        scheme=scheme,  # type: ignore[arg-type]
    )
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous(),
    )
    petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    problem = pulse.DynamicProblem(
        model=model,
        geometry=geometry,
        bcs=pulse.BoundaryConditions(dirichlet=[_rollers(mesh)]),
        parameters={"dt": pulse.Variable(dt_ms * 1e-3, "s"), "petsc_options": petsc_options},
    )
    return problem, backend


@dataclass(frozen=True)
class Study:
    name: str
    split: str
    dynamic: bool
    inputs: Callable[[float], dict[str, float]]
    t_end: float
    dts: tuple[float, ...]
    reference_dt: float
    eta_Pa_s: float
    rho_kg_m3: float
    h_m: float


STUDIES = {
    s.name: s
    for s in (
        Study(
            "static-caisplit",
            "caisplit",
            False,
            _caisplit_inputs,
            40.0,
            (1.0, 0.5, 0.25, 0.1, 0.05),
            0.01,
            0.0,
            0.0,
            1.0,
        ),
        Study(
            "static-zetasplit",
            "zetasplit",
            False,
            _zetasplit_inputs,
            80.0,
            (1.0, 0.5, 0.25, 0.1, 0.05),
            0.01,
            0.0,
            0.0,
            1.0,
        ),
        Study(
            "dynamic-zetasplit",
            "zetasplit",
            True,
            _d2_inputs,
            150.0,
            (2.0, 1.0, 0.5, 0.25),
            0.05,
            100.0,
            1e3,
            0.01,
        ),
    )
}


def _kp_kPa() -> float:
    """Uniaxial small-strain stiffness 3a of Holzapfel-Ogden's isotropic term, in kPa."""
    a = pulse.HolzapfelOgden.transversely_isotropic_parameters()["a"]
    return float(3 * a.to_base_units() / 1e3)


def run_one(
    study: Study,
    mech_module: ModuleType,
    scheme: str,
    dt: float,
    outdir: Path,
) -> tuple[str | None, float | None]:
    """Run one (study, scheme, dt) and write its files; return ``(failure, t_fail_ms)``."""
    t0 = time.perf_counter()
    problem: pulse.StaticProblem | pulse.DynamicProblem
    if study.dynamic:
        problem, backend = _dynamic_mechanics(mech_module, scheme, dt)
    else:
        problem, backend = _static_mechanics(mech_module, scheme)
    recorder = Recorder(
        backend,
        outdir,
        snapshot_every_ms=dt,
        run_info={
            "study": study.name,
            "geometry": "dynamic-1cm-element" if study.dynamic else "static-unit-element",
            "split": study.split,
            "scheme": scheme,
            "dt_mech_ms": dt,
            "t_end_ms": study.t_end,
            "regime": {
                "Kp_kPa": _kp_kPa(),
                "eta_Pa_s": study.eta_Pa_s,
                "rho_kg_m3": study.rho_kg_m3,
                "h_m": study.h_m,
            },
        },
    )
    solver = problem.problem.solver
    n_steps = round(study.t_end / dt)
    n_residual = round(RESIDUAL_T_MS / dt) - 1
    setup_s = time.perf_counter() - t0

    failure: str | None = None
    t_fail: float | None = None
    mech_s = 0.0
    t1 = time.perf_counter()
    for n in range(n_steps):
        t_n, t_next = n * dt, (n + 1) * dt
        backend.t.value = t_n  # type: ignore[assignment]
        backend.dt.value = dt  # type: ignore[assignment]
        for name, value in study.inputs(t_next).items():
            backend.inputs[name].x.array[:] = value

        if n == n_residual:
            solver.setConvergenceHistory()
        ts = time.perf_counter()
        ok = problem.solve(raise_on_failure=False)
        mech_s += time.perf_counter() - ts
        with np.errstate(over="ignore", invalid="ignore"):
            finite = bool(np.all(np.isfinite(problem.u.x.array)))
        if n == n_residual:
            recorder.run_info["newton_residuals"] = [
                float(r) for r in solver.getConvergenceHistory()[0]
            ]
        if not ok or not finite:
            failure = "Newton did not converge" if not ok else "non-finite displacement"
            t_fail = t_next
            break

        backend.post_solve()
        recorder.step(t_next, solver.getIterationNumber())

    loop_s = time.perf_counter() - t1
    timings = {
        "setup_s": setup_s,
        "mech_s": mech_s,
        "loop_s": loop_s,
        "total_s": time.perf_counter() - t0,
    }
    recorder.finish(failure=failure, t_fail_ms=t_fail, timings=timings)
    (outdir / "timings.json").write_text(json.dumps(timings, indent=2))
    return failure, t_fail


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "output",
        help="Where the runs go, under element/<study>/<scheme>/dt<dt>/ (default: %(default)s).",
    )
    parser.add_argument(
        "--study",
        nargs="+",
        choices=sorted(STUDIES),
        default=sorted(STUDIES),
        help="Studies to run (default: all).",
    )
    args = parser.parse_args(argv)

    modules = {
        split: load_ode_modules(
            ODEFILES_DIR / f"ToRORd_dynCl_endo_{split}.ode",
            HERE / "generated_odes" / split,
        )
        for split in sorted({STUDIES[name].split for name in args.study})
    }

    for name in args.study:
        study = STUDIES[name]
        mech_module = modules[study.split].mechanics
        runs = [(s, dt) for s in SCHEMES for dt in study.dts]
        runs.append(("monolithic", study.reference_dt))
        for scheme, dt in runs:
            outdir = args.output_dir / "element" / name / scheme / f"dt{dt:g}"
            t0 = time.perf_counter()
            failure, t_fail = run_one(study, mech_module, scheme, dt, outdir)
            status = "ok" if failure is None else f"FAILED at {t_fail:g} ms ({failure})"
            print(f"{name} {scheme} dt={dt:g}: {status} [{time.perf_counter() - t0:.1f} s]")


if __name__ == "__main__":
    main()
