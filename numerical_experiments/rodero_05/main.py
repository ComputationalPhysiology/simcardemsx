"""Stage B: physcardems' ``em_tref7`` run of ``rodero_05``, on simcardemsx and pulse.

physcardems' ``cases/rodero_05/run_rodero_em_tref7.py`` (with ``configs/elife/
em_tref7.toml``) couples EP and mechanics on a coarse biventricular mesh of the Rodero
et al. cohort, with valve plugs closing the cavities, and drives both ventricles
through a five-phase cardiac cycle. This is the same run on this package: beat's
monodomain for the EP half of the zeta split of ToR-ORd-Land,
:class:`~simcardemsx.backends.GeneratedActivation` for its ``mechanics`` half, a
``pulse.DynamicProblem`` whose LV and RV are controlled cavities, and
``pulse.cycle.CycleController`` switching their constraints, driven by
:class:`~simcardemsx.controller.SimulationController` through
:class:`~simcardemsx.mechanics.Cycle`. The case is loaded by ``case.py``, from
physcardems' own case directory (``--case-dir``).

The set-up, following the run script:

- **EP** on P1 over the whole mesh: the zeta split's EP half, each node starting from
  the single-cell steady state of its ToR-ORd cell type, with its cell type and its
  ``i_Stim_Start`` set to the case's activation time, so ToR-ORd's own stimulus
  activates the tissue (no PDE stimulus); Niederer's conductivities and chi, C_m = 1
  uF/cm^2; 0.05 ms steps.
- **Parameters**: every Land parameter goes to each half of the split that declares
  it, and ``PCa_b`` x 2 to the EP half (:func:`place_land`).
- **Activation**: ``GeneratedActivation``, monolithic, its states on quadrature of
  degree 4, ``Ta`` masked to 0 in the valve plugs (``tension_scale``).
- **Mechanics**: ``pulse.DynamicProblem`` (rho 1e3 kg/m^3, alpha_m 0.2, alpha_f 0.4,
  pulse's defaults as in the run script), dt 2 ms; Holzapfel-Ogden with physcardems'
  parameters, its four moduli x 3 in the valve plugs through DG0 material parameters;
  ``Compressible2`` (kappa 1e6 Pa) and ``Viscous`` (eta 100 Pa s); an epicardial spring
  (1e8 Pa/m) and damper (5e3 Pa s/m), and nothing on the base; physcardems' solver
  options: GMRES preconditioned by a lagged LU (lag 20, persistent), and a fresh one
  after each phase switch.
- **Start**: one solve at zero cavity pressure (the unloaded solve; like the run
  script's, a ``DynamicProblem`` step from rest), then the cycle starts every cavity in
  PRELOAD at t = 0.

Known differences from physcardems, kept rather than removed:

- Values crossing back to EP (``Zetas``, ``Zetaw``, lambda) are averaged onto P1, not
  point-interpolated from DG1.
- The contraction states live on quadrature points of degree 4, not in DG1.
- The contraction model is the generated ``mechanics`` component of the ``.ode``, not
  physcardems' hand-written ``ZetaSplitConstDt``:

  - it integrates ``Cd``, a passive-tension state that does not enter ``Ta``;
  - its ``Ta`` is not clipped at 0 (``ZetaSplitConstDt`` clips the active fraction),
    so rapid shortening can make it negative: ``log.csv`` reports the minimum;
  - its ``Zetas`` and ``Zetaw`` are not clamped at -1, as ``ZetaSplitConstDt``'s are.

- The unloaded solve is at rest and is accepted. Here the backend's step is the
  identity until the first coupled step (its ``dt`` starts at 0), and the solve is
  accepted like any other (``post_solve``, then lambda back to EP), so the first
  coupled step's stretch rate is ``(lambda_1 - lambda_0) / dt``. physcardems'
  ``ZetaSplitConstDt`` has a fixed 2 ms step, so its unloaded solve already sees a
  stretch rate from lambda = 1, and the solve is not accepted by the active model:
  pulse keeps its u, v and a, but ``zeta.post_solve`` and the transfer back to EP are
  skipped, so its first coupled step again starts from lambda = 1, and EP's lambda
  stays 1 through it.
- The mechanics' initial contraction states are the ``.ode``'s defaults. physcardems
  sets ``Zetas``/``Zetaw`` from the steady states; those are 0.0 for every cell type,
  as ``Cd`` is, the same as the defaults, and :func:`check_mechanics_initial_states`
  stops the run if a case says otherwise.
- ``log.csv``'s phase columns are the phase each step was solved under;
  physcardems' are the phase after the step's switch, i.e. for the next step.
- det F is sampled at the degree-4 quadrature points, and ``n_detF_nonpositive``
  counts the points where it is not positive; physcardems samples each cell's
  vertices and centroid, and counts cells.
- ``log.csv`` has no ``J_max`` or ``wall_ep_s`` column, both of which physcardems'
  has, and physcardems' ``active_stats.csv`` (``Ta`` percentiles, and ``Ta`` and
  lambda medians per cell type) is not written.
- physcardems' pulse raised on a solve that did not converge
  (``snes_error_if_not_converged``), and its cycle controller caught that and
  retried. pulse 0.10's ``solve`` returns ``False`` instead, when the SNES converged reason is
  not positive, and ``pulse.cycle`` retries on that. The behaviour is the same: one
  retry from the rolled-back state, with a fresh factorization.
- pulse is 0.10 (active stress at the end of the step, controlled cavities,
  ``pulse.cycle``), not physcardems' 0.7+26. ``pulse.cycle`` ports the part
  of physcardems' cycle controller this run uses: not the ``ejection_pressure``
  valve override, the legacy filling laws or the FILLING stall counter, none of which
  em_tref7 sets or, with a prescribed inflow, reaches.
- Left out: the pseudo-ECG, checkpoints and restarts, the VTX field output and the
  mid-ventricular slice statistics, and ``sf_IKs`` (read by physcardems, never used).

Output, in ``--output-dir``:

- ``log.csv``: one row after the unloaded solve (``t_ms`` 0) and one per mechanics step:
  the maximum and minimum ``Ta`` and the range of lambda over the myocardium's
  quadrature points; per cavity the phase the step was solved under
  (0 PRELOAD, 1 IVC, 2 EJECTION, 3 IVR, 4 FILLING), V, P, the Windkessel's compliance
  pressure P_c and the outflow Q; the minimum of det F over every quadrature point, and
  how many are not positive; the last solve's Newton iterations, SNES converged reason
  and linear iterations, and how many solves the step took (2 = one retry); EP's
  membrane potential range; the step's wall time.
- ``pv_loops.png``: the PV loops with EF, where EDV is the volume at the last switch
  into IVC and ESV the minimum since; pressures, volumes, ``Ta`` and phases in time.
- ``summary.json``: EDV, ESV, EF and peak pressure per ventricle; the phase switches;
  the Newton, SNES-reason and retry counts; min det F and min ``Ta``; the Stage B
  acceptance criteria; where the Land parameters went. Among the criteria,
  ``reached_t_end`` says whether the last row is at ``--t-end``, in whole steps (the
  summary is also rewritten while the run goes on), and
  ``newton_converged_every_step`` is false if the run stopped on any exception,
  ``KeyboardInterrupt`` included, which ``failure`` then names. A step whose retry
  converged counts as converged; ``newton`` reports the retries.
- ``timings.json``: wall time in the EP ODE and PDE steps and the mechanics solves, the
  set-up (code generation, loading, form compilation, the unloaded solve), the loop,
  and the whole run.

Serial only. The 800 ms run takes hours; ``--t-end 20`` is the smoke test (physcardems'
``em_tref7_short.toml``).
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

from mpi4py import MPI

import beat
import dolfinx
import numpy as np
import pulse
import ufl
from pulse.cycle import CycleController, Phase

from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Cycle
from simcardemsx.ode_model import ODEModules, load_ode_modules

HERE = Path(__file__).resolve().parent
# case.py sits next to this script, not in the installed package. As rodero_05.case,
# with numerical_experiments/ on the path, it has the name mypy gives it too.
sys.path.insert(0, str(HERE.parent))
from rodero_05.case import CELLTYPES, PCL_MS, QUADRATURE_DEGREE, TAGS, Case, load_case

logger = logging.getLogger(__name__)

ODEFILE = HERE.parent / "odefiles" / "ToRORd_dynCl_endo_zetasplit.ode"
DEFAULT_CASE_DIR = HERE.parent.parent / "third-party" / "physcardems" / "cases" / "rodero_05"

CHAMBERS = ("LV", "RV")

#: Time steps and end time, in ms (run script, lines 44-46).
DT_MECH = 2.0
DT_EP = 0.05
T_END = 800.0

#: Holzapfel-Ogden, moduli in kPa and exponents dimensionless (run script, line 68).
MATERIAL_PARAMS = {
    "a": 0.61,
    "a_f": 1.56,
    "b": 7.5,
    "b_f": 35.31,
    "a_s": 0.7,
    "b_s": 33.24,
    "a_fs": 0.46,
    "b_fs": 5.09,
}
#: The moduli among them, which the valve plugs' stiffness scale multiplies.
MODULI = ("a", "a_f", "a_s", "a_fs")
#: ``FIBRE_COMPRESSION_RESISTANCE = False`` (line 69): the fibre and sheet terms act in
#: tension only, so ``use_heaviside`` and ``use_subplus`` are both True (lines 266-268).
FIBRE_COMPRESSION_RESISTANCE = False
#: ``Compressible2``'s bulk modulus, in Pa (line 70).
KAPPA_PA = 1e6

#: The epicardial spring, in Pa/m, and damper, in Pa s/m (lines 277-282).
EPI_SPRING = 1e8
EPI_DAMPING = 5e3

#: The steady-state lag of the lagged LU (line 56); the cycle controller refreshes it
#: after each phase switch.
PRECONDITIONER_LAG = 20
#: physcardems' solver options, on top of pulse's ``DynamicProblem`` defaults (lines
#: 287-300).
PETSC_OPTIONS: dict[str, Any] = {
    "snes_monitor": None,
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "ksp_max_it": 100,
    "ksp_rtol": 1e-6,
    "ksp_atol": 1e-14,
    "ksp_converged_reason": None,
    "snes_lag_preconditioner": PRECONDITIONER_LAG,
    "snes_lag_preconditioner_persists": True,
}

#: Membrane capacitance, in uF/cm^2 (line 250).
C_M_UF_PER_CM2 = 1.0

#: How often, in mechanics steps, the summary, plot and timings are rewritten during
#: the run (``log.csv`` is rewritten after every step).
WRITE_EVERY = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help="physcardems' rodero_05 case directory (default: %(default)s).",
    )
    parser.add_argument(
        "--tref",
        type=float,
        default=7.0,
        help="Scale of Land's Tref, 120 kPa (default: %(default)s, em_tref7). The "
        "steady states do not depend on it.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=T_END,
        help=f"End time in ms (default: %(default)s, one beat). Rounded to a whole number "
        f"of {DT_MECH} ms mechanics steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "output",
        help="Output directory (default: %(default)s).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Parameters and initial states
# ---------------------------------------------------------------------------


def place_land(land: dict[str, float], modules: ODEModules) -> dict[str, dict[str, float]]:
    """Each Land parameter, for every half of the split that declares it.

    physcardems writes the whole Land set into the EP parameter array wherever the EP
    half has the name (``parameters.py``, ``apply_to_ode_parameters``), and hands the
    mechanics-side subset to its contraction model. Here the same rule goes both ways:
    ``"ep"`` holds the names the EP module declares, ``"mechanics"`` those the
    ``mechanics`` module declares, and a name may be in both.

    Raises
    ------
    KeyError
        If a name is declared by neither half: it would change nothing.
    """
    placed = {
        "ep": {k: v for k, v in land.items() if k in modules.ep.parameter},
        "mechanics": {k: v for k, v in land.items() if k in modules.mechanics.parameter},
    }
    unused = sorted(set(land) - set(placed["ep"]) - set(placed["mechanics"]))
    if unused:
        raise KeyError(f"Land parameters {unused} are declared by neither half of the split")
    return placed


def check_mechanics_initial_states(
    initial_states: dict[int, dict[str, float]],
    module,
) -> dict[str, float]:
    """The ``mechanics`` component's states in the steady states, checked against the
    ``.ode`` defaults ``GeneratedActivation`` starts every quadrature point from.

    Returns the defaults, by state name.

    Raises
    ------
    ValueError
        If a cell type's steady state differs from them: the backend's states would
        then need setting per quadrature point, from the cell type, which this example
        does not do.
    """
    names = sorted(module.state, key=module.state.__getitem__)
    defaults = dict(zip(names, map(float, module.init_state_values())))
    differ = {
        (celltype, name): states[name]
        for celltype, states in initial_states.items()
        for name, default in defaults.items()
        if states[name] != default
    }
    if differ:
        raise ValueError(
            f"The steady states' contraction states {differ} differ from the .ode's "
            f"defaults {defaults}, which the backend starts every quadrature point from.",
        )
    return defaults


def make_ep_solver(
    case: Case,
    ep_module,
    ep_land: dict[str, float],
) -> beat.MonodomainSplittingSolver:
    """beat's monodomain on the case's mesh, P1, no PDE stimulus (run script, lines 201-258).

    States, missing variables (the contraction states EP reads) and parameters have one
    column per P1 dof: each dof starts from its cell type's steady state, and has its
    own cell type and ``i_Stim_Start`` (its activation time). The transfer plan writes
    lambda and the backend's outputs back into these arrays in place.
    """
    mesh = case.geometry.mesh
    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    num_points = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
    celltype = case.celltype
    if celltype.size != num_points:
        raise ValueError(f"{celltype.size} cell types for {num_points} P1 dofs")

    state_names = sorted(ep_module.state, key=ep_module.state.__getitem__)
    missing_names = sorted(ep_module.missing, key=ep_module.missing.__getitem__)
    states = np.zeros((len(state_names), num_points))
    missing = np.zeros((len(missing_names), num_points))
    for c in CELLTYPES:
        steady = case.initial_states[c]
        states[:, celltype == c] = np.array([steady[s] for s in state_names])[:, None]
        missing[:, celltype == c] = np.array([steady[s] for s in missing_names])[:, None]

    parameters = np.repeat(
        ep_module.init_parameter_values(i_Stim_Period=PCL_MS, lmbda=1.0, dLambda=0.0)[:, None],
        num_points,
        axis=1,
    )
    parameters[ep_module.parameter_index("celltype")] = celltype.astype(float)
    parameters[ep_module.parameter_index("i_Stim_Start")] = case.lat_ms
    for name, value in ep_land.items():
        parameters[ep_module.parameter_index(name)] = value
    for name, scale in case.ep_scales.items():
        parameters[ep_module.parameter_index(name)] *= scale

    conductivities = beat.conductivities.default_conductivities("Niederer")
    M = beat.conductivities.define_conductivity_tensor(f0=case.f0, **conductivities)
    C_m: float = (C_M_UF_PER_CM2 * beat.units.ureg("uF/cm**2")).to("uF/m**2").magnitude
    pde = beat.MonodomainModel(
        time=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        mesh=mesh,
        M=M,
        I_s=None,
        C_m=C_m,
    )
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(W, name="v_ode"),
        v_pde=pde.state,
        fun=ep_module.generalized_rush_larsen,
        init_states=states,
        parameters=parameters,
        num_states=states.shape[0],
        v_index=ep_module.state_index("v"),
        missing_variables=missing,
        num_missing_variables=missing.shape[0],
    )
    return beat.MonodomainSplittingSolver(pde=pde, ode=ode)


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def material(case: Case) -> pulse.HolzapfelOgden:
    """Holzapfel-Ogden, each modulus a DG0 field: the modulus times the case's stiffness
    scale (x 3 in the valve plugs).

    physcardems multiplies the whole passive stress by the scale (its ``ScaledModel``);
    Holzapfel-Ogden's stress is linear in its four moduli, so this is the same stress,
    through pulse's own spatial material parameters.
    """
    scale = case.stiffness_scale

    def parameter(name: str) -> pulse.Variable:
        value = MATERIAL_PARAMS[name]
        if name not in MODULI:
            return pulse.Variable(value, "dimensionless")
        field = dolfinx.fem.Function(scale.function_space, name=name)
        field.x.array[:] = value * scale.x.array
        return pulse.Variable(field, "kPa")

    return pulse.HolzapfelOgden(
        f0=case.f0,
        s0=case.s0,
        **{name: parameter(name) for name in MATERIAL_PARAMS},
        use_heaviside=not FIBRE_COMPRESSION_RESISTANCE,
        use_subplus=not FIBRE_COMPRESSION_RESISTANCE,
    )


def make_problem(case: Case, backend: GeneratedActivation) -> pulse.DynamicProblem:
    """The run script's mechanics (lines 264-303), its two cavities controlled."""
    geometry = case.geometry
    mesh = geometry.mesh
    model = pulse.CardiacModel(
        material=material(case),
        active=backend,
        compressibility=pulse.compressibility.Compressible2(kappa=pulse.Variable(KAPPA_PA, "Pa")),
        viscoelasticity=pulse.viscoelasticity.Viscous(),
    )

    def constant(value: float) -> dolfinx.fem.Constant:
        return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value))

    bcs = pulse.BoundaryConditions(
        robin=(
            pulse.RobinBC(value=pulse.Variable(constant(EPI_SPRING), "Pa / m"), marker=TAGS["EPI"]),
            pulse.RobinBC(
                value=pulse.Variable(constant(EPI_DAMPING), "Pa s/ m"),
                marker=TAGS["EPI"],
                damping=True,
            ),
        ),
    )
    petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"] | PETSC_OPTIONS
    return pulse.DynamicProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        cavities=[
            pulse.problem.Cavity(marker=c, control=pulse.problem.CavityControl(mesh))
            for c in CHAMBERS
        ],
        parameters={"dt": pulse.Variable(DT_MECH * 1e-3, "s"), "petsc_options": petsc_options},
    )


class SolveLog:
    """Wraps ``problem.solve`` to time every call and keep its outcome.

    ``pulse.cycle.CycleController`` retries a failed solve once, so a mechanics step
    is one or two calls: each is recorded with its Newton iterations, SNES converged
    reason and linear iterations.
    """

    def __init__(self, problem: pulse.StaticProblem):
        self.snes = problem.problem.solver
        self.attempts: list[dict[str, Any]] = []
        self.seconds = 0.0
        self._solve: Callable[..., bool] = problem.solve
        problem.solve = self  # type: ignore[method-assign]

    def __call__(self, *args, **kwargs) -> bool:
        start = time.perf_counter()
        try:
            converged = self._solve(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start
        self.attempts.append(
            {
                "iterations": int(self.snes.getIterationNumber()),
                "reason": cast(int, self.snes.getConvergedReason()),
                "linear_iterations": int(self.snes.getLinearSolveIterations()),
                "converged": bool(converged),
            },
        )
        return converged


def accumulate_time(fn: Callable, totals: dict[str, float], key: str) -> Callable:
    """Wrap ``fn`` so that the wall time spent in it is added to ``totals[key]``."""

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            totals[key] += time.perf_counter() - start

    return timed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def ejection_fraction(phase: np.ndarray, V: np.ndarray) -> dict[str, float] | None:
    """EDV at the last switch into IVC, ESV the smallest volume since, and EF.

    ``phase`` is the phase each row was solved under, so the switch is decided at the
    row before the first IVC row of the last run of IVC rows, and that row's volume is
    the one IVC then holds. ``None`` if the cycle never reached IVC.
    """
    is_ivc = phase == Phase.ISOVOLUMIC_CONTRACTION
    starts = np.flatnonzero(is_ivc[1:] & ~is_ivc[:-1]) + 1
    if starts.size == 0:
        return None
    switch = starts[-1] - 1
    EDV, ESV = float(V[switch]), float(V[switch:].min())
    return {"EDV_mL": EDV, "ESV_mL": ESV, "EF_percent": 100.0 * (EDV - ESV) / EDV}


def phase_sequence(phase: np.ndarray) -> list[str]:
    """The distinct phases in the order they were solved under."""
    names = [Phase(int(p)).name for p in phase]
    return [name for i, name in enumerate(names) if i == 0 or name != names[i - 1]]


def summarise(
    columns: dict[str, np.ndarray],
    switches: list[dict[str, Any]],
    attempts: list[list[dict[str, Any]]],
    failure: str | None,
    land_placement: dict[str, list[str]],
    t_end_ms: float,
) -> dict[str, Any]:
    """The acceptance numbers of the run so far; ``t_end_ms`` is where it is to end."""
    summary: dict[str, Any] = {"t_end_ms": float(columns["t_ms"][-1]), "failure": failure}
    five_phases = [p.name for p in Phase]
    for c in CHAMBERS:
        P = columns[f"P_{c}_kPa"]
        sequence = phase_sequence(columns[f"phase_{c}"])
        summary[c] = {
            "ejection": ejection_fraction(columns[f"phase_{c}"], columns[f"V_{c}_mL"]),
            "peak_P_kPa": float(P.max()),
            "peak_P_mmHg": float(P.max() * 1e3 / 133.322),
            "t_peak_P_ms": float(columns["t_ms"][np.argmax(P)]),
            "phases": sequence,
            "five_phases_in_order": sequence[:5] == five_phases,
        }
    summary["phase_switches"] = switches

    # Every step after the unloaded solve (row 0).
    steps = attempts[1:]
    final = [a[-1] for a in steps]
    reasons: dict[str, int] = {}
    for a in final:
        reasons[str(a["reason"])] = reasons.get(str(a["reason"]), 0) + 1
    iterations = np.array([a["iterations"] for a in final]) if final else np.zeros(0)
    summary["newton"] = {
        "steps": len(steps),
        "unloaded_solve": attempts[0][-1] if attempts else None,
        "iterations_min": int(iterations.min()) if iterations.size else None,
        "iterations_mean": float(iterations.mean()) if iterations.size else None,
        "iterations_max": int(iterations.max()) if iterations.size else None,
        # 2 and 3: converged on the residual (absolute, relative); 4: on the step size.
        "final_reasons": reasons,
        "retries": sum(len(a) - 1 for a in steps),
        "steps_with_retry_ms": [float(t) for t, a in zip(columns["t_ms"][1:], steps) if len(a) > 1],
        "all_attempts_reasons": sorted({str(x["reason"]) for a in steps for x in a}),
    }
    detF = columns["detF_min"]
    summary["detF_min"] = float(detF.min())
    summary["t_detF_min_ms"] = float(columns["t_ms"][np.argmin(detF)])
    summary["Ta_min_kPa"] = float(columns["Ta_min_kPa"].min())
    summary["Ta_max_kPa"] = float(columns["Ta_max_kPa"].max())
    summary["lmbda_min"] = float(columns["lmbda_min"].min())
    summary["lmbda_max"] = float(columns["lmbda_max"].max())
    summary["criteria"] = {
        "reached_t_end": bool(np.isclose(columns["t_ms"][-1], t_end_ms)),
        "newton_converged_every_step": failure is None,
        "detF_positive_every_step": bool(np.all(columns["n_detF_nonpositive"] == 0)),
        **{f"{c}_five_phases_in_order": summary[c]["five_phases_in_order"] for c in CHAMBERS},
    }
    summary["land_placement"] = land_placement
    return summary


def plot(columns: dict[str, np.ndarray], summary: dict[str, Any], path: Path) -> None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig = plt.figure(layout="constrained", figsize=(12, 9))
    grid = fig.add_gridspec(3, 2)
    ax_loop = fig.add_subplot(grid[:2, 0])
    ax_p = fig.add_subplot(grid[0, 1])
    ax_v = fig.add_subplot(grid[1, 1], sharex=ax_p)
    ax_ta = fig.add_subplot(grid[2, 1], sharex=ax_p)
    ax_j = fig.add_subplot(grid[2, 0], sharex=ax_p)
    t = columns["t_ms"]
    titles = []
    for c, colour in (("LV", "crimson"), ("RV", "steelblue")):
        V, P = columns[f"V_{c}_mL"], columns[f"P_{c}_kPa"]
        ax_loop.plot(V, P, color=colour, label=c)
        ax_p.plot(t, P, color=colour, label=f"P {c}")
        ax_p.plot(t, columns[f"Pc_{c}_kPa"], color=colour, linestyle="--", label=f"P_c {c}")
        ax_v.plot(t, V, color=colour, label=c)
        ejection = summary[c]["ejection"]
        titles.append(
            f"{c} EF --"
            if ejection is None
            else f"{c} EF {ejection['EF_percent']:.1f}% (EDV {ejection['EDV_mL']:.1f}, "
            f"ESV {ejection['ESV_mL']:.1f} mL)",
        )
    ax_loop.set_xlabel("V [mL]")
    ax_loop.set_ylabel("P [kPa]")
    ax_loop.set_title(" | ".join(titles), fontsize=10)
    ax_loop.legend()
    ax_p.set_ylabel("P [kPa]")
    ax_p.legend(fontsize="x-small", ncol=2)
    ax_v.set_ylabel("V [mL]")
    ax_v.legend(fontsize="x-small")
    ax_ta.plot(t, columns["Ta_max_kPa"], color="0.2", label="max Ta")
    ax_ta.plot(t, columns["Ta_min_kPa"], color="0.6", label="min Ta")
    ax_ta.set_ylabel("Ta [kPa]")
    ax_ta.set_xlabel("t [ms]")
    ax_phase = ax_ta.twinx()
    for c, colour in (("LV", "crimson"), ("RV", "steelblue")):
        ax_phase.step(t, columns[f"phase_{c}"], where="pre", color=colour, alpha=0.5)
    ax_phase.set_yticks([p.value for p in Phase])
    ax_phase.set_yticklabels(["PRE", "IVC", "EJ", "IVR", "FILL"], fontsize="x-small")
    ax_ta.legend(fontsize="x-small", loc="upper left")
    ax_j.plot(t, columns["detF_min"], color="0.2")
    ax_j.set_ylabel("min det F")
    ax_j.set_xlabel("t [ms]")
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    start_total = time.perf_counter()
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    for lib in ("numba", "matplotlib", "scifem"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise RuntimeError("rodero_05 runs in serial only, as physcardems' run does")
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    num_steps = round(args.t_end / DT_MECH)

    # ---------------------------------------------------------
    # 1. The case, the generated code, and where parameters go
    # ---------------------------------------------------------
    modules = load_ode_modules(ODEFILE, HERE / "generated_odes" / ODEFILE.stem)
    case = load_case(args.case_dir, tref_scale=args.tref, comm=comm)
    placed = place_land(case.land, modules)
    land_placement = {half: sorted(values) for half, values in placed.items()}
    logger.info(f"Land parameters into the EP half: {placed['ep']}")
    logger.info(f"Land parameters into the mechanics half: {placed['mechanics']}")
    logger.info(f"EP scales: {case.ep_scales}")
    mechanics_states = check_mechanics_initial_states(case.initial_states, modules.mechanics)
    logger.info(
        f"Contraction states start at the .ode defaults {mechanics_states}, as in every "
        "cell type's steady state",
    )

    geometry = case.geometry
    mesh = geometry.mesh
    volumes = {c: geometry.volume(c) * 1e6 for c in CHAMBERS}
    logger.info(
        f"{mesh.topology.index_map(3).size_local} cells; reference cavity volumes "
        f"LV {volumes['LV']:.1f} mL, RV {volumes['RV']:.1f} mL",
    )

    # ---------------------------------------------------------
    # 2. Activation, mechanics, the cycle, EP, the controller
    # ---------------------------------------------------------
    backend = GeneratedActivation(
        modules.mechanics,
        mesh,
        case.f0,
        quadrature_degree=QUADRATURE_DEGREE,
        parameters=placed["mechanics"],
        tension_scale=case.myocardium_mask,
    )
    logger.info("Building the mechanics problem (form compilation takes a while)")
    problem = make_problem(case, backend)
    cycle = CycleController(
        problem,
        {"LV": case.lv, "RV": case.rv},
        preconditioner_lag=PRECONDITIONER_LAG,
    )
    ep_solver = make_ep_solver(case, modules.ep, placed["ep"])
    controller = SimulationController(Cycle(cycle), ep_solver, backend, modules, DT_MECH, DT_EP)

    timings = {"ep_ode_s": 0.0, "ep_pde_s": 0.0}
    ep_solver.ode.step = accumulate_time(ep_solver.ode.step, timings, "ep_ode_s")  # type: ignore[method-assign]
    ep_solver.pde.step = accumulate_time(ep_solver.pde.step, timings, "ep_pde_s")  # type: ignore[method-assign]
    solves = SolveLog(problem)

    # ---------------------------------------------------------
    # 3. Recording
    # ---------------------------------------------------------
    points = backend.space.element.interpolation_points
    myocardium_q = dolfinx.fem.Function(backend.space)
    myocardium_q.interpolate(dolfinx.fem.Expression(case.myocardium_mask, points))
    in_myocardium = myocardium_q.x.array > 0.5
    detF = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(problem.u)), points)
    cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    v_index = modules.ep.state_index("v")
    ode = ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)  # make_ep_solver's

    rows: list[dict[str, float]] = []
    attempts: list[list[dict[str, Any]]] = []
    switches: list[dict[str, Any]] = []

    def record(t: float, during: dict[str, Phase], first_attempt: int, wall: float) -> None:
        # The quadrature tension, masked and in kPa, before active_tension's P1 average.
        Ta = backend.tension_kPa.x.array[in_myocardium]
        lmbda = backend.outputs["lmbda"].x.array[in_myocardium]
        J = detF.eval(mesh, cells)
        v = ode.values[v_index]
        step_attempts = solves.attempts[first_attempt:]
        last = step_attempts[-1]
        row: dict[str, float] = {
            "t_ms": t,
            "Ta_max_kPa": float(Ta.max()),
            "Ta_min_kPa": float(Ta.min()),
            "lmbda_min": float(lmbda.min()),
            "lmbda_max": float(lmbda.max()),
        }
        for c in CHAMBERS:
            r = cycle.records[c]
            row |= {
                f"phase_{c}": int(during[c]),
                f"V_{c}_mL": r.V * 1e6,
                f"P_{c}_kPa": r.P * 1e-3,
                f"Pc_{c}_kPa": r.P_c * 1e-3,
                f"Q_{c}_mL_s": r.Q * 1e6,
            }
        row |= {
            "detF_min": float(J.min()),
            "n_detF_nonpositive": int(np.count_nonzero(J <= 0.0)),
            "newton_iterations": last["iterations"],
            "snes_reason": last["reason"],
            "linear_iterations": last["linear_iterations"],
            "solve_attempts": len(step_attempts),
            "v_min_mV": float(v.min()),
            "v_max_mV": float(v.max()),
            "wall_s": wall,
        }
        rows.append(row)
        attempts.append(step_attempts)
        for c in CHAMBERS:
            if cycle.records[c].phase != during[c]:
                switches.append(
                    {
                        "chamber": c,
                        "from": during[c].name,
                        "to": cycle.records[c].phase.name,
                        "t_ms": t,
                    },
                )

    def write_outputs(failure: str | None, final: bool) -> None:
        with open(outdir / "log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if not final and len(rows) % WRITE_EVERY != 1:
            return
        columns = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        summary = summarise(
            columns,
            switches,
            attempts,
            failure,
            land_placement,
            t_end_ms=num_steps * DT_MECH,
        )
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        (outdir / "timings.json").write_text(json.dumps(timings, indent=2))
        plot(columns, summary, outdir / "pv_loops.png")
        if final:
            logger.info(f"Summary: {json.dumps(summary, indent=2)}")
            logger.info(f"Timings: {timings}")

    # ---------------------------------------------------------
    # 4. The unloaded solve
    # ---------------------------------------------------------
    # EP's resting cross-bridge states into the backend, which is then solved with
    # every cavity at zero pressure (a new CavityControl is in pressure mode at 0) and
    # its own step the identity (dt = 0), and accepted: the tissue starts at rest at
    # this stretch, which also goes to EP.
    start = time.perf_counter()
    controller.plan.forward(0.0)
    if not problem.solve():
        raise RuntimeError("The unloaded solve did not converge")
    backend.post_solve()
    controller.plan.backward()
    cycle.initialize(0.0)
    record(0.0, {c: cycle.cycles[c].phase for c in CHAMBERS}, 0, time.perf_counter() - start)
    timings["setup_s"] = time.perf_counter() - start_total
    logger.info(
        f"Unloaded solve: {rows[0]['newton_iterations']} iterations; LV "
        f"{rows[0]['V_LV_mL']:.1f} mL, RV {rows[0]['V_RV_mL']:.1f} mL; resting Ta "
        f"{rows[0]['Ta_min_kPa']:.2f} to {rows[0]['Ta_max_kPa']:.2f} kPa; lambda "
        f"{rows[0]['lmbda_min']:.3f} to {rows[0]['lmbda_max']:.3f}; set-up "
        f"{timings['setup_s']:.0f} s",
    )

    # ---------------------------------------------------------
    # 5. Run
    # ---------------------------------------------------------
    failure: str | None = None
    start_loop = time.perf_counter()
    try:
        for _ in range(num_steps):
            during = {c: cycle.cycles[c].phase for c in CHAMBERS}
            first_attempt = len(solves.attempts)
            start = time.perf_counter()
            controller.step()
            record(controller.t, during, first_attempt, time.perf_counter() - start)
            row = rows[-1]
            logger.info(
                f"t={row['t_ms']:6.1f} ms | Ta [{row['Ta_min_kPa']:6.2f}, "
                f"{row['Ta_max_kPa']:6.2f}] kPa, lambda [{row['lmbda_min']:.3f}, "
                f"{row['lmbda_max']:.3f}] | "
                + " | ".join(
                    f"{c} {during[c].name[:8]:8s} V={row[f'V_{c}_mL']:6.1f} "
                    f"P={row[f'P_{c}_kPa']:6.2f}"
                    for c in CHAMBERS
                )
                + f" | its={row['newton_iterations']} reason={row['snes_reason']} "
                f"solves={row['solve_attempts']} | detF min {row['detF_min']:.3f} | "
                f"{row['wall_s']:.1f} s",
            )
            timings["mech_s"] = solves.seconds
            timings["loop_s"] = time.perf_counter() - start_loop
            write_outputs(None, final=False)
    except BaseException as error:
        # Any exception, KeyboardInterrupt included: the summary must not report a run
        # that stopped early as converged. The solves of the step it stopped in, which
        # no row records.
        unrecorded = solves.attempts[sum(len(a) for a in attempts) :]
        t_failed = controller.t_failed if controller.t_failed is not None else controller.t
        failure = f"{type(error).__name__} at t = {t_failed} ms: {error}; solves {unrecorded}"
        logger.exception("The run stopped before t_end")
        raise
    finally:
        timings["mech_s"] = solves.seconds
        timings["loop_s"] = time.perf_counter() - start_loop
        timings["total_s"] = time.perf_counter() - start_total
        write_outputs(failure, final=True)


if __name__ == "__main__":
    main()
