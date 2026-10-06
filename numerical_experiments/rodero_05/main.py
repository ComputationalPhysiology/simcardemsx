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
- ``log.csv``'s ``phase_*`` columns are the phase each step was solved under;
  physcardems' are the phase after the step's switch, i.e. for the next step, which
  are the ``next_phase_*`` columns here.
- det F is sampled at the degree-4 quadrature points, and ``n_detF_nonpositive``
  counts the points where it is not positive; physcardems samples each cell's
  vertices and centroid, and counts cells.
- ``log.csv`` has no ``J_max`` or ``wall_ep_s`` column, both of which physcardems'
  has. ``post.py`` writes physcardems' ``active_stats.csv`` without its
  ``frac_h_zero`` column, and the pseudo-ECG from the saved ``v`` (see ``post.py``),
  not during the run.
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
- A restart continues ``log.csv`` and ``results.bp`` from the checkpoint; physcardems'
  starts its logs empty.
- Left out: the mid-ventricular slice statistics, and ``sf_IKs`` (read by physcardems,
  never used).

Output, in ``--output-dir``, in the upstream CLIs' layout (:mod:`simcardemsx.results`):

- ``config.resolved.toml``: :func:`settings`, with where the Land parameters went,
  rewritten by every run, fresh or restarted.
- ``results.bp``: EP's ``v`` and ``cai`` (P1) every ``--save-every-ep`` ms, and ``u``,
  ``lmbda``, ``tension_kPa`` and ``stiffness_kPa`` (the last three on the backend's
  quadrature points; the tension masked) every ``--save-every`` ms, all from t = 0.
- ``log.csv`` (:data:`LOG_FIELDS`): one row after the unloaded solve (``t_ms`` 0) and
  one per mechanics step, appended as the run goes: the maximum and minimum ``Ta`` and
  the range of lambda over the myocardium's quadrature points; per cavity the phase the
  step was solved under and the phase after it (``phase_*``, ``next_phase_*``: 0
  PRELOAD, 1 IVC, 2 EJECTION, 3 IVR, 4 FILLING), V, P, the Windkessel's compliance
  pressure P_c and the outflow Q; the minimum of det F over every quadrature point, and
  how many are not positive; the last solve's Newton iterations, SNES converged reason
  and linear iterations, how many solves the step took (2 = one retry) and the first
  solve's iterations and reason; EP's membrane potential range; the step's wall time.
  ``pulse.cycle`` retries a solve at most once, so the first and last solves are every
  solve, and the summary is computed from ``log.csv`` alone.
- ``restart.bp`` and ``restart.json``: a checkpoint every ``--checkpoint-every`` ms,
  after every step at which a cavity's phase switched (as physcardems does), and at the
  end of the run.
- ``pv_loops.png``: the PV loops with EF, where EDV is the volume at the last switch
  into IVC and ESV the minimum since; pressures, volumes, ``Ta`` and phases in time.
- ``summary.json``: EDV, ESV, EF and peak pressure per ventricle; the phase switches;
  the Newton, SNES-reason and retry counts; min det F and min ``Ta``; the Stage B
  acceptance criteria; where the Land parameters went. Among the criteria,
  ``reached_t_end`` says whether the last row is at ``--t-end``, in whole steps (the
  summary is also rewritten while the run goes on), and
  ``newton_converged_every_step`` is false if the run stopped on any exception,
  ``KeyboardInterrupt`` included, which ``failure`` then names. A step whose retry
  converged counts as converged; ``newton`` reports the retries. After a restart it
  covers the whole run, from ``log.csv``.
- ``timings.json``: this process's wall time in the EP ODE and PDE steps and the
  mechanics solves, the set-up (code generation, loading, form compilation, the
  unloaded solve), the loop, and the whole run (after a restart, only the part it ran).
- ``run.json``: the provenance of this process (git commit, versions, ranks, time),
  ``history`` (that of every process that wrote the run), ``restart``, ``status``
  (``running`` from the start of the loop, then ``finished`` or ``failed``),
  ``failure``, ``t_fail_ms`` and ``reached_t_end``. Written when the loop starts, and
  again last of all at the end, failure included.

``post.py`` writes the summary and the plot again from these, by the same functions,
with VTX fields, ``Ta`` and lambda statistics per cell type and the pseudo-ECG, into
``post/``.

The folder rules are the CLIs'. A run refuses a folder that holds any of these files,
unless ``--overwrite`` (which deletes only them) or ``--restart`` is given. The default
folder may hold the results of a run from before these rules, so a plain re-run into it
is refused too. ``--restart`` continues from the checkpoint, and refuses one written
with other physics (:func:`physics`): it may change ``--t-end``, the output options and
the solver options, nothing else. If the checkpoint is at or past ``--t-end`` it takes
no step. A restart skips the unloaded solve and the start of the cycle, which the
checkpoint replaces. The lagged LU is not part of a checkpoint: a restarted run
factorizes afresh at its first solve, so it differs from the uninterrupted run within
the solver tolerances, not bit for bit.

Serial only. The 800 ms run takes hours; ``--t-end 20`` is the smoke test (physcardems'
``em_tref7_short.toml``).
"""

import argparse
import dataclasses
import hashlib
import inspect
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
from simcardemsx.checkpoint import Checkpointer, write_json
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Cycle
from simcardemsx.ode_model import ODEModules, load_ode_modules
from simcardemsx.provenance import provenance
from simcardemsx.results import ARTIFACTS, CsvLog, ResultsWriter, write_resolved_settings

HERE = Path(__file__).resolve().parent
# case.py and post.py sit next to this script, and demo_io beside its directory, not in
# the installed package. As rodero_05.case, with numerical_experiments/ on the path, it
# has the name mypy gives it too.
sys.path.insert(0, str(HERE.parent))
import demo_io  # noqa: E402

from rodero_05 import case as case_data  # noqa: E402
from rodero_05.case import (  # noqa: E402
    CELLTYPES,
    PCL_MS,
    QUADRATURE_DEGREE,
    TAGS,
    Case,
    files_sha256,
    load_case,
)
from rodero_05.post import CHAMBERS, plot, summarise  # noqa: E402

logger = logging.getLogger(__name__)

ODEFILE = HERE.parent / "odefiles" / "ToRORd_dynCl_endo_zetasplit.ode"
DEFAULT_CASE_DIR = HERE.parent.parent / "third-party" / "physcardems" / "cases" / "rodero_05"

#: Time steps and end time, in ms (run script, lines 44-46).
DT_MECH = 2.0
DT_EP = 0.05
T_END = 800.0

#: The spaces of ``results.bp``, which ``post.py`` rebuilds: the EP ODE space, P1 (the
#: transfer plan averages what crosses back onto P1 or DG0 only), holds ``v`` and
#: ``cai``; pulse's displacement space, given to the problem, holds ``u``; and the
#: backend's scalar quadrature space at ``QUADRATURE_DEGREE`` holds the rest.
EP_ODE_ELEMENT = ("Lagrange", 1)
U_SPACE = "P_2"
#: The EP half's parameters every node shares; ``celltype`` and ``i_Stim_Start`` are
#: each node's own (run script, lines 236-241).
EP_ODE_PARAMETERS = {"i_Stim_Period": PCL_MS, "lmbda": 1.0, "dLambda": 0.0}

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
#: the run (``log.csv`` is appended after every step).
WRITE_EVERY = 25

EP_RESULTS = ("v", "cai")
MECHANICS_RESULTS = ("u", *demo_io.ACTIVATION_RESULTS)


def _cavity_fields(c: str) -> tuple[str, ...]:
    return (
        f"phase_{c}",
        f"next_phase_{c}",
        f"V_{c}_mL",
        f"P_{c}_kPa",
        f"Pc_{c}_kPa",
        f"Q_{c}_mL_s",
    )


#: ``log.csv``'s columns (see the module docstring).
LOG_FIELDS = (
    "t_ms",
    "Ta_max_kPa",
    "Ta_min_kPa",
    "lmbda_min",
    "lmbda_max",
    *(name for c in CHAMBERS for name in _cavity_fields(c)),
    "detF_min",
    "n_detF_nonpositive",
    "newton_iterations",
    "snes_reason",
    "linear_iterations",
    "solve_attempts",
    "first_iterations",
    "first_reason",
    "v_min_mV",
    "v_max_mV",
    "wall_s",
)
#: Everything a run writes into its output folder, and so what --overwrite deletes.
OUTPUT_ARTIFACTS = (*ARTIFACTS, "summary.json", "pv_loops.png")
#: The arguments that do not define the physics: the run's length and its output.
NOT_PHYSICS = (
    "t_end",
    "output_dir",
    "save_every",
    "save_every_ep",
    "checkpoint_every",
    "restart",
    "overwrite",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with ``save_every`` resolved from its default."""
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
        default=case_data.TREF_SCALE,
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
        help="Output directory (default: %(default)s). It must hold no results of an "
        "earlier run, unless --overwrite or --restart is given.",
    )
    parser.add_argument(
        "--save-every",
        type=float,
        default=None,
        help="Save u, lmbda, tension_kPa and stiffness_kPa to results.bp every this many "
        f"ms, a whole multiple of the {DT_MECH} ms mechanics step (default: {DT_MECH}).",
    )
    parser.add_argument(
        "--save-every-ep",
        type=float,
        default=1.0,
        help=f"Save EP's v and cai to results.bp every this many ms, a whole multiple of "
        f"the {DT_EP} ms EP step; post.py's pseudo-ECG has one row per save "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=50.0,
        help="Write a checkpoint (restart.bp, restart.json) every this many ms, a whole "
        f"multiple of the {DT_MECH} ms mechanics step, after every phase switch and at the "
        "end of the run (default: %(default)s).",
    )
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument(
        "--restart",
        action="store_true",
        help="Continue the run in the output directory from its checkpoint. Refused if "
        "the physics differ from the checkpoint's; --t-end and the output options may.",
    )
    flags.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the results in the output directory. Only this example's own files "
        "are deleted.",
    )
    args = parser.parse_args(argv)
    if args.save_every is None:
        args.save_every = DT_MECH
    return args


# ---------------------------------------------------------------------------
# The model, as the run reads it
# ---------------------------------------------------------------------------


def material_parameters() -> dict[str, dict[str, pulse.Variable]]:
    """The mechanics' material as :func:`material` and :func:`make_problem` give it to
    pulse, as ``pulse.Variable``s, before the valve plugs' stiffness scale multiplies
    the moduli: :data:`MATERIAL_PARAMS` for Holzapfel-Ogden, ``Compressible2``'s
    ``kappa`` (:data:`KAPPA_PA`), and the viscosity's ``eta``, pulse's default."""
    return {
        "HolzapfelOgden": {
            name: pulse.Variable(value, "kPa" if name in MODULI else "dimensionless")
            for name, value in MATERIAL_PARAMS.items()
        },
        "Compressible2": {"kappa": pulse.Variable(KAPPA_PA, "Pa")},
        "Viscous": {"eta": pulse.viscoelasticity.Viscous().eta},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_settings(tref: float) -> dict[str, Any]:
    """The model as the run reads it, as plain values: the time steps; what ``case.py``
    copies from physcardems (where the files are, the cell types, tags, valve-plug
    stiffness, reference scaling, pacing period, Land parameters and their calibration,
    and EP's scales); the Land parameters at ``tref``; EP (its ODE space and shared
    parameters, theta, conductivities and C_m); the mechanics (spaces, quadrature
    degree, material in SI base units, boundary conditions, cavities, density and the
    generalized-alpha parameters); the activation backend; and each cavity's cycle as
    passed to ``pulse.cycle``, in SI.

    beat's theta and conductivities, pulse's viscosity, density and generalized-alpha
    parameters, and the backend's scheme and formulation are read off the libraries,
    whose defaults the run uses, so a change to those defaults changes these too.
    """
    material = material_parameters()
    dynamic = pulse.DynamicProblem.default_parameters()
    backend = inspect.signature(GeneratedActivation).parameters
    conductivities = beat.conductivities.default_conductivities("Niederer")
    return {
        "dt_mech": DT_MECH,
        "dt_ep": DT_EP,
        "case": {
            "geometry_dir": case_data.GEOMETRY_DIR,
            "steady_state_glob": case_data.STEADY_STATE_GLOB,
            "celltypes": {str(c): name for c, name in CELLTYPES.items()},
            "tags": dict(TAGS),
            "myocardium_tags": list(case_data.MYOCARDIUM_TAGS),
            "valve_stiffness_scale": case_data.VALVE_STIFFNESS_SCALE,
            "reference_scale": case_data.REFERENCE_SCALE,
            "pcl_ms": PCL_MS,
            "land_base": dict(case_data.LAND_BASE),
            "land_overrides": dict(case_data.LAND_OVERRIDES),
            "land_scales": dict(case_data.LAND_SCALES),
            "mechanics_only_keys": list(case_data.MECHANICS_ONLY_KEYS),
            "ep_scales": dict(case_data.EP_SCALES),
        },
        "land": case_data.land_parameters(tref),
        "ep": {
            "ode_element": list(EP_ODE_ELEMENT),
            "ode_parameters": dict(EP_ODE_PARAMETERS),
            "per_node": "steady state, celltype and i_Stim_Start (the LAT) of the case",
            "pde_stimulus": "none",
            "theta": inspect.signature(beat.MonodomainSplittingSolver).parameters["theta"].default,
            "conductivities": {
                "name": "Niederer",
                "units": "SI base units (S/m, 1/m)",
                **{
                    name: float(value.to_base_units().magnitude)
                    for name, value in conductivities.items()
                },
            },
            "C_m_uF_per_cm2": C_M_UF_PER_CM2,
        },
        "mechanics": {
            "quadrature_degree": QUADRATURE_DEGREE,
            "u_space": U_SPACE,
            "material": {
                "units": "SI base units (Pa, Pa s)",
                **{
                    model: {name: float(value.to_base_units()) for name, value in values.items()}
                    for model, values in material.items()
                },
                "moduli_scaled_by_stiffness": list(MODULI),
                "use_heaviside": not FIBRE_COMPRESSION_RESISTANCE,
                "use_subplus": not FIBRE_COMPRESSION_RESISTANCE,
            },
            "robin": {"EPI": {"spring_Pa_per_m": EPI_SPRING, "damper_Pa_s_per_m": EPI_DAMPING}},
            "dirichlet": "none",
            "cavities": {c: "controlled" for c in CHAMBERS},
            "rho_kg_per_m3": float(dynamic["rho"].to_base_units()),
            "alpha_m": dynamic["alpha_m"],
            "alpha_f": dynamic["alpha_f"],
        },
        "activation": {
            "backend": "GeneratedActivation",
            "scheme": backend["scheme"].default,
            "formulation": backend["formulation"].default,
            "tension_scale": "the myocardium mask (DG0): 0 in the valve plugs",
            "initial_states": "the .ode's defaults",
        },
        "cycle": {
            name: dataclasses.asdict(params)
            for name, params in case_data.cycle_parameters().items()
        },
    }


def solver_settings() -> dict[str, Any]:
    """physcardems' solver options, as the run gives them to pulse on top of its
    defaults, and the cycle's preconditioner lag. Not physics. PETSc's flags (``None``)
    are written as ``""``: TOML has no null."""
    return {
        "petsc_options": {k: "" if v is None else v for k, v in PETSC_OPTIONS.items()},
        "preconditioner_lag": PRECONDITIONER_LAG,
    }


def settings(
    args: argparse.Namespace,
    land_placement: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """What ``config.resolved.toml`` holds: every argument, with ``case_dir`` as its path
    (as given), its resolved path and the sha256 of the files the run reads from it
    (``case.files_sha256``), ``odefile`` (the ``.ode`` file's path, file name and
    sha256), :func:`model_settings`, ``solver`` (:func:`solver_settings`), and, if
    given, ``land_placement``: which Land parameters went to which half of the split
    (derived from the ``.ode`` file and the Land parameters, so not physics of its own).

    TOML has no null, so an argument that is ``None`` is not in the file. Raises
    ``FileNotFoundError`` if the ``.ode`` file or a file of the case is missing.
    """
    arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    case_dir = Path(args.case_dir)
    arguments["case_dir"] = {
        "path": str(case_dir),
        "resolved": str(case_dir.resolve()),
        "files_sha256": files_sha256(case_dir),
    }
    arguments["odefile"] = {"path": str(ODEFILE), "name": ODEFILE.name, "sha256": _sha256(ODEFILE)}
    result = {**arguments, **model_settings(args.tref), "solver": solver_settings()}
    if land_placement is not None:
        result["land_placement"] = land_placement
    return result


def physics(args: argparse.Namespace) -> dict[str, Any]:
    """What a restart must share with the checkpointed run: :func:`settings` without the
    run length and the output options (:data:`NOT_PHYSICS`), the solver options, the
    ``.ode`` file's path and the case directory's path as given. The case directory is
    in it by its resolved path and the sha256 of its files' contents (cheap: about 10
    MB), the ``.ode`` file by its name and sha256.

    Raises ``FileNotFoundError`` if the ``.ode`` file or a file of the case is missing.
    """
    result = settings(args)
    for key in (*NOT_PHYSICS, "solver"):
        del result[key]
    del result["odefile"]["path"]
    del result["case_dir"]["path"]
    return result


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
    W = dolfinx.fem.functionspace(mesh, EP_ODE_ELEMENT)
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
        ep_module.init_parameter_values(**EP_ODE_PARAMETERS)[:, None],
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
    """Holzapfel-Ogden with :func:`material_parameters`, each modulus a DG0 field: the
    modulus times the case's stiffness scale (x 3 in the valve plugs).

    physcardems multiplies the whole passive stress by the scale (its ``ScaledModel``);
    Holzapfel-Ogden's stress is linear in its four moduli, so this is the same stress,
    through pulse's own spatial material parameters.
    """
    scale = case.stiffness_scale
    parameters = material_parameters()["HolzapfelOgden"]

    def parameter(name: str) -> pulse.Variable:
        if name not in MODULI:
            return parameters[name]
        field = dolfinx.fem.Function(scale.function_space, name=name)
        field.x.array[:] = parameters[name].value * scale.x.array
        return pulse.Variable(field, "kPa")

    return pulse.HolzapfelOgden(
        f0=case.f0,
        s0=case.s0,
        **{name: parameter(name) for name in parameters},
        use_heaviside=not FIBRE_COMPRESSION_RESISTANCE,
        use_subplus=not FIBRE_COMPRESSION_RESISTANCE,
    )


def make_problem(case: Case, backend: GeneratedActivation) -> pulse.DynamicProblem:
    """The run script's mechanics (lines 264-303), its two cavities controlled."""
    geometry = case.geometry
    mesh = geometry.mesh
    parameters = material_parameters()
    model = pulse.CardiacModel(
        material=material(case),
        active=backend,
        compressibility=pulse.compressibility.Compressible2(**parameters["Compressible2"]),
        viscoelasticity=pulse.viscoelasticity.Viscous(**parameters["Viscous"]),
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
        parameters={
            "dt": pulse.Variable(DT_MECH * 1e-3, "s"),
            "u_space": U_SPACE,
            "petsc_options": petsc_options,
        },
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
        raise SystemExit("rodero_05 runs in serial only, as physcardems' run does")
    outdir: Path = args.output_dir

    # Everything that can refuse the run does so here: before any file is touched, and
    # before anything expensive is generated, loaded, built or compiled.
    strides, run_physics = demo_io.prepare_run(
        outdir,
        restart=args.restart,
        overwrite=args.overwrite,
        strides={
            "--save-every-ep": (args.save_every_ep, DT_EP),
            "--save-every": (args.save_every, DT_MECH),
            "--checkpoint-every": (args.checkpoint_every, DT_MECH),
        },
        physics=lambda: physics(args),
        artifacts=OUTPUT_ARTIFACTS,
    )
    save_ep_every = strides["--save-every-ep"]
    save_every = strides["--save-every"]
    checkpoint_every = strides["--checkpoint-every"]
    t_end_ms = round(args.t_end / DT_MECH) * DT_MECH

    # ---------------------------------------------------------
    # 1. The case, the generated code, and where parameters go
    # ---------------------------------------------------------
    modules = load_ode_modules(ODEFILE, HERE / "generated_odes" / ODEFILE.stem)
    case = load_case(args.case_dir, tref_scale=args.tref, comm=comm)
    placed = place_land(case.land, modules)
    land_placement = {half: sorted(values) for half, values in placed.items()}
    # Every run's, a restart's too: the latest run's settings win, as in the CLIs.
    if comm.rank == 0:
        write_resolved_settings(outdir / "config.resolved.toml", settings(args, land_placement))
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
    # Cycle(cycle) itself, not wrapped: the controller then checkpoints the cycle too.
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

    checkpointer = Checkpointer(controller, outdir, physics=run_physics)
    results = ResultsWriter(outdir)
    log = CsvLog(outdir / "log.csv", LOG_FIELDS)
    #: log.csv's rows, from t = 0: the summary and the plot are drawn from them.
    rows: list[dict[str, float]] = []

    def record(t: float, during: dict[str, Phase], first_attempt: int, wall: float) -> None:
        """Append ``log.csv``'s row for the step ending at ``t``, solved under the
        phases ``during``, whose solves are those from ``first_attempt`` on."""
        # The quadrature tension, masked and in kPa, before active_tension's P1 average.
        Ta = backend.tension_kPa.x.array[in_myocardium]
        lmbda = backend.outputs["lmbda"].x.array[in_myocardium]
        J = detF.eval(mesh, cells)
        v = ode.values[v_index]
        step_attempts = solves.attempts[first_attempt:]
        first, last = step_attempts[0], step_attempts[-1]
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
                f"next_phase_{c}": int(r.phase),
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
            "first_iterations": first["iterations"],
            "first_reason": first["reason"],
            "v_min_mV": float(v.min()),
            "v_max_mV": float(v.max()),
            "wall_s": wall,
        }
        log.append(row)
        rows.append(row)

    # results.bp's fields. EP's are P1 copies of rows of the ODE's state array, refreshed
    # before each save; the others are the problem's and the backend's own Functions.
    ep_space = dolfinx.fem.functionspace(mesh, EP_ODE_ELEMENT)
    ep_fields = {name: dolfinx.fem.Function(ep_space, name=name) for name in EP_RESULTS}
    ep_rows = {name: modules.ep.state_index(name) for name in EP_RESULTS}
    mechanics_fields = {
        "u": problem.u,
        "lmbda": backend.outputs["lmbda"],
        "tension_kPa": backend.tension_kPa,
        "stiffness_kPa": backend.stiffness_kPa,
    }

    def save_ep(t: float) -> None:
        for name, f in ep_fields.items():
            f.x.array[:] = ode.values[ep_rows[name]]
        results.write(t, ep_fields)

    # ---------------------------------------------------------
    # 4. The unloaded solve, or the checkpoint
    # ---------------------------------------------------------
    # EP's resting cross-bridge states into the backend, which is then solved with
    # every cavity at zero pressure (a new CavityControl is in pressure mode at 0) and
    # its own step the identity (dt = 0), and accepted: the tissue starts at rest at
    # this stretch, which also goes to EP. A restart takes all of it, and the cycle's
    # state, from the checkpoint instead.
    start = time.perf_counter()
    if not args.restart:
        controller.plan.forward(0.0)
        if not problem.solve():
            raise RuntimeError("The unloaded solve did not converge")
        backend.post_solve()
        controller.plan.backward()
        cycle.initialize(0.0)
    unloaded = {c: cycle.cycles[c].phase for c in CHAMBERS}

    def write_initial() -> None:
        save_ep(controller.t)
        results.write(controller.t, mechanics_fields)
        record(controller.t, unloaded, 0, time.perf_counter() - start)

    # A refused or failed restore leaves the folder as it was, apart from
    # config.resolved.toml: nothing below it, neither the end checkpoint nor run.json,
    # is reached. A restart gets the rows up to its checkpoint from log.csv.
    rows.extend(
        demo_io.start_or_resume(
            checkpointer,
            log,
            results,
            restart=args.restart,
            result_names=[*EP_RESULTS, *MECHANICS_RESULTS],
            write_initial=write_initial,
        ),
    )
    timings["setup_s"] = time.perf_counter() - start_total
    if not args.restart:
        logger.info(
            f"Unloaded solve: {rows[0]['newton_iterations']} iterations; LV "
            f"{rows[0]['V_LV_mL']:.1f} mL, RV {rows[0]['V_RV_mL']:.1f} mL; resting Ta "
            f"{rows[0]['Ta_min_kPa']:.2f} to {rows[0]['Ta_max_kPa']:.2f} kPa; lambda "
            f"{rows[0]['lmbda_min']:.3f} to {rows[0]['lmbda_max']:.3f}; set-up "
            f"{timings['setup_s']:.0f} s",
        )

    # ---------------------------------------------------------
    # 5. Output: summary, plot, timings, run.json
    # ---------------------------------------------------------
    failure: str | None = None
    t_fail: float | None = None

    def write_summary(final: bool = False) -> None:
        columns = {key: np.array([float(row[key]) for row in rows]) for key in LOG_FIELDS}
        summary = summarise(columns, failure, land_placement, t_end_ms)
        if comm.rank == 0:
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
            plot(columns, summary, outdir / "pv_loops.png")
        if final:
            logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    def write_timings() -> None:
        if comm.rank == 0:
            (outdir / "timings.json").write_text(json.dumps(timings, indent=2))

    def write_run(extra: dict[str, Any], status: str) -> None:
        """``run.json``: the provenance of this process, then the run's own keys."""
        reached = failure is None and abs(controller.t - t_end_ms) <= 1e-9 * max(1.0, t_end_ms)
        write_json(
            outdir / "run.json",
            {
                **extra["provenance"],
                "history": extra["history"],
                "restart": extra["restart"],
                "status": status,
                "failure": failure,
                "t_fail_ms": t_fail,
                "reached_t_end": reached,
            },
            comm,
        )

    write_run(
        {"provenance": provenance(HERE), "history": checkpointer.history, "restart": args.restart},
        "running",
    )

    # ---------------------------------------------------------
    # 6. Run
    # ---------------------------------------------------------
    # The step under way: the phases it is solved under, its first solve, its start.
    step: dict[str, Any] = {}

    # The controller counts steps from 1. A step that fails is rolled back, but the EP
    # saves of its micro-steps stay in results.bp: a restart from the step's start
    # recomputes them, so it does not save them again.
    def on_ep_step(t: float, ep_step_idx: int) -> None:
        if ep_step_idx % save_ep_every == 0:
            save_ep(t)

    def on_mech_step(t: float, mech_step_idx: int, newton_iterations: int) -> None:
        during = step["during"]
        record(t, during, step["first_attempt"], time.perf_counter() - step["start"])
        if mech_step_idx % save_every == 0:
            results.write(t, mechanics_fields)
        row = rows[-1]
        logger.info(
            f"t={row['t_ms']:6.1f} ms | Ta [{row['Ta_min_kPa']:6.2f}, "
            f"{row['Ta_max_kPa']:6.2f}] kPa, lambda [{row['lmbda_min']:.3f}, "
            f"{row['lmbda_max']:.3f}] | "
            + " | ".join(
                f"{c} {during[c].name[:8]:8s} V={row[f'V_{c}_mL']:6.1f} P={row[f'P_{c}_kPa']:6.2f}"
                for c in CHAMBERS
            )
            + f" | its={row['newton_iterations']} reason={row['snes_reason']} "
            f"solves={row['solve_attempts']} | detF min {row['detF_min']:.3f} | "
            f"{row['wall_s']:.1f} s",
        )

    # The run is round(t_end / dt) steps from t = 0, so a restart at or past it takes none.
    remaining_steps = demo_io.steps_to_take(controller, args.t_end)
    start_loop = time.perf_counter()
    try:
        for _ in range(remaining_steps):
            step.update(
                during={c: cycle.cycles[c].phase for c in CHAMBERS},
                first_attempt=len(solves.attempts),
                start=time.perf_counter(),
            )
            controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)
            # As physcardems does: periodically, and after every phase switch.
            switched = any(cycle.records[c].phase != step["during"][c] for c in CHAMBERS)
            if controller.mech_step_idx % checkpoint_every == 0 or switched:
                checkpointer.write()
            timings["mech_s"] = solves.seconds
            timings["loop_s"] = time.perf_counter() - start_loop
            if controller.mech_step_idx % WRITE_EVERY == 0:
                write_summary()
                write_timings()
    except BaseException as error:
        # Any exception, KeyboardInterrupt included: the summary must not report a run
        # that stopped early as converged. A step that raised before its mech_callback
        # was rolled back (t_failed is set), and no row records its solves, so the
        # failure lists them.
        failure, t_fail = demo_io.failure_at(controller, error)
        if controller.t_failed is not None and step:
            failure += f"; solves of the failed step: {solves.attempts[step['first_attempt'] :]}"
        raise
    finally:
        timings["mech_s"] = solves.seconds
        timings["loop_s"] = time.perf_counter() - start_loop
        timings["total_s"] = time.perf_counter() - start_total
        logger.info(f"Timings: {timings}")
        # The end checkpoint and this example's own files first, each guarded, and
        # run.json last.
        demo_io.finish(
            None,
            checkpointer,
            [
                ("summary.json and pv_loops.png", lambda: write_summary(final=True)),
                ("timings.json", write_timings),
            ],
            failure=failure,
            t_fail_ms=t_fail,
            timings=timings,
            here=HERE,
            restart=args.restart,
            write_run=lambda extra: write_run(extra, "finished" if failure is None else "failed"),
        )


if __name__ == "__main__":
    main()
