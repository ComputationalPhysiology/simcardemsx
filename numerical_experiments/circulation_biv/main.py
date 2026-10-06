"""Stage A: EP-driven contraction of the UKB biventricular mesh in a closed-loop circulation.

This is pulse's ``demo/time_dependent/monolithic_3d0d_biv.py`` with the prescribed
Bestel tension replaced by simcardemsx's activation: beat (EP, monodomain) drives the
``mechanics`` component of the zeta split of ToR-ORd-Land, stepped inside the
mechanics Newton iteration by :class:`~simcardemsx.backends.GeneratedActivation`,
through :class:`~simcardemsx.controller.SimulationController`. The mechanics is a
``pulse.DynamicProblem`` whose two cavities are closed by Regazzoni's circuit
(``pulse.circulation.GotranxCirculation``), with the circuit's states in the same
Newton system as the displacement and both cavity pressures;
:class:`~simcardemsx.mechanics.CirculationClock` sets the circuit's clock on the
controller's.

The set-up follows the demo:

1. the mean UKB shape at end diastole, clipped at the valve plane, in metres, with
   LDRB fibres on ``Quadrature_6``;
2. Regazzoni's circuit, run alone from the mesh's end-diastolic volumes to a limit
   cycle, gives the end-diastolic pressures;
3. the mesh is prestressed to those pressures, to recover the unloaded reference;
4. a volume-controlled inflation takes it back to the end-diastolic volumes;
5. the coupled problem starts from there, with the circuit state of step 2.

EP runs on the same mesh, on P1, with ToR-ORd's own cellular stimulus (every point
fires at ``t = 0``, period 1000 ms, the circuit's beat). Mechanics steps are 2 ms, EP
steps 0.05 ms. Land's ``Tref`` is the ``.ode``'s 120 kPa times ``--tref``, by default
3, physcardems' eLife calibration. At the ``.ode``'s own ``Tref`` (``--tref 1``) the
LV's isovolumic peak stays about 2 mmHg below aortic pressure and it does not eject
in the first beat; the RV does.

Output goes to ``--outdir``, in the upstream CLIs' layout (:mod:`simcardemsx.results`):

- ``config.resolved.toml``: :func:`settings`, rewritten by every run, fresh or restarted.
- ``results.bp``: EP's ``v`` and ``cai`` (P1) every ``--save-every-ep`` ms, and ``u``,
  ``lmbda``, ``tension_kPa`` and ``stiffness_kPa`` (the last three on the backend's
  quadrature points) every ``--save-every`` ms, all from t = 0.
- ``log.csv``: one row at ``t = 0`` and one after each mechanics step, appended as the
  run goes. ``V_*_mL`` are the 3D cavity volumes ``V(u)`` and ``p_*_mmHg`` the cavity
  pressures; the ``circuit_*`` columns are every circuit state in the circuit's own
  units (``V_*`` mL, ``p_*`` mmHg, ``Q_*`` mL/s); ``total_volume_mL`` is the total
  blood volume with the 3D cavity volumes in place of the circuit's ``V_LV``/``V_RV``
  and ``conservation_drift`` its relative change since ``t = 0``; ``Ta_mean_kPa`` is
  the volume mean of the backend's ``active_tension``; ``newton_iterations`` is 0 at
  ``t = 0``, where nothing is solved.
- ``restart.bp`` and ``restart.json``: a checkpoint every ``--checkpoint-every`` ms and
  at the end of the run, with the scheme comparison's ``restart_recorder_<t>.npz``.
- ``pv_loops.png``: pressure-volume loops, and pressures, volumes and tension in time.
- ``summary.json``: per ventricle, ``EDV_mL`` and ``ESV_mL``, the largest and
  smallest ``V`` over the run, with ``V_range_mL``, their difference, and
  ``V_range_fraction``, that over ``EDV_mL``. These are not a stroke volume and an
  ejection fraction: the run is not a periodic beat, and ``V`` also changes while the
  outflow valve is shut. Then the peak pressure, and the intervals in which the
  outflow valve is open (``p_LV > p_AR_SYS``, ``p_RV > p_AR_PUL``: exactly where the
  circuit's smooth-diode flows ``Q_AV``/``Q_PV`` are positive), each with the volume
  ejected over it (``ejected_mL``, the volume actually ejected), and ``ejects``,
  whether any was. Then the largest conservation drift, and the Newton iteration
  counts with ``failed_at_ms``, the controller's time when the loop raised, if it did,
  and ``tref_scale``, the ``--tref`` the run used. After a restart, both cover the
  whole run, from ``log.csv``.
- ``timings.json``: this process's wall time in the EP ODE step, the EP PDE step and
  the mechanics solve (as in ``strong_coupling_zetasplit``), plus the loop and the
  whole run, ``setup_s`` (the start of ``main`` to the start of the loop) and
  ``newton_its``, the total of Newton iterations (after a restart, only the part it
  ran).
- ``steps.csv``, ``run.json`` (last, with the provenance of every process that wrote
  the run) and, with ``--snapshot-every``, ``snapshots.npz``: the scheme comparison's
  record of the run (``scheme_comparison/record.py``).

``post.py`` writes ``summary.json`` and ``pv_loops.png`` again from these, by the same
functions, with VTX fields and λ and ``Ta`` statistics, into ``post/``.

The folder rules are the CLIs'. A run refuses a folder that holds any of these files,
unless ``--overwrite`` (which deletes only them) or ``--restart`` is given. The default
folder may hold the results of a run from before these rules, so a plain re-run into it
is refused too. ``--restart`` continues from the checkpoint, and refuses one written
with other physics (:func:`physics`): it may change ``--t-end`` and the output options,
nothing else. If the checkpoint is at or past ``--t-end`` it takes no step. A restart
still builds the problem, and reads the circuit's operating point and the prestress
from the cache, but skips the inflation and the initial state of the problem and the
backend, which the checkpoint replaces.

Serial only: the cavity pressures and the circuit's states are read as ``x.array[0]``,
which assumes one rank holds them.

The mesh, the circuit's operating point and the prestressed displacement are cached
under ``meshes/``; delete that directory to recompute them. Generating the mesh needs
``ukb-atlas`` (with pyvista, for the clipping) and ``fenicsx-ldrb``; install them by
hand when you need them, ``python3 -m pip install ukb-atlas fenicsx-ldrb pyvista``
(the first two are also in the ``demo`` extra). Mind numpy: ``fenicsx-ldrb`` depends
on numba, which requires numpy < 2.5 (numba 0.65), so installing it downgrades a
newer numpy -- in the dev container 2.5.3 became 2.4.6 -- and putting that numpy back
afterwards leaves numba, and with it ``ldrb``, unimportable. Running from a cached
mesh needs none of them.
"""

import argparse
import functools
import hashlib
import json
import logging
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from mpi4py import MPI

import beat
import dolfinx
import io4dolfinx
import numpy as np
import pint
import pulse
import ufl
from circulation import base, regazzoni2020
from circulation.units import mmHg_to_kPa
from pulse.circulation import ChamberCoupling, GotranxCirculation, mL, mmHg

import cardiac_geometries
import cardiac_geometries.geometry
from simcardemsx.backends import GeneratedActivation
from simcardemsx.checkpoint import Checkpointer
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import CirculationClock
from simcardemsx.ode_model import load_ode_modules
from simcardemsx.results import ARTIFACTS, CsvLog, ResultsWriter, write_resolved_settings

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
# scheme_comparison, demo_io and this example's post sit next to this example, not in
# the installed package.
sys.path.insert(0, str(HERE.parent))
import demo_io  # noqa: E402
from scheme_comparison.record import Recorder  # noqa: E402

from circulation_biv.post import plot, summarise  # noqa: E402

ODEFILE = HERE.parent / "odefiles" / "ToRORd_dynCl_endo_zetasplit.ode"
GEODIR = HERE / "meshes" / "ukb_mean_ed_clipped"
PRESTRESS_FILE = "prestress_biv.bp"

CHAMBERS = ("LV", "RV")

CHAR_LENGTH = 10.0  # mm; the demo's: the atlas is smooth, so a coarse mesh suffices
#: The demo's quadrature degree, for the geometry, its ``Quadrature_6`` fibres and the
#: backend's states alike: FFCx evaluates a whole integral at a quadrature-element
#: coefficient's own degree, so all three must agree.
QUAD_DEGREE = 6

PERIOD = 1000.0  # ms: one beat, the circuit's RR = 1 / HR and ToR-ORd's pacing period

#: The default ``--tref``: physcardems' eLife calibration scales Land's ``Tref`` by 3
#: (``src/physcardems/parameters.py``, ``LAND_SCALES``). At the ``.ode``'s own ``Tref``
#: (``--tref 1``) the LV's isovolumic peak stays about 2 mmHg below aortic pressure, so
#: its aortic valve does not open in the first beat.
TREF_SCALE = 3.0
DT_MECH = 2.0  # ms, the demo's; the default of --dt-mech
SCHEMES = ("monolithic", "segregated", "stabilized")
DT_EP = 0.05  # ms

#: Absolute Newton tolerance of the coupled problem, tightened from pulse's default
#: 1e-6 as in every coupled problem of this package. The cavity constraint rows are
#: ``V_state - V(u)`` in m^3, so 1e-6 would allow volume errors of up to a millilitre.
SNES_ATOL = 1e-9

EP_RESULTS = ("v", "cai")
MECHANICS_RESULTS = ("u", "lmbda", "tension_kPa", "stiffness_kPa")
#: Everything a run writes into its output folder, and so what --overwrite deletes.
OUTPUT_ARTIFACTS = (*ARTIFACTS, *demo_io.RECORDER_ARTIFACTS, "summary.json", "pv_loops.png")
#: The arguments that do not define the physics: the run's length and its output.
NOT_PHYSICS = (
    "t_end",
    "outdir",
    "snapshot_every",
    "save_every",
    "save_every_ep",
    "checkpoint_every",
    "restart",
    "overwrite",
)


def accumulate_time(fn: Callable, totals: dict[str, float], key: str) -> Callable:
    """Wrap ``fn`` so that the wall time spent in it is added to ``totals[key]``."""

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            totals[key] += time.perf_counter() - start

    return timed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with ``save_every`` resolved from its default."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=PERIOD,
        help="End time in ms (default: %(default)s, one beat). "
        "Rounded to a whole number of mechanics steps.",
    )
    parser.add_argument(
        "--scheme",
        choices=SCHEMES,
        default="monolithic",
        help="Coupling scheme of the mechanics (default: %(default)s).",
    )
    parser.add_argument(
        "--dt-mech",
        type=float,
        default=DT_MECH,
        help="Mechanics time step in ms, a whole multiple of the EP step "
        f"({DT_EP} ms) (default: %(default)s).",
    )
    parser.add_argument(
        "--snapshot-every",
        type=float,
        default=None,
        help="Save per-point snapshots of the stretch, tension and stiffness every this "
        "many ms, to snapshots.npz (default: off).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=HERE / "output",
        help="Output directory (default: %(default)s). It must hold no results of an "
        "earlier run, unless --overwrite or --restart is given.",
    )
    parser.add_argument(
        "--tref",
        type=float,
        default=TREF_SCALE,
        help="Scale of Land's Tref, the .ode's 120 kPa (default: %(default)s, physcardems' "
        "eLife calibration). At 1 the LV does not eject in the first beat.",
    )
    parser.add_argument(
        "--save-every",
        type=float,
        default=None,
        help="Save u, lmbda, tension_kPa and stiffness_kPa to results.bp every this many "
        "ms, a whole multiple of --dt-mech (default: --dt-mech).",
    )
    parser.add_argument(
        "--save-every-ep",
        type=float,
        default=1.0,
        help=f"Save EP's v and cai to results.bp every this many ms, a whole multiple of "
        f"the {DT_EP} ms EP step (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=50.0,
        help="Write a checkpoint (restart.bp, restart.json) every this many ms, a whole "
        "multiple of --dt-mech, and at the end of the run (default: %(default)s).",
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
        args.save_every = args.dt_mech
    return args


# ---------------------------------------------------------------------------
# The model, as the run reads it
# ---------------------------------------------------------------------------


def _file(path: Path) -> dict[str, str]:
    """``path``'s file name and sha256. Raises ``FileNotFoundError`` if it is missing."""
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def material_parameters() -> dict[str, dict[str, Any]]:
    """The mechanics' material as :func:`cardiac_model` gives it to pulse, as
    ``pulse.Variable``s: Holzapfel-Ogden's transversely isotropic parameters, the
    compressibility's ``kappa`` and the viscosity's ``eta``, all pulse's defaults."""
    return {
        "HolzapfelOgden": dict(pulse.HolzapfelOgden.transversely_isotropic_parameters()),
        "Compressible": {"kappa": pulse.Compressible().kappa},
        "Viscous": {"eta": pulse.viscoelasticity.Viscous().eta},
    }


def circuit_parameters(period_ms: float) -> dict[str, float]:
    """Regazzoni's parameters at one beat per ``period_ms``. The heart rate goes in
    before flattening: ``flat_ode_parameters`` derives ``RR`` and the chambers'
    activation offsets from it, which overriding the flat ``HR`` would not."""
    nested = base.remove_units(regazzoni2020.Regazzoni2020.default_parameters())
    flat = regazzoni2020.flat_ode_parameters(nested | {"HR": 1000.0 / period_ms})
    return {name: float(value) for name, value in flat.items()}


def model_settings() -> dict[str, Any]:
    """The model as the run reads it, as plain values: EP's time step, the beat's
    period, the geometry (the atlas shape, mesh size, base normal, fibres and units),
    EP (its ODE space and parameters, conductivities, chi and C_m), the mechanics
    (spaces, quadrature degree, material in SI base units, boundary conditions and
    density), the circuit (its ``.ode`` file by name and sha256, the components
    dropped, its time scheme and parameters), and how the initial state is reached
    (the circuit's limit cycle, the prestress and the inflation).

    The material is read off pulse and the circuit's parameters off ``circulation``, so
    a change to either library's defaults changes these too. Raises
    ``FileNotFoundError`` if the circuit's ``.ode`` file is missing.
    """
    material = material_parameters()
    return {
        "dt_ep": DT_EP,
        "period_ms": PERIOD,
        "geometry": {
            "atlas": {"mode": -1, "std": 0, "case": "ED", "clipped": True},
            "char_length_mm": CHAR_LENGTH,
            # Rotated so the base normal is this axis: the sliding base holds u_x.
            "base_normal": [1.0, 0.0, 0.0],
            # The atlas is in mm; the mesh is scaled to metres, as the chamber coupling
            # assumes.
            "mesh_unit": "m",
            "fibres": {
                "space": f"Quadrature_{QUAD_DEGREE}",
                "ldrb_angles": {
                    "alpha_endo_lv": 60,
                    "alpha_epi_lv": -60,
                    "alpha_endo_rv": 90,
                    "alpha_epi_rv": -25,
                    "beta_endo_lv": -20,
                    "beta_epi_lv": 20,
                    "beta_endo_rv": 0,
                    "beta_epi_rv": 20,
                },
            },
        },
        "ep": {
            "ode_element": ["P", 1],
            # ToR-ORd's own stimulus, with its period set to the beat; no PDE stimulus.
            "ode_parameters": {"i_Stim_Period": PERIOD},
            "theta": 1,
            "chi_per_mm": 140.0,
            "C_m_uF_per_mm2": 0.01,
            "conductivities_S_per_m": {
                "sigma_el": 0.62,
                "sigma_et": 0.24,
                "sigma_il": 0.17,
                "sigma_it": 0.019,
            },
        },
        "mechanics": {
            "quadrature_degree": QUAD_DEGREE,
            "u_space": "P_2",
            "material": {
                "units": "SI base units (Pa, Pa s)",
                **{
                    model: {name: float(value.to_base_units()) for name, value in values.items()}
                    for model, values in material.items()
                },
            },
            # Springs, then dampers, on these markers, in this order.
            "robin_Pa_per_m": {"EPI": 1.0e5, "BASE": 1.0e6},
            "robin_damping_Pa_s_per_m": {"EPI": 5.0e3, "BASE": 5.0e3},
            "dirichlet": {"BASE": "u_x = 0 (sliding in its plane)"},
            "rho_kg_per_m3": 1e3,
        },
        "circulation": {
            "ode": _file(Path(regazzoni2020.ODE_FILE)),
            "drop_components": ["timing", "LV", "RV"],
            "scheme": "backward_euler",
            "time_unit": "s",
            "parameters": circuit_parameters(PERIOD),
        },
        "initial_state": {
            "circuit_limit_cycle": {"beats": 10, "dt_s": 0.001},
            "prestress_ramp_steps": 20,
            # The steps of the volume-controlled inflation after the unloaded solve.
            "inflation_steps": 19,
        },
    }


def settings(args: argparse.Namespace) -> dict[str, Any]:
    """What ``config.resolved.toml`` holds: every argument, ``odefile`` (the ``.ode``
    file's path, file name and sha256), and :func:`model_settings`.

    TOML has no null, so an argument that is ``None`` is not in the file. Raises
    ``FileNotFoundError`` if an ``.ode`` file does not exist.
    """
    arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    arguments["odefile"] = {"path": str(ODEFILE), **_file(ODEFILE)}
    return {**arguments, **model_settings()}


def physics(args: argparse.Namespace) -> dict[str, Any]:
    """What a restart must share with the checkpointed run: :func:`settings` without the
    run length and the output options (:data:`NOT_PHYSICS`), and without the ``.ode``
    file's path. Solver options are not in it, as in the CLIs.

    Raises ``FileNotFoundError`` if an ``.ode`` file does not exist.
    """
    result = settings(args)
    for key in NOT_PHYSICS:
        del result[key]
    del result["odefile"]["path"]
    return result


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def generate_mesh(geodir: Path, comm: MPI.Intracomm, run_settings: Mapping[str, Any]) -> None:
    """The demo's mesh: the atlas mean shape at end diastole, clipped at the valve plane
    so the mesh has a single ``BASE``, with LDRB fibres on ``Quadrature_6``."""
    import ldrb  # type: ignore[import-untyped,import-not-found]

    geometry = run_settings["geometry"]
    atlas = geometry["atlas"]
    logger.info(f"Generating the UKB mesh in {geodir}")
    geo = cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=atlas["mode"],
        std=atlas["std"],
        case=atlas["case"],
        char_length_max=geometry["char_length_mm"],
        char_length_min=geometry["char_length_mm"],
        clipped=atlas["clipped"],
    )
    geo = geo.rotate(target_normal=geometry["base_normal"], base_marker="BASE")
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=atlas["clipped"]),
        **geometry["fibres"]["ldrb_angles"],
        fiber_space=geometry["fibres"]["space"],
    )
    if (geodir / "geometry.bp").exists():
        shutil.rmtree(geodir / "geometry.bp")
    cardiac_geometries.geometry.save_geometry(
        path=geodir / "geometry.bp",
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        info=geo.info,
        f0=system.f0,
        s0=system.s0,
        n0=system.n0,
    )


def base_normal(geometry: pulse.HeartGeometry) -> np.ndarray:
    """Area-averaged outward normal of ``BASE``, as a unit vector (the demo's check)."""
    comm = geometry.mesh.comm
    n = ufl.FacetNormal(geometry.mesh)
    ds = geometry.ds(geometry.markers["BASE"][0])
    vector = np.array(
        [
            comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(n[i] * ds)), op=MPI.SUM)
            for i in range(3)
        ],
    )
    return vector / np.linalg.norm(vector)


def load_geometry(geodir: Path, comm: MPI.Intracomm, run_settings: Mapping[str, Any]):
    """The cached mesh, rotated so the base normal is ``run_settings``' (x), in metres,
    with the mechanics' quadrature degree.

    Rotated after loading, as in the demo: the folder also holds the unrotated
    ``.msh``, which is what ``from_folder`` gives back, and the sliding-base condition
    constrains ``u_x`` only.
    """
    target = run_settings["geometry"]["base_normal"]
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    geo = geo.rotate(target_normal=target, base_marker="BASE")
    geo.mesh.geometry.x[:] *= 1e-3  # mm -> m: the chamber coupling assumes metres
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": run_settings["mechanics"]["quadrature_degree"]},
    )
    up = base_normal(geometry)
    if abs(float(np.dot(up, target))) < 0.99:
        raise RuntimeError(
            f"the base normal is {up.round(3)}, not the {target} axis the sliding-base "
            "condition assumes -- the rotation did not take effect",
        )
    return geo, geometry


# ---------------------------------------------------------------------------
# Mechanics: the demo's material, compressibility and boundary conditions
# ---------------------------------------------------------------------------


def cardiac_model(f0, s0, active: pulse.active_model.ActiveModel) -> pulse.CardiacModel:
    """Holzapfel-Ogden (transversely isotropic), compressible, viscous, with
    :func:`material_parameters`.

    The viscous term does nothing without a strain rate, so the static prestressing
    and inflation are unaffected by it.
    """
    parameters = material_parameters()
    return pulse.CardiacModel(
        material=pulse.HolzapfelOgden(f0=f0, s0=s0, **parameters["HolzapfelOgden"]),
        active=active,
        compressibility=pulse.Compressible(**parameters["Compressible"]),
        viscoelasticity=pulse.viscoelasticity.Viscous(**parameters["Viscous"]),
    )


def robin_bcs(
    geometry: pulse.HeartGeometry,
    mechanics: Mapping[str, Any],
) -> tuple[pulse.RobinBC, ...]:
    """Springs on the epicardium and base, with the dynamic arm's damping on both, as
    ``mechanics`` (the run settings' ``"mechanics"``) gives them."""

    def spring(marker: str, value: float, damping: bool = False) -> pulse.RobinBC:
        return pulse.RobinBC(
            value=pulse.Variable(
                dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(value)),
                "Pa s/ m" if damping else "Pa / m",
            ),
            marker=geometry.markers[marker][0],
            damping=damping,
        )

    return (
        *(spring(marker, value) for marker, value in mechanics["robin_Pa_per_m"].items()),
        *(
            spring(marker, value, damping=True)
            for marker, value in mechanics["robin_damping_Pa_s_per_m"].items()
        ),
    )


def sliding_base(geometry: pulse.HeartGeometry) -> Callable:
    """Hold the base in its own plane (``u_x = 0``) but let it slide within it."""

    facet_tags = geometry.facet_tags
    assert facet_tags is not None  # the UKB mesh is always written with facet markers

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.bcs.DirichletBC]:
        facets = facet_tags.find(geometry.markers["BASE"][0])
        dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]

    return dirichlet_bc


# ---------------------------------------------------------------------------
# The operating point: circuit limit cycle, prestressing, inflation
# ---------------------------------------------------------------------------


def circuit_operating_point(
    EDV: dict[str, float],
    cachedir: Path,
    comm: MPI.Intracomm,
    run_settings: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """The circuit alone, from the mesh's end-diastolic volumes, run to a limit cycle.

    Returns its state (circuit units) and its end-diastolic pressures (kPa).
    """
    state_file = cachedir / "circ_state.json"
    if comm.rank == 0 and not state_file.exists():
        logger.info("Running the circuit alone to a limit cycle...")
        limit_cycle = run_settings["initial_state"]["circuit_limit_cycle"]
        standalone = regazzoni2020.Regazzoni2020(
            parameters={"HR": 1000.0 / run_settings["period_ms"]},
            add_units=False,
            outdir=cachedir / "regazzoni_standalone",
        )
        history = standalone.solve(
            num_beats=limit_cycle["beats"],
            initial_state={f"V_{c}": EDV[c] / mL for c in CHAMBERS},
            dt=limit_cycle["dt_s"],
        )
        state = dict(zip(standalone.state_names(), map(float, standalone.state)))
        cache = {"state": state} | {f"p_{c}_ED": float(history[f"p_{c}"][-1]) for c in CHAMBERS}
        state_file.write_text(json.dumps(cache, indent=4))
    comm.barrier()
    cached = json.loads(state_file.read_text())
    p_ED = {c: mmHg_to_kPa(cached[f"p_{c}_ED"]) for c in CHAMBERS}
    return {k: float(v) for k, v in cached["state"].items()}, p_ED


def _vector_space(mesh: dolfinx.mesh.Mesh, u_space: str) -> dolfinx.fem.FunctionSpace:
    """pulse's displacement space ``u_space`` (e.g. ``"P_2"``) on ``mesh``."""
    family, degree = u_space.split("_")
    return dolfinx.fem.functionspace(mesh, (family, int(degree), (mesh.geometry.dim,)))


def read_prestress(
    mesh: dolfinx.mesh.Mesh,
    cachedir: Path,
    run_settings: Mapping[str, Any],
) -> dolfinx.fem.Function:
    """The cached displacement from the unloaded to the loaded (as meshed)
    configuration, on the mechanics' displacement space.

    Raises ``FileNotFoundError`` if it is not cached.
    """
    fname = cachedir / PRESTRESS_FILE
    if not fname.exists():
        raise FileNotFoundError(f"No prestress in the cache: {fname} does not exist")
    u_pre = dolfinx.fem.Function(_vector_space(mesh, run_settings["mechanics"]["u_space"]))
    io4dolfinx.read_function(fname, u_pre, time=0.0, name="u_pre")
    return u_pre


def prestress(
    geometry: pulse.HeartGeometry,
    geo,
    p_ED: dict[str, float],
    cachedir: Path,
    comm: MPI.Intracomm,
    run_settings: Mapping[str, Any],
) -> dolfinx.fem.Function:
    """The displacement from the unloaded to the loaded (as meshed) configuration,
    computed and cached if it is not cached yet."""
    fname = cachedir / PRESTRESS_FILE
    mechanics = run_settings["mechanics"]
    if not fname.exists():
        logger.info("Prestressing to recover the unloaded reference configuration...")
        traction = {
            c: pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa") for c in CHAMBERS
        }
        problem = pulse.unloading.PrestressProblem(
            geometry=geometry,
            model=cardiac_model(geo.f0, geo.s0, pulse.active_model.Passive()),
            bcs=pulse.BoundaryConditions(
                robin=robin_bcs(geometry, mechanics),
                dirichlet=(sliding_base(geometry),),
                neumann=tuple(
                    pulse.NeumannBC(traction=traction[c], marker=geometry.markers[c][0])
                    for c in CHAMBERS
                ),
            ),
            parameters={
                "u_space": mechanics["u_space"],
                "mesh_unit": run_settings["geometry"]["mesh_unit"],
            },
            targets=[
                pulse.unloading.TargetPressure(traction=traction[c], target=p_ED[c], name=c)
                for c in CHAMBERS
            ],
            ramp_steps=run_settings["initial_state"]["prestress_ramp_steps"],
        )
        u_pre = problem.unload()
        io4dolfinx.write_function_on_input_mesh(fname, u_pre, time=0.0, name="u_pre")
    comm.barrier()
    return read_prestress(geometry.mesh, cachedir, run_settings)


def inflate(
    geometry: pulse.HeartGeometry,
    f0,
    s0,
    unloaded: dict[str, float],
    EDV: dict[str, float],
    run_settings: Mapping[str, Any],
) -> pulse.StaticProblem:
    """Volume-controlled inflation, passive, from the unloaded volumes to ``EDV``."""
    mechanics = run_settings["mechanics"]
    volume = {
        c: dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(unloaded[c]))
        for c in CHAMBERS
    }
    inflation = pulse.StaticProblem(
        model=cardiac_model(f0, s0, pulse.active_model.Passive()),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(
            robin=robin_bcs(geometry, mechanics),
            dirichlet=(sliding_base(geometry),),
        ),
        cavities=[pulse.problem.Cavity(marker=c, volume=volume[c]) for c in CHAMBERS],
        parameters={
            "mesh_unit": run_settings["geometry"]["mesh_unit"],
            "u_space": mechanics["u_space"],
        },
    )
    inflation.solve()
    steps = run_settings["initial_state"]["inflation_steps"]
    for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
        for c in CHAMBERS:
            volume[c].value = unloaded[c] + fraction * (EDV[c] - unloaded[c])
        if not inflation.solve():
            raise RuntimeError(f"inflation failed at {fraction:.2f} of the way to end diastole")
    return inflation


# ---------------------------------------------------------------------------
# EP
# ---------------------------------------------------------------------------


def make_ep_solver(
    ep_module,
    ode_space: dolfinx.fem.FunctionSpace,
    f0,
    dx: ufl.Measure,
    run_settings: Mapping[str, Any],
) -> beat.MonodomainSplittingSolver:
    """beat's monodomain splitting solver on ``ode_space``'s mesh (in metres), with
    ``ode_space`` (P1) as its ODE space.

    No stimulus current in the PDE: ToR-ORd's own cellular stimulus fires at every
    point at t = 0, with its period set to the beat. States, parameters and missing
    variables have one column per point: the transfer plan writes ``lmbda`` and the
    backend's outputs back into these arrays in place, and they differ between points.
    """
    ep = run_settings["ep"]
    mesh = ode_space.mesh
    mesh_unit = run_settings["geometry"]["mesh_unit"]
    sigma = ep["conductivities_S_per_m"]
    chi: pint.registry.Quantity = ep["chi_per_mm"] * beat.units.ureg("mm**-1")
    C_m: pint.registry.Quantity = ep["C_m_uF_per_mm2"] * beat.units.ureg("uF/mm**2")
    M = beat.conductivities.define_conductivity_tensor(
        chi=chi,
        f0=f0,
        g_il=sigma["sigma_il"] * beat.units.ureg("S/m"),
        g_it=sigma["sigma_it"] * beat.units.ureg("S/m"),
        g_el=sigma["sigma_el"] * beat.units.ureg("S/m"),
        g_et=sigma["sigma_et"] * beat.units.ureg("S/m"),
    )
    pde = beat.MonodomainModel(
        time=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        mesh=mesh,
        M=M,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        dx=dx,
    )
    v_ode = dolfinx.fem.Function(ode_space)
    num_points = v_ode.x.array.size
    states = np.tile(ep_module.init_state_values()[:, None], (1, num_points))
    parameters = np.tile(
        ep_module.init_parameter_values(**ep["ode_parameters"])[:, None],
        (1, num_points),
    )
    missing = getattr(ep_module, "missing", {})
    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=ep_module.generalized_rush_larsen,
        init_states=states,
        parameters=parameters,
        num_states=states.shape[0],
        v_index=ep_module.state_index("v"),
        missing_variables=np.zeros((len(missing), num_points)) if missing else None,
        num_missing_variables=len(missing),
    )
    return beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=ep["theta"])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def log_fields(circuit_states: Sequence[str]) -> tuple[str, ...]:
    """``log.csv``'s columns, with a ``circuit_<name>`` column per circuit state, in the
    circuit's order."""
    return (
        "t_ms",
        *(f"V_{c}_mL" for c in CHAMBERS),
        *(f"p_{c}_mmHg" for c in CHAMBERS),
        *(f"circuit_{name}" for name in circuit_states),
        "total_volume_mL",
        "conservation_drift",
        "Ta_mean_kPa",
        "newton_iterations",
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    start_total = time.perf_counter()
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    for lib in ("numba", "matplotlib", "scifem"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        # The cavity pressures and the circuit's states are read as x.array[0].
        raise SystemExit("circulation_biv runs in serial only")
    outdir: Path = args.outdir
    dt_mech = args.dt_mech

    # Everything that can refuse the run does so here: before any file is touched, and
    # before anything expensive is generated, built or compiled.
    strides, run_physics = demo_io.prepare_run(
        outdir,
        restart=args.restart,
        overwrite=args.overwrite,
        strides={
            "--dt-mech": (dt_mech, DT_EP),
            "--save-every-ep": (args.save_every_ep, DT_EP),
            "--save-every": (args.save_every, dt_mech),
            "--checkpoint-every": (args.checkpoint_every, dt_mech),
        },
        physics=lambda: physics(args),
        artifacts=OUTPUT_ARTIFACTS,
    )
    save_ep_every = strides["--save-every-ep"]
    save_every = strides["--save-every"]
    checkpoint_every = strides["--checkpoint-every"]

    run_settings = settings(args)
    # Every run's, a restart's too: the latest run's settings win, as in the CLIs.
    if comm.rank == 0:
        write_resolved_settings(outdir / "config.resolved.toml", run_settings)
    mechanics_settings = run_settings["mechanics"]
    circulation_settings = run_settings["circulation"]
    period = run_settings["period_ms"]

    cachedir = GEODIR / "cache"
    if comm.rank == 0:
        cachedir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    # ---------------------------------------------------------
    # 1. Generated ODE code
    # ---------------------------------------------------------
    modules = load_ode_modules(ODEFILE, HERE / "generated_odes" / ODEFILE.stem)

    # ---------------------------------------------------------
    # 2. Geometry and the operating point
    # ---------------------------------------------------------
    if not (GEODIR / "geometry.bp").exists():
        generate_mesh(GEODIR, comm, run_settings)
    comm.barrier()
    geo, geometry = load_geometry(GEODIR, comm, run_settings)

    EDV = {c: comm.allreduce(geometry.volume(c), op=MPI.SUM) for c in CHAMBERS}
    logger.info(
        f"Mesh end-diastolic volumes: LV {EDV['LV'] / mL:.1f} mL, RV {EDV['RV'] / mL:.1f} mL",
    )

    circ_state, p_ED = circuit_operating_point(EDV, cachedir, comm, run_settings)
    logger.info(
        f"Circuit end-diastolic pressures: LV {p_ED['LV']:.2f} kPa, RV {p_ED['RV']:.2f} kPa",
    )

    u_pre = prestress(geometry, geo, p_ED, cachedir, comm, run_settings)
    geometry.deform(u_pre)
    f0 = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
    s0 = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")

    if not args.restart:
        # A restart takes the configuration, the pressures and the circuit state from
        # the checkpoint instead.
        unloaded = {c: comm.allreduce(geometry.volume(c), op=MPI.SUM) for c in CHAMBERS}
        logger.info(
            f"Unloaded volumes: LV {unloaded['LV'] / mL:.1f} mL, RV {unloaded['RV'] / mL:.1f} mL",
        )
        inflation = inflate(geometry, f0, s0, unloaded, EDV, run_settings)
        p_inflated = [float(p.x.array[0]) for p in inflation.cavity_pressures]
        # The circuit starts from the volumes the inflation actually reached, so the
        # cavity constraint holds at t = 0 and the conservation check starts from a
        # consistent state.
        V0 = {
            c: comm.allreduce(geometry.volume(c, u=inflation.u), op=MPI.SUM) / mL for c in CHAMBERS
        }
        logger.info(
            f"Inflated to LV {V0['LV']:.2f} mL at {p_inflated[0] / mmHg:.2f} mmHg, "
            f"RV {V0['RV']:.2f} mL at {p_inflated[1] / mmHg:.2f} mmHg",
        )

    # ---------------------------------------------------------
    # 3. The coupled mechanics problem
    # ---------------------------------------------------------
    tref = modules.mechanics.init_parameter_values()[modules.mechanics.parameter["Tref"]]
    backend = GeneratedActivation(
        modules.mechanics,
        geometry.mesh,
        f0,
        quadrature_degree=mechanics_settings["quadrature_degree"],
        parameters={"Tref": args.tref * float(tref)},
        scheme=args.scheme,
    )
    logger.info(f"Tref = {args.tref} x {float(tref)} kPa")
    circuit = GotranxCirculation(
        ode_file=regazzoni2020.ODE_FILE,
        parameters=circulation_settings["parameters"],
        drop_components=tuple(circulation_settings["drop_components"]),
    )
    beat_phase = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    problem = pulse.DynamicProblem(
        model=cardiac_model(f0, s0, backend),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(
            robin=robin_bcs(geometry, mechanics_settings),
            dirichlet=(sliding_base(geometry),),
        ),
        # No volume: each chamber coupling replaces it with the circuit's volume state.
        cavities=[pulse.problem.Cavity(marker=c, volume=None) for c in CHAMBERS],
        circulation=circuit,
        chambers=[ChamberCoupling(c, f"V_{c}", f"p_{c}") for c in CHAMBERS],
        circulation_missing={"beat_phase": beat_phase},
        parameters={
            "mesh_unit": run_settings["geometry"]["mesh_unit"],
            "u_space": mechanics_settings["u_space"],
            "circulation_scheme": circulation_settings["scheme"],
            "rho": pulse.Variable(mechanics_settings["rho_kg_per_m3"], "kg/m^3"),
            "dt": pulse.Variable(dt_mech * 1e-3, "s"),
            "petsc_options": petsc_options,
        },
    )

    if not args.restart:
        # Start from the inflated configuration, at rest, and from the circuit state
        # that matches it.
        problem.u.x.array[:] = inflation.u.x.array
        problem.u_old.x.array[:] = inflation.u.x.array
        for i, pressure in enumerate(p_inflated):
            problem.cavity_pressures[i].x.array[:] = pressure
            problem.cavity_pressures_old[i].x.array[:] = pressure
        problem.v_old.x.array[:] = 0.0
        problem.a_old.x.array[:] = 0.0
        for name, *states in zip(
            circuit.state_names,
            problem.circulation_states,
            problem.circulation_states_old,
            problem.circulation_states_prev,
        ):
            value = V0[name[2:]] if name in ("V_LV", "V_RV") else circ_state[name]
            for state in states:
                state.x.array[:] = value

    # ---------------------------------------------------------
    # 4. EP and the controller
    # ---------------------------------------------------------
    # P1: values going back to EP are averaged onto the EP ODE space, which the
    # transfer plan supports for P1 and DG0 only.
    ep_ode_space = dolfinx.fem.functionspace(
        geometry.mesh,
        tuple(run_settings["ep"]["ode_element"]),
    )
    ep_solver = make_ep_solver(modules.ep, ep_ode_space, f0, geometry.dx, run_settings)
    ode = ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)  # as make_ep_solver builds it
    clock = CirculationClock(
        problem,
        time_unit=circulation_settings["time_unit"],
        beat_phase=beat_phase,
        period=period,
    )
    controller = SimulationController(
        clock,
        ep_solver,
        backend,
        modules,
        dt_mech,
        run_settings["dt_ep"],
    )

    if not args.restart:
        # The tissue starts at rest at the inflated stretch, not at lambda = 1: without
        # this, the first step would see a stretch rate of (lambda(u_0) - 1) / dt, and
        # the zeta states and EP's troponin would start from the unloaded stretch.
        backend.reset_stretch()
        controller.plan.backward()

    material = material_parameters()
    a = material["HolzapfelOgden"]["a"]
    recorder = Recorder(
        backend,
        outdir,
        snapshot_every_ms=args.snapshot_every,
        run_info={
            "geometry": "biv",
            "split": "zetasplit",
            "scheme": args.scheme,
            "dt_mech_ms": dt_mech,
            "t_end_ms": args.t_end,
            "regime": {
                # Uniaxial small-strain stiffness 3a of Holzapfel-Ogden's isotropic term.
                "Kp_kPa": float(3 * a.to_base_units() / 1e3),
                "eta_Pa_s": float(material["Viscous"]["eta"].to_base_units()),
                "rho_kg_m3": mechanics_settings["rho_kg_per_m3"],
                "h_m": run_settings["geometry"]["char_length_mm"] * 1e-3,
            },
        },
    )
    # The recorder's rows and snapshots are restored with the rest of the run.
    checkpointer = Checkpointer(controller, outdir, physics=run_physics, extra=[recorder])
    results = ResultsWriter(outdir)
    log = CsvLog(outdir / "log.csv", log_fields(circuit.state_names))
    newton_its = 0

    timings = {"ep_ode_s": 0.0, "ep_pde_s": 0.0, "mech_s": 0.0}
    ep_solver.ode.step = accumulate_time(ep_solver.ode.step, timings, "ep_ode_s")  # type: ignore[method-assign]
    ep_solver.pde.step = accumulate_time(ep_solver.pde.step, timings, "ep_pde_s")  # type: ignore[method-assign]
    problem.solve = accumulate_time(problem.solve, timings, "mech_s")  # type: ignore[method-assign]

    # ---------------------------------------------------------
    # 5. Recording
    # ---------------------------------------------------------
    volume_forms = {
        c: dolfinx.fem.form(
            geometry.volume_form(u=problem.u) * geometry.ds(geometry.markers[c][0]),
        )
        for c in CHAMBERS
    }
    myocardium = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * geometry.dx)),
        op=MPI.SUM,
    )
    tension_form = dolfinx.fem.form(backend.active_tension * geometry.dx)
    compliance = {
        state: circuit.parameters[f"C_{state[2:]}"]  # type: ignore[index]
        for state in ("p_AR_SYS", "p_VEN_SYS", "p_AR_PUL", "p_VEN_PUL")
    }
    #: log.csv's rows, from t = 0: the summary and the plot are drawn from them, and
    #: conservation_drift is measured from the first.
    rows: list[dict[str, float]] = []

    def record(t: float, newton_iterations: int) -> dict[str, float]:
        y = {
            name: float(state.x.array[0])
            for name, state in zip(circuit.state_names, problem.circulation_states)
        }
        V = {
            c: comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM) / mL
            for c, form in volume_forms.items()
        }
        # The circulation package's total blood volume (Regazzoni2020.compute_volumes),
        # with the 3D cavities in place of the circuit's V_LV and V_RV.
        total = (
            y["V_LA"]
            + y["V_RA"]
            + V["LV"]
            + V["RV"]
            + sum(C * y[state] for state, C in compliance.items())
        )
        row = {"t_ms": t}
        row |= {f"V_{c}_mL": V[c] for c in CHAMBERS}
        row |= {
            f"p_{c}_mmHg": float(problem.cavity_pressures[i].x.array[0]) / mmHg
            for i, c in enumerate(CHAMBERS)
        }
        row |= {f"circuit_{name}": value for name, value in y.items()}
        row["total_volume_mL"] = total
        total_0 = rows[0]["total_volume_mL"] if rows else total
        row["conservation_drift"] = abs(total - total_0) / abs(total_0)
        row["Ta_mean_kPa"] = (
            comm.allreduce(dolfinx.fem.assemble_scalar(tension_form), op=MPI.SUM) / myocardium
        )
        row["newton_iterations"] = newton_iterations
        return row

    def log_step(t: float, newton_iterations: int) -> dict[str, float]:
        row = record(t, newton_iterations)
        log.append(row)
        rows.append(row)
        return row

    # results.bp's fields. EP's are P1 copies of rows of the ODE's state array, refreshed
    # before each save; the others are the problem's and the backend's own Functions.
    ep_fields = {name: dolfinx.fem.Function(ep_ode_space, name=name) for name in EP_RESULTS}
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

    def write_initial() -> None:
        save_ep(controller.t)
        results.write(controller.t, mechanics_fields)
        log_step(controller.t, 0)

    # A refused or failed restore leaves the folder as it was: nothing below it, neither
    # the end checkpoint nor run.json, is reached. On a fresh run write_initial logs the
    # row at t = 0 into rows; a restart gets the rows up to its checkpoint from log.csv.
    resumed = demo_io.start_or_resume(
        checkpointer,
        log,
        results,
        restart=args.restart,
        result_names=[*EP_RESULTS, *MECHANICS_RESULTS],
        write_initial=write_initial,
    )
    rows.extend(resumed)

    # The controller counts steps from 1. A step that fails is rolled back, but the EP
    # saves of its micro-steps stay in results.bp: a restart from the step's start
    # recomputes them bit for bit, so it does not save them again.
    def on_ep_step(t: float, ep_step_idx: int) -> None:
        if ep_step_idx % save_ep_every == 0:
            save_ep(t)

    def on_mech_step(t: float, step: int, newton_iterations: int) -> None:
        nonlocal newton_its
        newton_its += newton_iterations
        recorder.step(t, newton_iterations)
        row = log_step(t, newton_iterations)
        if step % save_every == 0:
            results.write(t, mechanics_fields)
        if step % 25 == 0 or step == 1:
            logger.info(
                f"t={t:6.1f} ms  LV {row['V_LV_mL']:6.1f} mL {row['p_LV_mmHg']:6.1f} mmHg  "
                f"RV {row['V_RV_mL']:6.1f} mL {row['p_RV_mmHg']:5.1f} mmHg  "
                f"Ta={row['Ta_mean_kPa']:6.2f} kPa  drift={row['conservation_drift']:.1e}  "
                f"newton={newton_iterations}",
            )

    # ---------------------------------------------------------
    # 6. Run
    # ---------------------------------------------------------
    # The run is round(t_end / dt_mech) steps from t = 0, so a restart at or past it
    # takes none.
    remaining_steps = demo_io.steps_to_take(controller, args.t_end)
    failure: str | None = None
    t_fail: float | None = None
    start_loop = time.perf_counter()
    timings["setup_s"] = start_loop - start_total
    try:
        for _ in range(remaining_steps):
            controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)
            if controller.mech_step_idx % checkpoint_every == 0:
                checkpointer.write()
    except BaseException as e:
        # BaseException: an interrupt is recorded too. A step that raised before its
        # mech_callback was rolled back (t_failed is set); anything raised after that
        # follows an accepted step, whose output may be half written.
        failure, t_fail = demo_io.failure_at(controller, e)
        raise
    finally:
        timings["loop_s"] = time.perf_counter() - start_loop
        timings["total_s"] = time.perf_counter() - start_total
        timings["newton_its"] = newton_its
        logger.info(f"Timings: {timings}")

        def columns() -> dict[str, np.ndarray]:
            return {key: np.array([row[key] for row in rows]) for key in log.fields}

        def write_summary() -> None:
            summary = summarise(columns(), t_fail)
            summary["tref_scale"] = args.tref
            if comm.rank == 0:
                (outdir / "summary.json").write_text(json.dumps(summary, indent=4))
            logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        def write_timings() -> None:
            if comm.rank == 0:
                (outdir / "timings.json").write_text(json.dumps(timings, indent=4))

        def write_plot() -> None:
            if comm.rank == 0:
                plot(columns(), outdir / "pv_loops.png")

        # The end checkpoint and this example's own files first, each guarded, and
        # run.json (the mark of a finished run, for scheme_comparison/run.py) last.
        demo_io.finish(
            recorder,
            checkpointer,
            [
                ("summary.json", write_summary),
                ("timings.json", write_timings),
                ("pv_loops.png", write_plot),
            ],
            failure=failure,
            t_fail_ms=t_fail,
            timings=timings,
            here=HERE,
            restart=args.restart,
        )


if __name__ == "__main__":
    main()
