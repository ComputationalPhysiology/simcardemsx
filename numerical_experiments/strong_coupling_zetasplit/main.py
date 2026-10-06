"""The worked example: EP and mechanics coupled on a slab, for any split of an ``.ode`` file.

EP (fenicsx-beat, monodomain) and mechanics (fenicsx-pulse, quasistatic) are coupled
through :class:`~simcardemsx.controller.SimulationController`, with the ``mechanics``
component of ``--odefile`` stepped inside the mechanics Newton iteration by
:class:`~simcardemsx.backends.GeneratedActivation` (or, with ``--crossbridge``, a
crossbridge model through :class:`~simcardemsx.backends.CrossbridgeSegregated`).

Output goes to ``--output-dir`` (default ``output/<odefile stem>/``), in the upstream
CLIs' layout (:mod:`simcardemsx.results`):

- ``config.resolved.toml``: :func:`settings`, rewritten by every run, fresh or restarted.
- ``results.bp``: EP's ``v`` and ``cai`` (P1) every ``--save-every-ep`` ms, and ``u``,
  ``lmbda``, ``tension_kPa`` and ``stiffness_kPa`` (the last three on the backend's
  quadrature points) every ``--save-every`` ms, all from t = 0.
- ``log.csv``: one row per mechanics step, and one at t = 0: ``t_ms``,
  ``newton_iterations``, and the mesh means ``lmbda_mean`` and ``Ta_mean_kPa``.
- ``restart.bp`` and ``restart.json``: a checkpoint every ``--checkpoint-every`` ms and
  at the end of the run, with the scheme comparison's ``restart_recorder_<t>.npz``.
- ``timings.json``: this process's wall time in the EP ODE step, the EP PDE step and the
  mechanics solve, and its Newton iterations (after a restart, only the part it ran).
- The scheme comparison's ``steps.csv``, ``run.json`` (last, with the provenance of
  every process that wrote the run) and, with ``--snapshot-every``, ``snapshots.npz``
  (see ``scheme_comparison/record.py``).

``post.py`` replots from these into ``post/``. ``DataCollector`` is no longer used, so
its VTX files, text traces and plots, and ``lmbda_prev_mean.txt``, are not written.

The folder rules are the CLIs'. A run refuses a folder that holds any of these files,
unless ``--overwrite`` (which deletes only them) or ``--restart`` is given. The default
folder may hold the results of a run from before these rules, so a plain re-run into it
is refused too. ``--restart`` continues from the checkpoint, and refuses one written
with other physics (:func:`physics`): it may change ``--t-end`` and the output options,
nothing else. If the checkpoint is at or past ``--t-end`` it takes no step.
"""

import argparse
import functools
import hashlib
import inspect
import json
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from mpi4py import MPI

import beat
import dolfinx
import numpy as np
import pulse
import ufl

import cardiac_geometries
import cardiac_geometries.geometry
from simcardemsx.backends import CrossbridgeSegregated, GeneratedActivation
from simcardemsx.checkpoint import Checkpointer
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import load_ode_modules
from simcardemsx.results import ARTIFACTS, CsvLog, ResultsWriter, write_resolved_settings

HERE = Path(__file__).resolve().parent
# scheme_comparison and demo_io sit next to this example, not in the installed package.
sys.path.insert(0, str(HERE.parent))
import demo_io  # noqa: E402
from scheme_comparison.record import Recorder  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_ODEFILE = Path("../odefiles/ToRORd_dynCl_endo_zetasplit.ode")
SCHEMES = ("monolithic", "segregated", "stabilized")
CROSSBRIDGE_MODELS = ("Land2017", "RDQ18", "RDQ20MF", "Lewalle2024")
T_END = 40.0  # ms

# The model: every value below is read by the run, and recorded by model_settings().
DT_EP = 0.05  # ms
SLAB_DX = 0.5  # Resolution of the slab mesh, mm
#: The slab, as cardiac_geometries generates it: size and resolution in mm, fibre angles.
SLAB: dict[str, Any] = {
    "lx": 2.0,
    "ly": 1.0,
    "lz": 0.5,
    "dx": SLAB_DX,
    "fiber_angle_endo": 0,
    "fiber_angle_epi": 0,
    "fiber_space": "DG_1",
}
MESH_UNIT = "mm"
QUAD_DEGREE = 4  # Degree of quadrature for the mechanics mesh
#: The spaces of ``results.bp``, which ``post.py`` rebuilds: the EP ODE space, P1 (the
#: transfer plan averages what crosses back onto P1 or DG0 only), holds ``v`` and
#: ``cai``; pulse's displacement space, given to the problem, holds ``u``; and the
#: backend's scalar quadrature space at ``QUAD_DEGREE`` holds the rest.
EP_ODE_ELEMENT = ("P", 1)
U_SPACE = "P_2"
#: The stimulus: beat's PDE stimulus with this amplitude, on the cells with no vertex
#: past this corner (mm), tagged STIM_MARKER. The EP model's own stimulus is switched
#: off. Its start and duration are beat's defaults.
STIM_AMPLITUDE = 50_000.0  # uA/cm**3
STIM_BOX_MM = (1.5, 1.5, 1.5)
STIM_MARKER = 1
#: The compressibility model, a class of ``pulse.compressibility``.
COMPRESSIBILITY = "Incompressible"
#: The boundary conditions, as dirichlet_bc in main() codes them.
BCS = {"X0": "u_x = 0", "Y0": "u_y = 0", "Z0": "u_z = 0", "base": "free"}

EP_RESULTS = ("v", "cai")
MECHANICS_RESULTS = ("u", "lmbda", "tension_kPa", "stiffness_kPa")
LOG_FIELDS = ("t_ms", "newton_iterations", "lmbda_mean", "Ta_mean_kPa")
#: Everything a run writes into its output folder, and so what --overwrite deletes.
OUTPUT_ARTIFACTS = (*ARTIFACTS, *demo_io.RECORDER_ARTIFACTS)
#: The arguments that do not define the physics: the run's length and its output.
NOT_PHYSICS = (
    "t_end",
    "output_dir",
    "snapshot_every",
    "save_every",
    "save_every_ep",
    "checkpoint_every",
    "restart",
    "overwrite",
)


def material_parameters() -> dict[str, Any]:
    """Holzapfel-Ogden's parameters as the run gives them to pulse: pulse's own
    transversely isotropic defaults, as ``pulse.Variable``s."""
    parameters: dict[str, Any] = dict(pulse.HolzapfelOgden.transversely_isotropic_parameters())
    return parameters


def model_settings() -> dict[str, Any]:
    """The model as the run reads it, as plain values: the time step of EP, the slab,
    EP's ODE space, conductivities, chi, C_m and stimulus, and the mechanics' spaces,
    compressibility, material (in base units: Pa) and boundary conditions.

    The stimulus's start and duration are read off beat's ``define_stimulus`` and the
    material off pulse, so a change to either library's defaults changes these too.
    """
    stimulus_defaults = inspect.signature(beat.stimulation.define_stimulus).parameters
    return {
        "dt_ep": DT_EP,
        "geometry": {**SLAB, "mesh_unit": MESH_UNIT},
        "ep": {
            "ode_element": list(EP_ODE_ELEMENT),
            "ode_parameters": {"i_Stim_Amplitude": 0.0},
            "theta": 1,
            "chi_per_mm": 140.0,
            "C_m_uF_per_mm2": 0.01,
            "conductivities_S_per_m": {
                "sigma_el": 0.62,
                "sigma_et": 0.24,
                "sigma_il": 0.17,
                "sigma_it": 0.019,
            },
            "stimulus": {
                "amplitude_uA_per_cm3": STIM_AMPLITUDE,
                "start_ms": stimulus_defaults["start"].default,
                "duration_ms": stimulus_defaults["duration"].default,
                "box_mm": list(STIM_BOX_MM),
                "marker": STIM_MARKER,
            },
        },
        "mechanics": {
            "quadrature_degree": QUAD_DEGREE,
            "u_space": U_SPACE,
            "compressibility": COMPRESSIBILITY,
            "material": {
                "model": "HolzapfelOgden",
                "units": "SI base units (Pa)",
                **{
                    name: float(value.to_base_units())
                    for name, value in material_parameters().items()
                },
            },
            "bcs": dict(BCS),
        },
    }


def disable_logger():
    for lib in ["numba", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def create_stim_tags(
    mesh,
    stim_marker=STIM_MARKER,
    stimx=STIM_BOX_MM[0],
    stimy=STIM_BOX_MM[1],
    stimz=STIM_BOX_MM[2],
):
    tol = 1e-6

    def S1_subdomain(x):
        return np.logical_and(
            np.logical_and(x[0] <= stimx + tol, x[1] <= stimy + tol),
            x[2] <= stimz + tol,
        )

    cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, S1_subdomain)

    stim_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        cells,
        np.full(len(cells), stim_marker, dtype=np.int32),
    )
    stim_tags.name = "stimulus"
    return stim_tags


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
    """The command line, with ``output_dir`` and ``save_every`` resolved from their
    defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--odefile",
        type=Path,
        default=DEFAULT_ODEFILE,
        help="gotranx .ode file with a 'mechanics' component (default: %(default)s). "
        "Output goes to output/<odefile stem>/ unless --output-dir is given.",
    )
    parser.add_argument(
        "--scheme",
        choices=SCHEMES,
        default="monolithic",
        help="Coupling scheme of the mechanics (default: %(default)s).",
    )
    parser.add_argument(
        "--crossbridge",
        choices=CROSSBRIDGE_MODELS,
        default=None,
        help="Run this crossbridge contraction model (Ca_i split) instead of the "
        "generated mechanics component. Needs the Ca_i split as --odefile, e.g. "
        "../odefiles/ToRORd_dynCl_endo_caisplit.ode. --scheme segregated is the naive "
        "scheme, stabilized the stabilized one; monolithic is not available "
        "(default: off).",
    )
    parser.add_argument(
        "--sl-ref",
        type=float,
        default=None,
        help="Reference sarcomere length SL_ref in um for --crossbridge; required for RDQ18 "
        "(default: the model's own SL0).",
    )
    parser.add_argument(
        "--dt-mech",
        type=float,
        default=DT_EP,
        help=f"Mechanics time step in ms, a whole multiple of the {DT_EP} ms EP step "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=T_END,
        help="End time in ms (default: %(default)s). Make it a whole multiple of --dt-mech: "
        "the run takes round(t_end / dt_mech) steps, so it otherwise stops at the nearest "
        "multiple, short of t_end or past it, and run.json records reached_t_end: false.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/<odefile stem>). It must hold no results "
        "of an earlier run, unless --overwrite or --restart is given.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=float,
        default=None,
        help="Save per-point snapshots of the stretch, tension and stiffness every this "
        "many ms, to snapshots.npz (default: off).",
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
    if args.output_dir is None:
        args.output_dir = Path("output") / args.odefile.stem
    if args.save_every is None:
        args.save_every = args.dt_mech
    return args


def settings(args: argparse.Namespace) -> dict[str, Any]:
    """What ``config.resolved.toml`` holds: every argument, with ``odefile`` as its
    path (as given), file name and sha256, and :func:`model_settings`.

    TOML has no null, so an argument that is ``None`` is not in the file. Raises
    ``FileNotFoundError`` if the ``.ode`` file does not exist.
    """
    arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    odefile = Path(args.odefile)
    arguments["odefile"] = {
        "path": str(odefile),
        "name": odefile.name,
        "sha256": hashlib.sha256(odefile.read_bytes()).hexdigest(),
    }
    return {**arguments, **model_settings()}


def physics(args: argparse.Namespace) -> dict[str, Any]:
    """What a restart must share with the checkpointed run: :func:`settings` without the
    run length and the output options (:data:`NOT_PHYSICS`), and without the ``.ode``
    file's path, so that the same file at another path still restarts. Solver options
    are not in it, as in the CLIs.

    Raises ``FileNotFoundError`` if the ``.ode`` file does not exist.
    """
    result = settings(args)
    for key in NOT_PHYSICS:
        del result[key]
    del result["odefile"]["path"]
    return result


def mesh_dir(dx: float) -> Path:
    """Where the slab mesh of resolution ``dx`` is generated on first use and read after:
    beside this file, so that ``post.py`` finds it from any working directory."""
    return HERE / "meshes" / f"slab_dx{dx}"


def build_geometry(run_settings: Mapping[str, Any]) -> cardiac_geometries.geometry.Geometry:
    """The slab ``run_settings["geometry"]`` describes, read from :func:`mesh_dir`
    (generated there first if it is missing), with the stimulus region as its cell tags
    (``cfun``) and the mechanics quadrature degree.

    Raises ``ValueError`` if the mesh found there is of another slab.
    """
    comm = MPI.COMM_WORLD
    slab = {key: run_settings["geometry"][key] for key in SLAB}
    directory = mesh_dir(slab["dx"])
    # One directory per resolution: the mesh is only generated when its directory is
    # missing, and is checked against the slab asked for when it is not.
    if not directory.is_dir():
        cardiac_geometries.mesh.slab(
            outdir=directory,
            lx=slab["lx"],
            ly=slab["ly"],
            lz=slab["lz"],
            dx=slab["dx"],
            create_fibers=True,
            fiber_angle_endo=slab["fiber_angle_endo"],
            fiber_angle_epi=slab["fiber_angle_epi"],
            fiber_space=slab["fiber_space"],
            comm=comm,
            use_dolfinx=True,
        )
    info = json.loads((directory / "info.json").read_text())
    # cardiac_geometries records the size as Lx, Ly and Lz.
    cached = {key: info[{"lx": "Lx", "ly": "Ly", "lz": "Lz"}.get(key, key)] for key in SLAB}
    if cached != slab:
        raise ValueError(
            f"The mesh in {directory} is of the slab {cached}, not {slab}: delete the "
            "directory to generate it again",
        )

    geo = cardiac_geometries.geometry.Geometry.from_file(comm=comm, path=directory / "geometry.bp")
    stimulus = run_settings["ep"]["stimulus"]
    stimx, stimy, stimz = stimulus["box_mm"]
    geo.cfun = create_stim_tags(
        geo.mesh,
        stim_marker=stimulus["marker"],
        stimx=stimx,
        stimy=stimy,
        stimz=stimz,
    )
    geo.quadrature_degree = run_settings["mechanics"]["quadrature_degree"]
    return geo


def main(argv: list[str] | None = None):
    start_total = time.perf_counter()
    args = parse_args(argv)
    if args.crossbridge is not None and args.scheme == "monolithic":
        raise SystemExit(
            "--scheme monolithic is not available with --crossbridge: the crossbridge models "
            "are NumPy and cannot be stepped inside Newton. Use --scheme stabilized or "
            "segregated.",
        )
    outdir: Path = args.output_dir
    odefile: Path = args.odefile
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

    comm = MPI.COMM_WORLD
    run_settings = settings(args)
    # Every run's, a restart's too: the latest run's settings win, as in the CLIs.
    if comm.rank == 0:
        write_resolved_settings(outdir / "config.resolved.toml", run_settings)

    logging.basicConfig(level=logging.DEBUG)
    disable_logger()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.DEBUG)

    # ---------------------------------------------------------
    # 1. Pre-processing: Generate ODE Code
    # ---------------------------------------------------------
    out_dir = Path("generated_odes") / odefile.stem

    logger.info(f"Generating ODE modules from {odefile}")
    modules = load_ode_modules(odefile, out_dir)
    ep_module = modules.ep

    # ---------------------------------------------------------
    # 2. Setup Meshes & Geometries
    # ---------------------------------------------------------
    geo = build_geometry(run_settings)
    stim_tags = geo.cfun
    assert stim_tags is not None
    mech_geo = geo
    ep_geo = geo
    mesh = mech_geo.mesh
    ep_mesh = ep_geo.mesh

    # P1: values going back to EP are averaged onto the EP ODE space, which the
    # transfer plan supports for P1 and DG0 only.
    ep_settings = run_settings["ep"]
    mechanics_settings = run_settings["mechanics"]
    quadrature_degree = mechanics_settings["quadrature_degree"]
    ep_ode_space = dolfinx.fem.functionspace(ep_mesh, tuple(ep_settings["ode_element"]))

    # ---------------------------------------------------------
    # 3. Setup EP Solver (fenicsx-beat)
    # ---------------------------------------------------------
    # Every value is read from the run's settings, which config.resolved.toml records.
    mesh_unit = run_settings["geometry"]["mesh_unit"]
    chi = ep_settings["chi_per_mm"] * beat.units.ureg("mm**-1")
    C_m = ep_settings["C_m_uF_per_mm2"] * beat.units.ureg("uF/mm**2")
    sigma = ep_settings["conductivities_S_per_m"]

    M = beat.conductivities.define_conductivity_tensor(
        chi=chi,
        f0=ep_geo.f0,
        g_il=sigma["sigma_il"] * beat.units.ureg("S/m"),
        g_it=sigma["sigma_it"] * beat.units.ureg("S/m"),
        g_el=sigma["sigma_el"] * beat.units.ureg("S/m"),
        g_et=sigma["sigma_et"] * beat.units.ureg("S/m"),
    )

    time_ep = dolfinx.fem.Constant(ep_mesh, 0.0)

    stimulus = ep_settings["stimulus"]
    I_s = beat.stimulation.define_stimulus(
        mesh=ep_mesh,
        chi=chi,
        time=time_ep,
        subdomain_data=stim_tags,
        marker=stimulus["marker"],
        mesh_unit=mesh_unit,
        amplitude=stimulus["amplitude_uA_per_cm3"] * beat.units.ureg("uA/cm**3"),
        start=stimulus["start_ms"],
        duration=stimulus["duration_ms"],
    )

    pde = beat.MonodomainModel(
        time=time_ep,
        mesh=ep_mesh,
        M=M,
        I_s=I_s,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        dx=ep_geo.dx,
    )

    v_ode = dolfinx.fem.Function(ep_ode_space)
    num_points_ep = v_ode.x.array.size

    # One column per point, for states, parameters and missing variables alike: the
    # transfer plan writes lambda and the values EP needs back into these arrays in
    # place, and they differ between points.
    y_ep = np.tile(ep_module.init_state_values()[:, None], (1, num_points_ep))
    p_ep = np.tile(
        ep_module.init_parameter_values(**ep_settings["ode_parameters"])[:, None],
        (1, num_points_ep),
    )
    ep_missing = getattr(ep_module, "missing", {})

    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=ep_module.generalized_rush_larsen,
        init_states=y_ep,
        parameters=p_ep,
        num_states=len(y_ep),
        v_index=ep_module.state_index("v"),
        missing_variables=np.zeros((len(ep_missing), num_points_ep)) if ep_missing else None,
        num_missing_variables=len(ep_missing),
    )

    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=ep_settings["theta"])

    # ---------------------------------------------------------
    # 4. Setup Mechanics Solver (fenicsx-pulse)
    # ---------------------------------------------------------
    material = pulse.HolzapfelOgden(f0=mech_geo.f0, s0=mech_geo.s0, **material_parameters())
    comp_model = getattr(pulse.compressibility, mechanics_settings["compressibility"])()

    # The contraction model, stepped inside Newton. Its states live on a quadrature
    # space of the same degree as the mechanics form (the controller checks this).
    backend: GeneratedActivation | CrossbridgeSegregated
    if args.crossbridge is not None:
        backend = CrossbridgeSegregated(
            mech_geo.f0,
            mesh,
            args.crossbridge,
            quadrature_degree=quadrature_degree,
            SL_ref=args.sl_ref,
            stabilized=args.scheme == "stabilized",
        )
    else:
        backend = GeneratedActivation(
            modules.mechanics,
            mesh,
            mech_geo.f0,
            quadrature_degree=quadrature_degree,
            scheme=args.scheme,
        )

    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=comp_model,
    )

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.bcs.DirichletBC]:
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0
        # The slab is always written with facet markers; ffun is optional only in general.
        ffun = mech_geo.ffun
        assert ffun is not None

        x0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(0), V0),
            ffun.dim,
            ffun.find(mech_geo.markers["X0"][0]),
        )
        y0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(1), V0),
            ffun.dim,
            ffun.find(mech_geo.markers["Y0"][0]),
        )
        z0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(2), V0),
            ffun.dim,
            ffun.find(mech_geo.markers["Z0"][0]),
        )

        return [
            dolfinx.fem.dirichletbc(zero, x0_dofs, V.sub(0)),
            dolfinx.fem.dirichletbc(zero, y0_dofs, V.sub(1)),
            dolfinx.fem.dirichletbc(zero, z0_dofs, V.sub(2)),
        ]

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))

    geometry = pulse.Geometry.from_cardiac_geometries(
        mech_geo,
        metadata={"quadrature_degree": quadrature_degree},
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    # Absolute tolerance, tightened from pulse's default 1e-6: at resting calcium the
    # first residual is already ~1e-9-1e-8 and stalls at round-off, which a pure
    # relative tolerance cannot converge (the line search then reports failure
    # although the state is converged).
    petsc_options["snes_atol"] = 1e-9

    # Important: BaseBC must be free since we manually constrain X, Y, Z boundaries
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={
            "base_bc": pulse.problem.BaseBC.free,
            "petsc_options": petsc_options,
            "u_space": mechanics_settings["u_space"],
        },
    )

    # ---------------------------------------------------------
    # 5. Initialize the coupling controller and the run's output
    # ---------------------------------------------------------
    controller = SimulationController(
        mechanics=problem,
        ep_solver=ep_solver,
        backend=backend,
        ode_modules=modules,
        dt_mech=dt_mech,
        dt_ep=run_settings["dt_ep"],
    )

    a = material_parameters()["a"]
    recorder = Recorder(
        backend,
        outdir,
        snapshot_every_ms=args.snapshot_every,
        run_info={
            "geometry": "slab",
            "split": odefile.stem.rsplit("_", 1)[-1],
            "backend": "generated"
            if args.crossbridge is None
            else f"crossbridge:{args.crossbridge}",
            "scheme": args.scheme,
            "dt_mech_ms": dt_mech,
            "t_end_ms": args.t_end,
            "regime": {
                # Uniaxial small-strain stiffness 3a of Holzapfel-Ogden's isotropic term.
                "Kp_kPa": float(3 * a.to_base_units() / 1e3),
                "eta_Pa_s": 0.0,
                "rho_kg_m3": 0.0,
                "h_m": SLAB_DX * 1e-3,
            },
        },
    )
    # The recorder's rows and snapshots are restored with the rest of the run.
    checkpointer = Checkpointer(controller, outdir, physics=run_physics, extra=[recorder])
    results = ResultsWriter(outdir)
    log = CsvLog(outdir / "log.csv", LOG_FIELDS)
    newton_its = 0

    # Timing baseline: wall time spent in the EP ODE step, the EP PDE step and the
    # mechanics solve, accumulated over the run. Replacing the methods on these objects
    # is deliberate: beat's splitting solver and the controller call them through
    # these attributes, which is what mypy's method-assign objects to.
    timings = {"ep_ode_s": 0.0, "ep_pde_s": 0.0, "mech_s": 0.0}
    ep_solver.ode.step = accumulate_time(ep_solver.ode.step, timings, "ep_ode_s")  # type: ignore[method-assign]
    ep_solver.pde.step = accumulate_time(ep_solver.pde.step, timings, "ep_pde_s")  # type: ignore[method-assign]
    problem.solve = accumulate_time(problem.solve, timings, "mech_s")  # type: ignore[method-assign]

    # results.bp's fields. EP's are P1 copies of rows of the ODE's state array, refreshed
    # before each save; the others are the problem's and the backend's own Functions.
    ep_fields = {name: dolfinx.fem.Function(ep_ode_space, name=name) for name in EP_RESULTS}
    ep_rows = {name: ep_module.state_index(name) for name in EP_RESULTS}
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

    # Mesh means at the mechanics form's quadrature points, for log.csv.
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * geometry.dx)),
        op=MPI.SUM,
    )
    lmbda_integral = dolfinx.fem.form(backend.lmbda_prev * geometry.dx)
    tension_integral = dolfinx.fem.form(backend.tension_kPa * geometry.dx)

    def mesh_mean(integral: dolfinx.fem.Form) -> float:
        return comm.allreduce(dolfinx.fem.assemble_scalar(integral), op=MPI.SUM) / volume

    def log_step(t: float, newton_iterations: int) -> None:
        log.append(
            {
                "t_ms": t,
                "newton_iterations": newton_iterations,
                "lmbda_mean": mesh_mean(lmbda_integral),
                "Ta_mean_kPa": mesh_mean(tension_integral),
            },
        )

    def write_initial() -> None:
        save_ep(controller.t)
        results.write(controller.t, mechanics_fields)
        log_step(controller.t, 0)

    # A refused or failed restore leaves the folder as it was: nothing below it, neither
    # the end checkpoint nor run.json, is reached.
    demo_io.start_or_resume(
        checkpointer,
        log,
        results,
        restart=args.restart,
        result_names=[*EP_RESULTS, *MECHANICS_RESULTS],
        write_initial=write_initial,
    )

    # ---------------------------------------------------------
    # 6. Define Callbacks & Run Simulation
    # ---------------------------------------------------------
    # The controller counts steps from 1. A step that fails is rolled back, but the EP
    # saves of its micro-steps stay in results.bp: a restart from the step's start
    # recomputes them bit for bit, so it does not save them again.
    def on_ep_step(current_t, ep_step_idx):
        if ep_step_idx % save_ep_every == 0:
            save_ep(current_t)

    def on_mech_step(current_t, mech_step_idx, newton_iters):
        nonlocal newton_its
        newton_its += newton_iters
        recorder.step(current_t, newton_iters)
        log_step(current_t, newton_iters)
        if mech_step_idx % save_every == 0:
            results.write(current_t, mechanics_fields)

    # The run is round(t_end / dt_mech) steps from t = 0, so a restart at or past it
    # takes none.
    remaining_steps = demo_io.steps_to_take(controller, args.t_end)

    # --- THE MAIN LOOP ---
    start_loop = time.perf_counter()
    timings["setup_s"] = start_loop - start_total
    failure: str | None = None
    t_fail: float | None = None
    try:
        for _ in range(remaining_steps):
            # The controller does all the interpolation, sub-stepping, and solving!
            controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)
            if controller.mech_step_idx % checkpoint_every == 0:
                checkpointer.write()
        timings["loop_s"] = time.perf_counter() - start_loop
    except BaseException as e:
        # BaseException: an interrupt is recorded too. A step that raised before its
        # mech_callback was rolled back (t_failed is set); anything raised after that
        # follows an accepted step, whose output may be half written.
        failure, t_fail = demo_io.failure_at(controller, e)
        raise
    finally:
        timings.setdefault("loop_s", time.perf_counter() - start_loop)
        timings["total_s"] = time.perf_counter() - start_total
        timings["newton_its"] = newton_its
        logger.info(f"Timings: {timings}")

        def write_timings() -> None:
            if comm.rank == 0:
                (outdir / "timings.json").write_text(json.dumps(timings, indent=4))

        # The end checkpoint and timings.json first, each guarded, and run.json (the mark
        # of a finished run, for scheme_comparison/run.py) last.
        demo_io.finish(
            recorder,
            checkpointer,
            [("timings.json", write_timings)],
            failure=failure,
            t_fail_ms=t_fail,
            timings=timings,
            here=HERE,
            restart=args.restart,
        )


if __name__ == "__main__":
    main()
