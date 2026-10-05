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
from simcardemsx.provenance import provenance
from simcardemsx.results import (
    ARTIFACTS,
    CsvLog,
    ResultsWriter,
    prepare_output,
    stride,
    write_resolved_settings,
)

HERE = Path(__file__).resolve().parent
# scheme_comparison sits next to this example, not in the installed package.
sys.path.insert(0, str(HERE.parent))
from scheme_comparison.record import (  # noqa: E402
    Recorder,
    failure_of,
    finish_after_artifacts,
)

logger = logging.getLogger(__name__)
QUAD_DEGREE = 4  # Degree of quadrature for the mechanics mesh
DEFAULT_ODEFILE = Path("../odefiles/ToRORd_dynCl_endo_zetasplit.ode")
SLAB_DX = 0.5  # Resolution of the slab mesh
#: Where the slab mesh is generated on first use and read after: beside this file, one
#: directory per resolution, so that ``post.py`` finds it from any working directory.
MESH_DIR = HERE / "meshes" / f"slab_dx{SLAB_DX}"
STIM_MARKER = 1
SCHEMES = ("monolithic", "segregated", "stabilized")
CROSSBRIDGE_MODELS = ("Land2017", "RDQ18", "RDQ20MF", "Lewalle2024")
DT_EP = 0.05  # ms
T_END = 40.0  # ms

#: The spaces of ``results.bp``, which ``post.py`` rebuilds: the EP ODE space, P1 (the
#: transfer plan averages what crosses back onto P1 or DG0 only), holds ``v`` and
#: ``cai``; pulse's displacement space, given to the problem, holds ``u``; and the
#: backend's scalar quadrature space at ``QUAD_DEGREE`` holds the rest.
EP_ODE_ELEMENT = ("P", 1)
U_SPACE = "P_2"
EP_RESULTS = ("v", "cai")
MECHANICS_RESULTS = ("u", "lmbda", "tension_kPa", "stiffness_kPa")
LOG_FIELDS = ("t_ms", "newton_iterations", "lmbda_mean", "Ta_mean_kPa")
#: Everything a run writes into its output folder, and so what --overwrite deletes.
OUTPUT_ARTIFACTS = (*ARTIFACTS, "steps.csv", "snapshots.npz", "restart_recorder_*.npz")
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
#: ``default_config()["sim"]``'s entries that are not physics either: the run length,
#: the output, and the ``.ode`` file's path (``physics`` holds its resolved path and
#: hash in its place).
NOT_PHYSICS_SIM = ("modelfile", "outdir", "sim_dur", "save_frequency_ep", "save_frequency_mech")


def default_config():
    return {
        "ep": {
            "conductivities": {
                "sigma_el": 0.62,
                "sigma_et": 0.24,
                "sigma_il": 0.17,
                "sigma_it": 0.019,
            },
            "stimulus": {
                "amplitude": 50000.0,
                "duration": 2,
                "start": 0.0,
                "xmax": 1.5,
                "xmin": 0.0,
                "ymax": 1.5,
                "ymin": 0.0,
                "zmax": 1.5,
                "zmin": 0.0,
            },
            "chi": 140.0,
            "C_m": 0.01,
        },
        "mechanics": {
            "material": {
                "a": 2.28,
                "a_f": 1.686,
                "a_fs": 0.0,
                "a_s": 0.0,
                "b": 9.726,
                "b_f": 15.779,
                "b_fs": 0.0,
                "b_s": 0.0,
            },
            "bcs": [
                {"V": "u_x", "expression": 0, "marker": 1, "param_numbers": 0, "type": "Dirichlet"},
                {"V": "u_y", "expression": 0, "marker": 3, "param_numbers": 0, "type": "Dirichlet"},
                {"V": "u_z", "expression": 0, "marker": 5, "param_numbers": 0, "type": "Dirichlet"},
            ],
        },
        "sim": {
            "N": 1,
            "dt": 0.05,
            "mech_mesh": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz",
            "markerfile": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz_surface_ffun",
            "modelfile": "../odefiles/ToRORd_dynCl_endo_zetasplit.ode",
            "outdir": "output",
            "sim_dur": 40,
            "save_frequency_ep": 20,
            "save_frequency_mech": 1,
        },
        "output": {
            "all_ep": ["v"],
            "all_mech": ["Ta", "lambda"],
            "point_ep": [
                {"name": "v", "x": 0, "y": 0, "z": 0},
            ],
            "point_mech": [
                {"name": "Ta", "x": 0, "y": 0, "z": 0},
                {"name": "lambda", "x": 0, "y": 0, "z": 0},
            ],
        },
    }


def disable_logger():
    for lib in ["numba", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def create_stim_tags(mesh, stim_marker=STIM_MARKER, stimx=1.5, stimy=1.5, stimz=1.5):
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
    """Every argument (paths as ``str``), and :func:`default_config` with its ``sim``
    entries set from them: what ``config.resolved.toml`` holds.

    TOML has no null, so an argument that is ``None`` is not in the file.
    """
    config = default_config()
    config["sim"].update(
        modelfile=str(args.odefile),
        outdir=str(args.output_dir),
        dt=DT_EP,
        N=round(args.dt_mech / DT_EP),
        sim_dur=args.t_end,
    )
    arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    return {**arguments, **config}


def physics(args: argparse.Namespace) -> dict[str, Any]:
    """What a restart must share with the checkpointed run: :func:`settings` without the
    run length and the output (:data:`NOT_PHYSICS`, ``default_config``'s ``output`` and
    :data:`NOT_PHYSICS_SIM`), and with ``odefile`` as its resolved path and its sha256.

    Raises ``FileNotFoundError`` if the ``.ode`` file does not exist.
    """
    result = settings(args)
    for key in NOT_PHYSICS:
        del result[key]
    del result["output"]
    for key in NOT_PHYSICS_SIM:
        del result["sim"][key]
    odefile = Path(args.odefile)
    result["odefile"] = {
        "path": str(odefile.resolve()),
        "sha256": hashlib.sha256(odefile.read_bytes()).hexdigest(),
    }
    return result


def build_geometry(run_settings: Mapping[str, Any]) -> cardiac_geometries.geometry.Geometry:
    """The slab, read from :data:`MESH_DIR` (generated there first if it is missing),
    with the stimulus region, the box up to ``run_settings["ep"]["stimulus"]``'s
    ``xmax``, ``ymax`` and ``zmax``, as its cell tags (``cfun``, marker
    :data:`STIM_MARKER`), and the mechanics quadrature degree."""
    comm = MPI.COMM_WORLD
    # One directory per resolution: the mesh is only generated when its directory is
    # missing, so a single directory would silently reuse a slab of another resolution.
    if not MESH_DIR.is_dir():
        cardiac_geometries.mesh.slab(
            outdir=MESH_DIR,
            lx=2.0,
            ly=1.0,
            lz=0.5,
            dx=SLAB_DX,
            create_fibers=True,
            fiber_angle_endo=0,
            fiber_angle_epi=0,
            fiber_space="DG_1",
            comm=comm,
            use_dolfinx=True,
        )

    geo = cardiac_geometries.geometry.Geometry.from_file(comm=comm, path=MESH_DIR / "geometry.bp")
    stimulus = run_settings["ep"]["stimulus"]
    geo.cfun = create_stim_tags(
        geo.mesh,
        stim_marker=STIM_MARKER,
        stimx=stimulus["xmax"],
        stimy=stimulus["ymax"],
        stimz=stimulus["zmax"],
    )
    geo.quadrature_degree = QUAD_DEGREE
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
    def steps(option: str, every: float, dt: float) -> int:
        try:
            return stride(every, dt)
        except ValueError as error:
            raise ValueError(f"{option}: {error}") from error

    try:
        steps("--dt-mech", dt_mech, DT_EP)
        save_ep_every = steps("--save-every-ep", args.save_every_ep, DT_EP)
        save_every = steps("--save-every", args.save_every, dt_mech)
        checkpoint_every = steps("--checkpoint-every", args.checkpoint_every, dt_mech)
        run_physics = physics(args)
        prepare_output(
            outdir,
            restart=args.restart,
            overwrite=args.overwrite,
            physics=run_physics,
            artifacts=OUTPUT_ARTIFACTS,
        )
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

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
    ep_ode_space = dolfinx.fem.functionspace(ep_mesh, EP_ODE_ELEMENT)

    # ---------------------------------------------------------
    # 3. Setup EP Solver (fenicsx-beat)
    # ---------------------------------------------------------
    config = run_settings
    mesh_unit = "mm"
    chi = config["ep"]["chi"] * beat.units.ureg("mm**-1")
    C_m = config["ep"]["C_m"] * beat.units.ureg("uF/mm**2")

    M = beat.conductivities.define_conductivity_tensor(
        chi=chi,
        f0=ep_geo.f0,
        g_il=config["ep"]["conductivities"]["sigma_il"] * beat.units.ureg("S/m"),
        g_it=config["ep"]["conductivities"]["sigma_it"] * beat.units.ureg("S/m"),
        g_el=config["ep"]["conductivities"]["sigma_el"] * beat.units.ureg("S/m"),
        g_et=config["ep"]["conductivities"]["sigma_et"] * beat.units.ureg("S/m"),
    )

    time_ep = dolfinx.fem.Constant(ep_mesh, 0.0)

    I_s = beat.stimulation.define_stimulus(
        mesh=ep_mesh,
        chi=chi,
        time=time_ep,
        subdomain_data=stim_tags,
        marker=STIM_MARKER,
        mesh_unit=mesh_unit,
        amplitude=50_000.0 * beat.units.ureg("uA/cm**3"),
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
        ep_module.init_parameter_values(i_Stim_Amplitude=0.0)[:, None],
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

    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

    # ---------------------------------------------------------
    # 4. Setup Mechanics Solver (fenicsx-pulse)
    # ---------------------------------------------------------
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=mech_geo.f0, s0=mech_geo.s0, **material_params)
    comp_model = pulse.compressibility.Incompressible()

    # The contraction model, stepped inside Newton. Its states live on a quadrature
    # space of the same degree as the mechanics form (the controller checks this).
    backend: GeneratedActivation | CrossbridgeSegregated
    if args.crossbridge is not None:
        backend = CrossbridgeSegregated(
            mech_geo.f0,
            mesh,
            args.crossbridge,
            quadrature_degree=QUAD_DEGREE,
            SL_ref=args.sl_ref,
            stabilized=args.scheme == "stabilized",
        )
    else:
        backend = GeneratedActivation(
            modules.mechanics,
            mesh,
            mech_geo.f0,
            quadrature_degree=QUAD_DEGREE,
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
        metadata={"quadrature_degree": QUAD_DEGREE},
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
            "u_space": U_SPACE,
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
        dt_ep=DT_EP,
    )

    a = pulse.HolzapfelOgden.transversely_isotropic_parameters()["a"]
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

    if args.restart:
        # A refused or failed restore leaves the folder as it was: nothing below it,
        # neither the end checkpoint nor run.json, is reached.
        t_restart = checkpointer.restore()
        log.resume(t_restart)
        results.resume([*EP_RESULTS, *MECHANICS_RESULTS])
        logger.info(f"Restarted from the checkpoint at t = {t_restart} ms")
    else:
        log.start()
        save_ep(controller.t)
        results.write(controller.t, mechanics_fields)
        log_step(controller.t, 0)

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
    remaining_steps = round(args.t_end / dt_mech) - controller.mech_step_idx
    if remaining_steps <= 0:
        logger.info(f"The checkpoint, at t = {controller.t} ms, is at or past t_end: no step")

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
        # BaseException: an interrupt is recorded too. A step that raised was rolled
        # back, leaving the controller's t at its start; t_failed is its end.
        failure = failure_of(e)
        t_fail = controller.t_failed if controller.t_failed is not None else controller.t
        logger.exception(f"The coupled step ending at t = {t_fail} ms failed")
        raise
    finally:
        timings.setdefault("loop_s", time.perf_counter() - start_loop)
        timings["total_s"] = time.perf_counter() - start_total
        timings["newton_its"] = newton_its
        logger.info(f"Timings: {timings}")

        def write_checkpoint() -> None:
            # At the end of every run, finished or failed (a failed step was rolled back
            # to its start). At a time already checkpointed, only restart.json is new.
            if backend.step_pending:
                logger.warning(f"No checkpoint at t = {controller.t} ms: a step is pending")
                return
            checkpointer.write()

        def write_timings() -> None:
            if comm.rank == 0:
                (outdir / "timings.json").write_text(json.dumps(timings, indent=4))

        # The checkpoint and timings.json first, each guarded, and run.json (the mark of
        # a finished run, for scheme_comparison/run.py) last.
        finish_after_artifacts(
            recorder,
            [("checkpoint", write_checkpoint), ("timings.json", write_timings)],
            failure=failure,
            t_fail_ms=t_fail,
            timings=timings,
            extra={
                "provenance": provenance(HERE),
                "history": checkpointer.history,
                "restart": args.restart,
            },
        )


if __name__ == "__main__":
    main()
