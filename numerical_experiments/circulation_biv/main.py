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
steps 0.05 ms.

Output, in ``--outdir``:

- ``log.csv``: one row at ``t = 0`` and one after each mechanics step. ``V_*_mL`` are
  the 3D cavity volumes ``V(u)`` and ``p_*_mmHg`` the cavity pressures; the
  ``circuit_*`` columns are every circuit state in the circuit's own units (``V_*``
  mL, ``p_*`` mmHg, ``Q_*`` mL/s); ``total_volume_mL`` is the total blood volume with
  the 3D cavity volumes in place of the circuit's ``V_LV``/``V_RV`` and
  ``conservation_drift`` its relative change since ``t = 0``; ``Ta_mean_kPa`` is the
  volume mean of the backend's ``active_tension``; ``newton_iterations`` is 0 at
  ``t = 0``, where nothing is solved.
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
  counts with ``failed_at_ms``, the controller's time when the loop raised, if it did.
- ``timings.json``: wall time in the EP ODE step, the EP PDE step and the mechanics
  solve (as in ``strong_coupling_zetasplit``), plus the loop and the whole run.

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
import csv
import functools
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from mpi4py import MPI

import beat
import dolfinx
import io4dolfinx
import numpy as np
import pulse
import ufl
from circulation import base, regazzoni2020
from circulation.units import mmHg_to_kPa
from pulse.circulation import ChamberCoupling, GotranxCirculation, mL, mmHg

import cardiac_geometries
import cardiac_geometries.geometry
from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import CirculationClock
from simcardemsx.ode_model import load_ode_modules

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ODEFILE = HERE.parent / "odefiles" / "ToRORd_dynCl_endo_zetasplit.ode"
GEODIR = HERE / "meshes" / "ukb_mean_ed_clipped"

CHAMBERS = ("LV", "RV")
#: The artery each ventricle ejects into: its outflow valve is open when the
#: ventricle's pressure exceeds this one's.
OUTFLOW = {"LV": "p_AR_SYS", "RV": "p_AR_PUL"}

CHAR_LENGTH = 10.0  # mm; the demo's: the atlas is smooth, so a coarse mesh suffices
#: The demo's quadrature degree, for the geometry, its ``Quadrature_6`` fibres and the
#: backend's states alike: FFCx evaluates a whole integral at a quadrature-element
#: coefficient's own degree, so all three must agree.
QUAD_DEGREE = 6

PERIOD = 1000.0  # ms: one beat, the circuit's RR = 1 / HR and ToR-ORd's pacing period
DT_MECH = 2.0  # ms, the demo's
DT_EP = 0.05  # ms

#: Absolute Newton tolerance of the coupled problem, tightened from pulse's default
#: 1e-6 as in every coupled problem of this package. The cavity constraint rows are
#: ``V_state - V(u)`` in m^3, so 1e-6 would allow volume errors of up to a millilitre.
SNES_ATOL = 1e-9


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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=PERIOD,
        help="End time in ms (default: %(default)s, one beat). "
        f"Rounded to a whole number of {DT_MECH} ms mechanics steps.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=HERE / "output",
        help="Output directory (default: %(default)s).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def generate_mesh(geodir: Path, comm: MPI.Intracomm) -> None:
    """The demo's mesh: the atlas mean shape at end diastole, clipped at the valve plane
    so the mesh has a single ``BASE``, with LDRB fibres on ``Quadrature_6``."""
    import ldrb  # type: ignore[import-untyped,import-not-found]

    logger.info(f"Generating the UKB mesh in {geodir}")
    geo = cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=-1,
        std=0,
        case="ED",
        char_length_max=CHAR_LENGTH,
        char_length_min=CHAR_LENGTH,
        clipped=True,
    )
    geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        alpha_endo_rv=90,
        alpha_epi_rv=-25,
        beta_endo_lv=-20,
        beta_epi_lv=20,
        beta_endo_rv=0,
        beta_epi_rv=20,
        fiber_space=f"Quadrature_{QUAD_DEGREE}",
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


def load_geometry(geodir: Path, comm: MPI.Intracomm):
    """The cached mesh, rotated so the base normal is x, in metres.

    Rotated after loading, as in the demo: the folder also holds the unrotated
    ``.msh``, which is what ``from_folder`` gives back, and the sliding-base condition
    constrains ``u_x`` only.
    """
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")
    geo.mesh.geometry.x[:] *= 1e-3  # mm -> m: the chamber coupling assumes metres
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": QUAD_DEGREE},
    )
    up = base_normal(geometry)
    if abs(up[0]) < 0.99:
        raise RuntimeError(
            f"the base normal is {up.round(3)}, not the x axis the sliding-base "
            "condition assumes -- the rotation did not take effect",
        )
    return geo, geometry


# ---------------------------------------------------------------------------
# Mechanics: the demo's material, compressibility and boundary conditions
# ---------------------------------------------------------------------------


def cardiac_model(f0, s0, active: pulse.active_model.ActiveModel) -> pulse.CardiacModel:
    """Holzapfel-Ogden (transversely isotropic), compressible, viscous.

    The viscous term does nothing without a strain rate, so the static prestressing
    and inflation are unaffected by it.
    """
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    return pulse.CardiacModel(
        material=material,
        active=active,
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.viscoelasticity.Viscous(),
    )


def robin_bcs(geometry: pulse.HeartGeometry) -> tuple[pulse.RobinBC, ...]:
    """Springs on the epicardium and base, with the dynamic arm's damping on both."""

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
        spring("EPI", 1.0e5),
        spring("BASE", 1.0e6),
        spring("EPI", 5.0e3, damping=True),
        spring("BASE", 5.0e3, damping=True),
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


def circuit_parameters() -> dict[str, float]:
    """Regazzoni's parameters at 1 Hz. The heart rate goes in before flattening:
    ``flat_ode_parameters`` derives ``RR`` and the chambers' activation offsets from it,
    which overriding the flat ``HR`` would not."""
    nested = base.remove_units(regazzoni2020.Regazzoni2020.default_parameters())
    return regazzoni2020.flat_ode_parameters(nested | {"HR": 1000.0 / PERIOD})


def circuit_operating_point(
    EDV: dict[str, float],
    cachedir: Path,
    comm: MPI.Intracomm,
) -> tuple[dict[str, float], dict[str, float]]:
    """The circuit alone, from the mesh's end-diastolic volumes, run to a limit cycle.

    Returns its state (circuit units) and its end-diastolic pressures (kPa).
    """
    state_file = cachedir / "circ_state.json"
    if comm.rank == 0 and not state_file.exists():
        logger.info("Running the circuit alone to a limit cycle...")
        standalone = regazzoni2020.Regazzoni2020(
            parameters={"HR": 1000.0 / PERIOD},
            add_units=False,
            outdir=cachedir / "regazzoni_standalone",
        )
        history = standalone.solve(
            num_beats=10,
            initial_state={f"V_{c}": EDV[c] / mL for c in CHAMBERS},
            dt=0.001,
        )
        state = dict(zip(standalone.state_names(), map(float, standalone.state)))
        cache = {"state": state} | {f"p_{c}_ED": float(history[f"p_{c}"][-1]) for c in CHAMBERS}
        state_file.write_text(json.dumps(cache, indent=4))
    comm.barrier()
    cached = json.loads(state_file.read_text())
    p_ED = {c: mmHg_to_kPa(cached[f"p_{c}_ED"]) for c in CHAMBERS}
    return {k: float(v) for k, v in cached["state"].items()}, p_ED


def prestress(
    geometry: pulse.HeartGeometry,
    geo,
    p_ED: dict[str, float],
    cachedir: Path,
    comm: MPI.Intracomm,
) -> dolfinx.fem.Function:
    """The displacement from the unloaded to the loaded (as meshed) configuration."""
    fname = cachedir / "prestress_biv.bp"
    if not fname.exists():
        logger.info("Prestressing to recover the unloaded reference configuration...")
        traction = {
            c: pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa") for c in CHAMBERS
        }
        problem = pulse.unloading.PrestressProblem(
            geometry=geometry,
            model=cardiac_model(geo.f0, geo.s0, pulse.active_model.Passive()),
            bcs=pulse.BoundaryConditions(
                robin=robin_bcs(geometry),
                dirichlet=(sliding_base(geometry),),
                neumann=tuple(
                    pulse.NeumannBC(traction=traction[c], marker=geometry.markers[c][0])
                    for c in CHAMBERS
                ),
            ),
            parameters={"u_space": "P_2", "mesh_unit": "m"},
            targets=[
                pulse.unloading.TargetPressure(traction=traction[c], target=p_ED[c], name=c)
                for c in CHAMBERS
            ],
            ramp_steps=20,
        )
        u_pre = problem.unload()
        io4dolfinx.write_function_on_input_mesh(fname, u_pre, time=0.0, name="u_pre")
    comm.barrier()
    u_pre = dolfinx.fem.Function(dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,))))
    io4dolfinx.read_function(fname, u_pre, time=0.0, name="u_pre")
    return u_pre


def inflate(
    geometry: pulse.HeartGeometry,
    f0,
    s0,
    unloaded: dict[str, float],
    EDV: dict[str, float],
) -> pulse.StaticProblem:
    """Volume-controlled inflation, passive, from the unloaded volumes to ``EDV``."""
    volume = {
        c: dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(unloaded[c]))
        for c in CHAMBERS
    }
    inflation = pulse.StaticProblem(
        model=cardiac_model(f0, s0, pulse.active_model.Passive()),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(
            robin=robin_bcs(geometry),
            dirichlet=(sliding_base(geometry),),
        ),
        cavities=[pulse.problem.Cavity(marker=c, volume=volume[c]) for c in CHAMBERS],
        parameters={"mesh_unit": "m"},
    )
    inflation.solve()
    for fraction in np.linspace(0.0, 1.0, 20)[1:]:
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
    mesh: dolfinx.mesh.Mesh,
    f0,
    dx: ufl.Measure,
) -> beat.MonodomainSplittingSolver:
    """beat's monodomain splitting solver on ``mesh`` (in metres), with a P1 ODE space.

    No stimulus current in the PDE: ToR-ORd's own cellular stimulus fires at every
    point at t = 0, with its period set to the beat. States, parameters and missing
    variables have one column per point: the transfer plan writes ``lmbda`` and the
    backend's outputs back into these arrays in place, and they differ between points.
    """
    chi = 140.0 * beat.units.ureg("mm**-1")
    C_m = 0.01 * beat.units.ureg("uF/mm**2")
    M = beat.conductivities.define_conductivity_tensor(
        chi=chi,
        f0=f0,
        g_il=0.17 * beat.units.ureg("S/m"),
        g_it=0.019 * beat.units.ureg("S/m"),
        g_el=0.62 * beat.units.ureg("S/m"),
        g_et=0.24 * beat.units.ureg("S/m"),
    )
    pde = beat.MonodomainModel(
        time=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        mesh=mesh,
        M=M,
        C_m=C_m.to("uF/m**2").magnitude,
        dx=dx,
    )
    v_ode = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)))
    num_points = v_ode.x.array.size
    states = np.tile(ep_module.init_state_values()[:, None], (1, num_points))
    parameters = np.tile(
        ep_module.init_parameter_values(i_Stim_Period=PERIOD)[:, None],
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
    return beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def valve_open_intervals(
    t: np.ndarray,
    V: np.ndarray,
    p: np.ndarray,
    p_out: np.ndarray,
) -> list[dict[str, Any]]:
    """The runs of rows where ``p > p_out``, each with the volume ejected over it.

    The valve opens during the step ending at the first row of a run, so the volume
    ejected is measured from the row before it to the last row of the run.
    """
    intervals = []
    is_open = p > p_out
    i = 0
    while i < len(t):
        if not is_open[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and is_open[j + 1]:
            j += 1
        intervals.append(
            {
                "first_open_ms": float(t[i]),
                "last_open_ms": float(t[j]),
                "open_at_end": bool(j == len(t) - 1),
                "ejected_mL": float(V[max(i - 1, 0)] - V[j]),
            },
        )
        i = j + 1
    return intervals


def summarise(columns: dict[str, np.ndarray], failed_at: float | None) -> dict[str, Any]:
    t = columns["t_ms"]
    summary: dict[str, Any] = {}
    for c in CHAMBERS:
        V, p = columns[f"V_{c}_mL"], columns[f"p_{c}_mmHg"]
        EDV, ESV = float(V.max()), float(V.min())
        intervals = valve_open_intervals(t, V, p, columns[f"circuit_{OUTFLOW[c]}"])
        summary[c] = {
            "EDV_mL": EDV,
            "ESV_mL": ESV,
            # Not a stroke volume and an ejection fraction: the beat is not periodic,
            # and V also changes while the outflow valve is shut. See ejected_mL.
            "V_range_mL": EDV - ESV,
            "V_range_fraction": (EDV - ESV) / EDV,
            "peak_p_mmHg": float(p.max()),
            "t_peak_p_ms": float(t[np.argmax(p)]),
            "outflow_valve_open": intervals,
            "ejects": any(interval["ejected_mL"] > 0 for interval in intervals),
        }
    iterations = columns["newton_iterations"][1:]
    summary["max_conservation_drift"] = float(np.max(columns["conservation_drift"]))
    summary["newton_iterations"] = {
        "steps": int(iterations.size),
        "min": int(iterations.min()) if iterations.size else None,
        "mean": float(iterations.mean()) if iterations.size else None,
        "max": int(iterations.max()) if iterations.size else None,
        "failed_at_ms": failed_at,
    }
    summary["peak_Ta_mean_kPa"] = float(columns["Ta_mean_kPa"].max())
    return summary


def plot(columns: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig = plt.figure(layout="constrained", figsize=(11, 8))
    grid = fig.add_gridspec(3, 2)
    ax_loop = fig.add_subplot(grid[:, 0])
    ax_p = fig.add_subplot(grid[0, 1])
    ax_v = fig.add_subplot(grid[1, 1])
    ax_ta = fig.add_subplot(grid[2, 1], sharex=ax_p)
    t = columns["t_ms"]
    for c, colour in (("LV", "crimson"), ("RV", "steelblue")):
        V, p = columns[f"V_{c}_mL"], columns[f"p_{c}_mmHg"]
        ax_loop.plot(V, p, color=colour, label=c, linewidth=1.1)
        ax_p.plot(t, p, color=colour, label=f"p_{c}")
        p_out = columns[f"circuit_{OUTFLOW[c]}"]
        ax_p.plot(t, p_out, color=colour, linestyle="--", label=OUTFLOW[c])
        ax_v.plot(t, V, color=colour, label=c)
    ax_loop.set_xlabel("V [mL]")
    ax_loop.set_ylabel("p [mmHg]")
    ax_loop.set_title("Pressure-volume loops")
    ax_loop.legend()
    ax_p.set_ylabel("p [mmHg]")
    ax_p.legend(fontsize="x-small", ncol=2)
    ax_v.set_ylabel("V [mL]")
    ax_v.legend(fontsize="x-small")
    ax_ta.plot(t, columns["Ta_mean_kPa"], color="0.3")
    ax_ta.set_ylabel("mean Ta [kPa]")
    ax_ta.set_xlabel("t [ms]")
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
        raise RuntimeError("circulation_biv runs in serial only")
    outdir = args.outdir
    cachedir = GEODIR / "cache"
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
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
        generate_mesh(GEODIR, comm)
    comm.barrier()
    geo, geometry = load_geometry(GEODIR, comm)

    EDV = {c: comm.allreduce(geometry.volume(c), op=MPI.SUM) for c in CHAMBERS}
    logger.info(
        f"Mesh end-diastolic volumes: LV {EDV['LV'] / mL:.1f} mL, RV {EDV['RV'] / mL:.1f} mL",
    )

    circ_state, p_ED = circuit_operating_point(EDV, cachedir, comm)
    logger.info(
        f"Circuit end-diastolic pressures: LV {p_ED['LV']:.2f} kPa, RV {p_ED['RV']:.2f} kPa",
    )

    u_pre = prestress(geometry, geo, p_ED, cachedir, comm)
    geometry.deform(u_pre)
    f0 = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
    s0 = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")
    unloaded = {c: comm.allreduce(geometry.volume(c), op=MPI.SUM) for c in CHAMBERS}
    logger.info(
        f"Unloaded volumes: LV {unloaded['LV'] / mL:.1f} mL, RV {unloaded['RV'] / mL:.1f} mL",
    )

    inflation = inflate(geometry, f0, s0, unloaded, EDV)
    p_inflated = [float(p.x.array[0]) for p in inflation.cavity_pressures]
    # The circuit starts from the volumes the inflation actually reached, so the cavity
    # constraint holds at t = 0 and the conservation check starts from a consistent state.
    V0 = {c: comm.allreduce(geometry.volume(c, u=inflation.u), op=MPI.SUM) / mL for c in CHAMBERS}
    logger.info(
        f"Inflated to LV {V0['LV']:.2f} mL at {p_inflated[0] / mmHg:.2f} mmHg, "
        f"RV {V0['RV']:.2f} mL at {p_inflated[1] / mmHg:.2f} mmHg",
    )

    # ---------------------------------------------------------
    # 3. The coupled mechanics problem
    # ---------------------------------------------------------
    backend = GeneratedActivation(
        modules.mechanics,
        geometry.mesh,
        f0,
        quadrature_degree=QUAD_DEGREE,
    )
    circuit = GotranxCirculation(
        ode_file=regazzoni2020.ODE_FILE,
        parameters=circuit_parameters(),
        drop_components=("timing", "LV", "RV"),
    )
    beat_phase = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    problem = pulse.DynamicProblem(
        model=cardiac_model(f0, s0, backend),
        geometry=geometry,
        bcs=pulse.BoundaryConditions(
            robin=robin_bcs(geometry),
            dirichlet=(sliding_base(geometry),),
        ),
        # No volume: each chamber coupling replaces it with the circuit's volume state.
        cavities=[pulse.problem.Cavity(marker=c, volume=None) for c in CHAMBERS],
        circulation=circuit,
        chambers=[ChamberCoupling(c, f"V_{c}", f"p_{c}") for c in CHAMBERS],
        circulation_missing={"beat_phase": beat_phase},
        parameters={
            "mesh_unit": "m",
            "circulation_scheme": "backward_euler",
            "rho": pulse.Variable(1e3, "kg/m^3"),
            "dt": pulse.Variable(DT_MECH * 1e-3, "s"),
            "petsc_options": petsc_options,
        },
    )

    # Start from the inflated configuration, at rest, and from the circuit state that
    # matches it.
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
    ep_solver = make_ep_solver(modules.ep, geometry.mesh, f0, geometry.dx)
    clock = CirculationClock(problem, time_unit="s", beat_phase=beat_phase, period=PERIOD)
    controller = SimulationController(clock, ep_solver, backend, modules, DT_MECH, DT_EP)

    # The tissue starts at rest at the inflated stretch, not at lambda = 1: without this,
    # the first step would see a stretch rate of (lambda(u_0) - 1) / dt, and the zeta
    # states and EP's troponin would start from the unloaded stretch.
    F0 = ufl.grad(problem.u) + ufl.Identity(3)
    lmbda0 = dolfinx.fem.Expression(
        ufl.sqrt(ufl.inner(F0.T * F0 * f0, f0)),
        backend.space.element.interpolation_points,
    )
    backend.lmbda_prev.interpolate(lmbda0)
    backend.outputs["lmbda"].interpolate(lmbda0)
    controller.plan.backward()

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
    rows: list[dict[str, float]] = []

    def record(t: float, newton_iterations: int) -> None:
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
        rows.append(row)

    def on_mech_step(t: float, step: int, newton_iterations: int) -> None:
        record(t, newton_iterations)
        row = rows[-1]
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
    record(0.0, 0)
    num_steps = round(args.t_end / DT_MECH)
    failed_at = None
    start_loop = time.perf_counter()
    try:
        for _ in range(num_steps):
            controller.step(mech_callback=on_mech_step)
    except Exception:
        failed_at = controller.t
        logger.exception(f"The coupled step ending at t = {failed_at} ms failed")
        raise
    finally:
        timings["loop_s"] = time.perf_counter() - start_loop
        timings["total_s"] = time.perf_counter() - start_total
        columns = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        summary = summarise(columns, failed_at)
        if comm.rank == 0:
            with open(outdir / "log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (outdir / "summary.json").write_text(json.dumps(summary, indent=4))
            (outdir / "timings.json").write_text(json.dumps(timings, indent=4))
            plot(columns, outdir / "pv_loops.png")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
        logger.info(f"Timings: {timings}")


if __name__ == "__main__":
    main()
