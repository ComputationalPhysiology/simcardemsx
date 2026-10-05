"""Replot a slab run from its output folder: ``python post.py --output-dir <folder>``.

Reads the folder's ``config.resolved.toml``, ``results.bp`` and ``log.csv``, rebuilds
only the geometry and the spaces the run wrote from (``main.build_geometry``, not the
coupled problem), and writes ``post/``, replacing it whole:

- ``fields.bp``: VTX on P1 of ``u`` (interpolated from the run's P2), ``v``, and ``lmbda``
  and ``Ta`` (the backend's quadrature values, averaged onto P1), at every time any of
  them was saved. A field not saved at a time keeps its last saved value.
- ``traces.csv``: ``t_ms``, then ``v``, ``lmbda`` and ``Ta`` at :data:`POINT` (from the P1
  fields) and as volume means (at the mechanics form's quadrature points, as
  ``log.csv``'s), one row per saved time; a quantity not saved at a time is ``nan``.
- ``ep_volume_averages.png``, ``ep_point_traces.png``, ``mech_volume_averages.png`` and
  ``mech_point_traces.png``: ``DataCollector``'s four plots, drawn from ``traces.csv``.
- ``log.png``: the Newton iterations and the mean stretch of every mechanics step, from
  ``log.csv``.

Without matplotlib the plots are skipped, with a warning, and the rest is written.
Serial only.
"""

import argparse
import shutil
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl

from simcardemsx.averaging import make_averager
from simcardemsx.results import CsvLog, read_resolved_settings, read_results

HERE = Path(__file__).resolve().parent
# The example's main sits next to this file; importing it runs nothing.
sys.path.insert(0, str(HERE.parent))
from strong_coupling_zetasplit import main as slab  # noqa: E402

#: Where the traces are taken: ``DataCollector``'s point, a corner of the slab.
POINT = (0.0, 0.0, 0.0)
TRACE_FIELDS = (
    "t_ms",
    "v_point_mV",
    "v_mean_mV",
    "lmbda_point",
    "lmbda_mean",
    "Ta_point_kPa",
    "Ta_mean_kPa",
)
PLOTS = (
    "ep_volume_averages.png",
    "ep_point_traces.png",
    "mech_volume_averages.png",
    "mech_point_traces.png",
    "log.png",
)

# The plots' colours: one series per panel, so one hue.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def result_functions(settings: Mapping[str, Any]) -> dict[str, dolfinx.fem.Function]:
    """One Function per ``results.bp`` name, on the space the run wrote it from.

    ``v`` and ``cai`` are on the EP ODE space (``main.EP_ODE_ELEMENT``); ``u`` on pulse's
    displacement space (``main.U_SPACE``, built as pulse builds it); ``lmbda``,
    ``tension_kPa`` and ``stiffness_kPa`` on the backend's scalar quadrature space at
    ``main.QUAD_DEGREE`` with basix's default scheme, as both backends build it. All on
    one mesh, from ``main.build_geometry(settings)``.
    """
    mesh = slab.build_geometry(settings).mesh
    cell = mesh.basix_cell()
    ep_space = dolfinx.fem.functionspace(mesh, slab.EP_ODE_ELEMENT)
    family, degree = slab.U_SPACE.split("_")
    u_space = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element(family, cell, int(degree), shape=(mesh.topology.dim,)),
    )
    quadrature_space = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.quadrature_element(cell, value_shape=(), degree=slab.QUAD_DEGREE),
    )
    spaces = {
        **dict.fromkeys(slab.EP_RESULTS, ep_space),
        "u": u_space,
        **dict.fromkeys(("lmbda", "tension_kPa", "stiffness_kPa"), quadrature_space),
    }
    return {name: dolfinx.fem.Function(space, name=name) for name, space in spaces.items()}


def _cell_containing(mesh: dolfinx.mesh.Mesh, point: np.ndarray) -> int:
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, point)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, point).links(0)
    if len(cells) == 0:
        raise ValueError(f"The point {point[0].tolist()} is not in the mesh")
    return int(cells[0])


def write_fields_and_traces(
    post_dir: Path,
    functions: Mapping[str, dolfinx.fem.Function],
    saved: Mapping[str, Mapping[float, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Write ``fields.bp`` and ``traces.csv`` into ``post_dir``; return the traces as
    columns, by :data:`TRACE_FIELDS`.

    ``functions`` are :func:`result_functions`'s, and ``saved`` their values by name
    and time, as :func:`~simcardemsx.results.read_results` returns them.
    """
    v, lmbda, tension, u = (functions[name] for name in ("v", "lmbda", "tension_kPa", "u"))
    mesh = v.function_space.mesh
    P1 = dolfinx.fem.functionspace(mesh, ("P", 1))
    shown = {
        "u": dolfinx.fem.Function(
            dolfinx.fem.functionspace(mesh, ("P", 1, (mesh.topology.dim,))),
            name="u",
        ),
        "v": v,
        "lmbda": dolfinx.fem.Function(P1, name="lmbda"),
        "Ta": dolfinx.fem.Function(P1, name="Ta"),
    }
    average_lmbda = make_averager(lmbda, shown["lmbda"])
    average_tension = make_averager(tension, shown["Ta"])

    # The volume means use the mechanics form's measure, as log.csv's do.
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": slab.QUAD_DEGREE})
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * dx)).real
    integrals = {name: dolfinx.fem.form(f * dx) for name, f in (("v", v), ("lmbda", lmbda))}
    integrals["Ta"] = dolfinx.fem.form(tension * dx)

    point = np.array([POINT], dtype=np.float64)
    cell = _cell_containing(mesh, point)

    def at_point(f: dolfinx.fem.Function) -> float:
        return float(np.ravel(f.eval(point, np.array([cell], dtype=np.int32)))[0])

    def mean(name: str) -> float:
        return dolfinx.fem.assemble_scalar(integrals[name]).real / volume

    times = sorted({t for by_time in saved.values() for t in by_time})
    rows: list[dict[str, float]] = []
    writer = dolfinx.io.VTXWriter(mesh.comm, post_dir / "fields.bp", list(shown.values()))
    try:
        for t in times:
            for name, f in functions.items():
                if t in saved[name]:
                    f.x.array[:] = saved[name][t]
            shown["u"].interpolate(u)
            average_lmbda()
            average_tension()
            writer.write(t)

            row = dict.fromkeys(TRACE_FIELDS, np.nan)
            row["t_ms"] = t
            if t in saved["v"]:
                row["v_point_mV"], row["v_mean_mV"] = at_point(v), mean("v")
            if t in saved["lmbda"]:
                row["lmbda_point"], row["lmbda_mean"] = at_point(shown["lmbda"]), mean("lmbda")
            if t in saved["tension_kPa"]:
                row["Ta_point_kPa"], row["Ta_mean_kPa"] = at_point(shown["Ta"]), mean("Ta")
            rows.append(row)
    finally:
        writer.close()

    traces = CsvLog(post_dir / "traces.csv", TRACE_FIELDS)
    traces.start()
    for row in rows:
        traces.append(row)
    return {name: np.array([row[name] for row in rows]) for name in TRACE_FIELDS}


class Panel(NamedTuple):
    """One quantity over time: a line, or with ``counts`` a dot per value on an integer
    axis. ``nan`` values (times the quantity was not saved at) are left out."""

    title: str
    t: np.ndarray
    values: np.ndarray
    counts: bool = False


def _figure(path: Path, panels: Sequence[Panel]) -> None:
    """The panels stacked over a shared time axis, one series each."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(7.0, 0.6 + 2.2 * len(panels)), layout="constrained")
    axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    for ax, panel in zip(axes, panels):
        keep = np.isfinite(panel.values)
        t, values = panel.t[keep], panel.values[keep]
        if panel.counts:
            ax.plot(t, values, linestyle="none", marker="o", markersize=5, color=SERIES)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
        else:
            ax.plot(t, values, color=SERIES, linewidth=1.5)
            # Plain tick labels: an offset ("1e-5+9.999e-1") would sit on the title.
            ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_title(panel.title, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_SECONDARY, labelsize=8)
    axes[-1].set_xlabel("t (ms)", color=INK_SECONDARY)
    fig.savefig(path, dpi=150)


def write_plots(
    post_dir: Path,
    traces: Mapping[str, np.ndarray],
    log: Sequence[Mapping[str, float]],
) -> None:
    """Write :data:`PLOTS` into ``post_dir``, or warn and skip them without matplotlib."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        warnings.warn("matplotlib is not available: the plots are skipped", stacklevel=2)
        return
    where = f"at ({', '.join(f'{x:g}' for x in POINT)})"
    t = traces["t_ms"]
    _figure(
        post_dir / "ep_volume_averages.png",
        [Panel("v, volume mean (mV)", t, traces["v_mean_mV"])],
    )
    _figure(post_dir / "ep_point_traces.png", [Panel(f"v {where} (mV)", t, traces["v_point_mV"])])
    _figure(
        post_dir / "mech_volume_averages.png",
        [
            Panel("Ta, volume mean (kPa)", t, traces["Ta_mean_kPa"]),
            Panel("λ, volume mean", t, traces["lmbda_mean"]),
        ],
    )
    _figure(
        post_dir / "mech_point_traces.png",
        [
            Panel(f"Ta {where} (kPa)", t, traces["Ta_point_kPa"]),
            Panel(f"λ {where}", t, traces["lmbda_point"]),
        ],
    )

    def column(name: str) -> np.ndarray:
        return np.array([row[name] for row in log])

    # log.csv's row at t = 0 is the initial state, with no solve behind it.
    t_log, solved = column("t_ms"), column("t_ms") > 0
    _figure(
        post_dir / "log.png",
        [
            Panel(
                "Newton iterations per step",
                t_log[solved],
                column("newton_iterations")[solved],
                counts=True,
            ),
            Panel("λ, volume mean", t_log, column("lmbda_mean")),
        ],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The run's output folder (main.py's --output-dir).",
    )
    args = parser.parse_args(argv)
    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("post.py runs in serial only")
    folder: Path = args.output_dir

    settings = read_resolved_settings(folder / "config.resolved.toml")
    functions = result_functions(settings)
    saved = read_results(folder, functions)
    log = CsvLog(folder / "log.csv", slab.LOG_FIELDS).read()

    post_dir = folder / "post"
    shutil.rmtree(post_dir, ignore_errors=True)
    post_dir.mkdir()
    traces = write_fields_and_traces(post_dir, functions, saved)
    write_plots(post_dir, traces, log)


if __name__ == "__main__":
    main()
