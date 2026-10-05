"""Replot a slab run from its output folder: ``python post.py --output-dir <folder>``.

Reads the folder's ``config.resolved.toml``, ``results.bp`` and ``log.csv``, rebuilds
only the geometry and the spaces the run wrote from (``main.build_geometry``, not the
coupled problem), and writes ``post/``, replacing it whole:

- ``fields.bp``: VTX on P1 of ``u`` (interpolated from the run's displacement space),
  ``v``, and ``lmbda`` and ``Ta`` (the backend's quadrature values, averaged onto P1),
  at every time any of them was saved. A field not saved at a time keeps its last
  saved value.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

from simcardemsx.results import CsvLog, read_resolved_settings, read_results

HERE = Path(__file__).resolve().parent
# The example's main sits next to this file, and demo_io beside its directory; importing
# them runs nothing.
sys.path.insert(0, str(HERE.parent))
from demo_io import (  # noqa: E402
    Panel,
    cell_containing,
    figure,
    plots_available,
    write_p1_fields,
)
from demo_io import result_functions as demo_result_functions  # noqa: E402

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


def result_functions(settings: Mapping[str, Any]) -> dict[str, dolfinx.fem.Function]:
    """One Function per ``results.bp`` name, on the space the run wrote it from, as the
    run's resolved ``settings`` describe it (:func:`demo_io.result_functions`). All on
    one mesh, the slab ``settings["geometry"]`` describes, from
    ``main.build_geometry(settings)``.
    """
    return demo_result_functions(slab.build_geometry(settings).mesh, settings, slab.EP_RESULTS)


def write_fields_and_traces(
    post_dir: Path,
    functions: Mapping[str, dolfinx.fem.Function],
    saved: Mapping[str, Mapping[float, np.ndarray]],
    quadrature_degree: int,
) -> dict[str, np.ndarray]:
    """Write ``fields.bp`` and ``traces.csv`` into ``post_dir``; return the traces as
    columns, by :data:`TRACE_FIELDS`.

    ``functions`` are :func:`result_functions`'s, and ``saved`` their values by name
    and time, as :func:`~simcardemsx.results.read_results` returns them. The volume
    means are taken at ``quadrature_degree``, the mechanics form's.
    """
    v, lmbda, tension = (functions[name] for name in ("v", "lmbda", "tension_kPa"))
    mesh = v.function_space.mesh

    # The volume means use the mechanics form's measure, as log.csv's do.
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * dx)).real
    integrals = {name: dolfinx.fem.form(f * dx) for name, f in (("v", v), ("lmbda", lmbda))}
    integrals["Ta"] = dolfinx.fem.form(tension * dx)

    point = np.array([POINT], dtype=np.float64)
    cell = cell_containing(mesh, point)

    def at_point(f: dolfinx.fem.Function) -> float:
        return float(np.ravel(f.eval(point, np.array([cell], dtype=np.int32)))[0])

    def mean(name: str) -> float:
        return dolfinx.fem.assemble_scalar(integrals[name]).real / volume

    rows: list[dict[str, float]] = []

    def trace(t: float, p1: Mapping[str, dolfinx.fem.Function]) -> None:
        row = dict.fromkeys(TRACE_FIELDS, np.nan)
        row["t_ms"] = t
        if t in saved["v"]:
            row["v_point_mV"], row["v_mean_mV"] = at_point(v), mean("v")
        if t in saved["lmbda"]:
            row["lmbda_point"], row["lmbda_mean"] = at_point(p1["lmbda"]), mean("lmbda")
        if t in saved["tension_kPa"]:
            row["Ta_point_kPa"], row["Ta_mean_kPa"] = at_point(p1["Ta"]), mean("Ta")
        rows.append(row)

    write_p1_fields(
        post_dir / "fields.bp",
        functions,
        saved,
        {"u": "u", "v": "v", "lmbda": "lmbda", "Ta": "tension_kPa"},
        trace,
    )

    traces = CsvLog(post_dir / "traces.csv", TRACE_FIELDS)
    traces.start()
    for row in rows:
        traces.append(row)
    return {name: np.array([row[name] for row in rows]) for name in TRACE_FIELDS}


def write_plots(
    post_dir: Path,
    traces: Mapping[str, np.ndarray],
    log: Sequence[Mapping[str, float]],
) -> None:
    """Write :data:`PLOTS` into ``post_dir``, or warn and skip them without matplotlib."""
    if not plots_available():
        return
    where = f"at ({', '.join(f'{x:g}' for x in POINT)})"
    t = traces["t_ms"]
    figure(
        post_dir / "ep_volume_averages.png",
        [Panel("v, volume mean (mV)", t, traces["v_mean_mV"])],
    )
    figure(post_dir / "ep_point_traces.png", [Panel(f"v {where} (mV)", t, traces["v_point_mV"])])
    figure(
        post_dir / "mech_volume_averages.png",
        [
            Panel("Ta, volume mean (kPa)", t, traces["Ta_mean_kPa"]),
            Panel("λ, volume mean", t, traces["lmbda_mean"]),
        ],
    )
    figure(
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
    figure(
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
    traces = write_fields_and_traces(
        post_dir,
        functions,
        saved,
        settings["mechanics"]["quadrature_degree"],
    )
    write_plots(post_dir, traces, log)


if __name__ == "__main__":
    main()
