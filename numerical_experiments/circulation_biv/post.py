"""Replot a BiV run from its output folder: ``python post.py --output-dir <folder>``.

Reads the folder's ``config.resolved.toml``, ``log.csv``, ``results.bp`` and, if it is
there, ``run.json``. Rebuilds only the geometry and the spaces the run wrote from, not
the coupled problem: the mesh comes from the cache under
``meshes/ukb_mean_ed_clipped/`` (``main.load_geometry``: rotated, in metres), deformed
to the unloaded reference by the cached prestress, as the run deforms it. A missing
cache is reported, not regenerated: generating it needs ``ukb-atlas`` and
``fenicsx-ldrb`` (see ``main.py``). Writes ``post/``, replacing it whole:

- ``summary.json`` and ``pv_loops.png``: what the run writes at its end, by the same
  functions (:func:`summarise`, :func:`plot`), from ``log.csv``. ``failed_at_ms`` is
  ``run.json``'s ``t_fail_ms``, and ``tref_scale`` the run's ``--tref``.
- ``fields.bp``: VTX on P1 of ``u`` (interpolated from the run's displacement space),
  ``v``, and ``lmbda`` and ``Ta`` (the backend's quadrature values of ``lmbda`` and
  ``tension_kPa``, averaged onto P1), at every time any of them was saved. A field not
  saved at a time keeps its last saved value.
- ``stats.csv``: ``t_ms``, then for λ and ``Ta`` (``tension_kPa``) the mean over the
  myocardium (at the mechanics form's quadrature points, as ``steps.csv``'s) and the
  minimum and maximum over those points, one row per time both were saved at.

:func:`summarise` and :func:`plot` live here, and ``main.py`` imports them, so this
module imports ``main`` only inside the functions that use it. Without matplotlib
``pv_loops.png`` is skipped, with a warning, and the rest is written. Serial only.
"""

import argparse
import csv
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl

from simcardemsx.results import CsvLog, read_resolved_settings, read_results

HERE = Path(__file__).resolve().parent
# demo_io sits beside this example's directory, not in the installed package.
sys.path.insert(0, str(HERE.parent))
from demo_io import plots_available, write_p1_fields  # noqa: E402

#: The artery each ventricle ejects into: its outflow valve is open when the
#: ventricle's pressure exceeds this one's.
OUTFLOW = {"LV": "p_AR_SYS", "RV": "p_AR_PUL"}
STATS_FIELDS = (
    "t_ms",
    "lmbda_mean",
    "lmbda_min",
    "lmbda_max",
    "Ta_mean_kPa",
    "Ta_min_kPa",
    "Ta_max_kPa",
)


# ---------------------------------------------------------------------------
# The run's summary and plot, from log.csv's columns
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
    for c, outflow in OUTFLOW.items():
        V, p = columns[f"V_{c}_mL"], columns[f"p_{c}_mmHg"]
        EDV, ESV = float(V.max()), float(V.min())
        intervals = valve_open_intervals(t, V, p, columns[f"circuit_{outflow}"])
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
# The run's fields, from results.bp
# ---------------------------------------------------------------------------


def reference_mesh(settings: Mapping[str, Any]) -> dolfinx.mesh.Mesh:
    """The run's reference configuration: the cached mesh as ``main.load_geometry``
    gives it (rotated, in metres), deformed to the unloaded reference by the cached
    prestress (``main.read_prestress``), as ``settings`` describe the run.

    Raises ``FileNotFoundError``, naming the cache, if the mesh or the prestress is not
    in it.
    """
    from circulation_biv import main as biv

    if not (biv.GEODIR / "geometry.bp").exists():
        raise FileNotFoundError(
            f"The BiV mesh cache {biv.GEODIR} holds no geometry.bp. post.py does not "
            "generate it: run main.py once (generating the mesh needs ukb-atlas and "
            "fenicsx-ldrb).",
        )
    comm = MPI.COMM_WORLD
    _, geometry = biv.load_geometry(biv.GEODIR, comm, settings)
    geometry.deform(biv.read_prestress(geometry.mesh, biv.GEODIR / "cache", settings))
    return geometry.mesh


def result_functions(
    settings: Mapping[str, Any],
    mesh: dolfinx.mesh.Mesh,
) -> dict[str, dolfinx.fem.Function]:
    """One Function per ``results.bp`` name, on the space the run wrote it from, on
    ``mesh`` (:func:`reference_mesh`), as the run's resolved ``settings`` describe it.

    ``v`` and ``cai`` are on the EP ODE space (``settings["ep"]["ode_element"]``); ``u``
    on pulse's displacement space (``settings["mechanics"]["u_space"]``, built as pulse
    builds it); ``lmbda``, ``tension_kPa`` and ``stiffness_kPa`` on the backend's scalar
    quadrature space at ``settings["mechanics"]["quadrature_degree"]``.
    """
    from circulation_biv import main as biv

    cell = mesh.basix_cell()
    ep_space = dolfinx.fem.functionspace(mesh, tuple(settings["ep"]["ode_element"]))
    family, degree = settings["mechanics"]["u_space"].split("_")
    u_space = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element(family, cell, int(degree), shape=(mesh.topology.dim,)),
    )
    quadrature_space = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.quadrature_element(
            cell,
            value_shape=(),
            degree=settings["mechanics"]["quadrature_degree"],
        ),
    )
    spaces = {
        **dict.fromkeys(biv.EP_RESULTS, ep_space),
        "u": u_space,
        **dict.fromkeys(("lmbda", "tension_kPa", "stiffness_kPa"), quadrature_space),
    }
    return {name: dolfinx.fem.Function(space, name=name) for name, space in spaces.items()}


def write_fields_and_stats(
    post_dir: Path,
    functions: Mapping[str, dolfinx.fem.Function],
    saved: Mapping[str, Mapping[float, np.ndarray]],
    quadrature_degree: int,
) -> None:
    """Write ``fields.bp`` and ``stats.csv`` into ``post_dir``.

    ``functions`` are :func:`result_functions`'s, and ``saved`` their values by name and
    time, as :func:`~simcardemsx.results.read_results` returns them. The means are
    taken at ``quadrature_degree``, the mechanics form's.
    """
    lmbda, tension = functions["lmbda"], functions["tension_kPa"]
    mesh = lmbda.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * dx)).real
    integrals = {"lmbda": dolfinx.fem.form(lmbda * dx), "Ta": dolfinx.fem.form(tension * dx)}
    rows: list[dict[str, float]] = []

    def stats(t: float, p1: Mapping[str, dolfinx.fem.Function]) -> None:
        if t not in saved["lmbda"] or t not in saved["tension_kPa"]:
            return
        row = {"t_ms": t}
        for key, f, unit in (("lmbda", lmbda, ""), ("Ta", tension, "_kPa")):
            row[f"{key}_mean{unit}"] = dolfinx.fem.assemble_scalar(integrals[key]).real / volume
            row[f"{key}_min{unit}"] = float(f.x.array.min())
            row[f"{key}_max{unit}"] = float(f.x.array.max())
        rows.append(row)

    write_p1_fields(
        post_dir / "fields.bp",
        functions,
        saved,
        {"u": "u", "v": "v", "lmbda": "lmbda", "Ta": "tension_kPa"},
        stats,
    )
    log = CsvLog(post_dir / "stats.csv", STATS_FIELDS)
    log.start()
    for row in rows:
        log.append(row)


def read_log(path: Path) -> list[dict[str, float]]:
    """``log.csv``'s rows as floats, by its own header (whose circuit columns are the
    circuit's states)."""
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    return CsvLog(path, header).read()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The run's output folder (main.py's --outdir).",
    )
    args = parser.parse_args(argv)
    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("post.py runs in serial only")
    folder: Path = args.output_dir

    settings = read_resolved_settings(folder / "config.resolved.toml")
    try:
        mesh = reference_mesh(settings)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    log = read_log(folder / "log.csv")
    run = folder / "run.json"
    failed_at = json.loads(run.read_text()).get("t_fail_ms") if run.exists() else None
    functions = result_functions(settings, mesh)
    saved = read_results(folder, functions)

    post_dir = folder / "post"
    shutil.rmtree(post_dir, ignore_errors=True)
    post_dir.mkdir()
    columns = {key: np.array([row[key] for row in log]) for key in log[0]}
    summary = summarise(columns, failed_at)
    summary["tref_scale"] = settings["tref"]
    (post_dir / "summary.json").write_text(json.dumps(summary, indent=4))
    write_fields_and_stats(
        post_dir,
        functions,
        saved,
        settings["mechanics"]["quadrature_degree"],
    )
    if plots_available():
        plot(columns, post_dir / "pv_loops.png")


if __name__ == "__main__":
    main()
