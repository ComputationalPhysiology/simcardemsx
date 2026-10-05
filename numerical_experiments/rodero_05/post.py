"""Replot a rodero_05 run from its output folder: ``python post.py --output-dir <folder>``.

Reads the folder's ``config.resolved.toml``, ``log.csv``, ``results.bp`` and, if it is
there, ``run.json``. Rebuilds only the geometry, from physcardems' case
(``case.load_case``, at the run's resolved case directory and ``--tref``), and the
spaces the run wrote from: no problem and no coupled forms. Writes ``post/``,
replacing it whole:

- ``summary.json`` and ``pv_loops.png``: what the run writes at its end, by the same
  functions (:func:`summarise`, :func:`plot`), from ``log.csv`` alone. ``failure`` is
  ``run.json``'s (:func:`run_failure`: a run whose ``run.json`` still says ``running``
  did not finish), and the Land placement ``config.resolved.toml``'s.
- ``fields.bp``: VTX on P1 of ``u`` (interpolated from the run's displacement space),
  ``v``, and ``lmbda`` and ``Ta`` (the backend's quadrature values of ``lmbda`` and of
  the masked ``tension_kPa``, averaged onto P1), at every time any of them was saved.
  A field not saved at a time keeps its last saved value.
- ``active_stats.csv`` (:data:`ACTIVE_STATS_FIELDS`): physcardems' ``active_stats.csv``
  without its ``frac_h_zero`` column. At every time both ``lmbda`` and
  ``tension_kPa`` were saved: ``t_ms``; the 5th, 50th and 95th percentiles of ``Ta``
  over the myocardium's quadrature points (where the case's DG0 mask is 1); and, for
  each ToR-ORd cell type, the median ``Ta`` and λ over the myocardium's quadrature
  points of that type. The cell type at a quadrature point is the case's P1 cell type
  interpolated there and rounded, as physcardems rounds it on DG1. A cell type with no
  such point is ``nan``.
- ``ecg.csv`` (:data:`ECG_FIELDS`) and ``ecg.png``: the pseudo-ECG, one row per time
  ``v`` was saved at (``--save-every-ep``). ``beat.ECGRecovery(v=v, sigma_b=1.0,
  C_m=1.0, M=1.0)``: the membrane current is recovered as ``I_m = div(grad v)`` (an L2
  projection on P1) and the potential at each electrode is ``1/(4 pi) * integral of
  I_m / |x - x_e|``. With ``M`` the identity and ``sigma_b`` 1, that is physcardems'
  convention, G = I (``src/physcardems/ecg.py``: ``integral of grad v . r / |r|^3``),
  integrated by parts; the two agree to second order in the mesh size. It is in mV for
  unit conductivities, as physcardems' is: a scaled, not a physical, body-surface
  potential. The electrodes are the case's (``case.electrodes``, in metres, in the
  imaged frame, as physcardems uses them), and the leads ``beat.ecg.Leads12`` of their
  potentials: I, II, III, aVR, aVL, aVF, and V1..V6 against Wilson's central terminal
  (``Leads12.V1_``...), which are physcardems' ``standard_leads``.

:func:`summarise` and :func:`plot` live here, and ``main.py`` imports them, so this
module imports ``main`` only inside the functions that use it. Without matplotlib the
plots are skipped, with a warning, and the rest is written. Serial only.
"""

import argparse
import csv
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

import beat
import beat.ecg
import dolfinx
import numpy as np
from pulse.cycle import Phase

from simcardemsx.results import CsvLog, read_resolved_settings, read_results

HERE = Path(__file__).resolve().parent
# case.py sits next to this file, and demo_io beside its directory; importing them runs
# nothing.
sys.path.insert(0, str(HERE.parent))
from demo_io import (  # noqa: E402
    GRID,
    INK,
    INK_SECONDARY,
    SERIES,
    plots_available,
    result_functions,
    write_p1_fields,
)

from rodero_05.case import CELLTYPES, Case, electrodes, load_case  # noqa: E402

CHAMBERS = ("LV", "RV")

#: ``active_stats.csv``'s columns.
ACTIVE_STATS_FIELDS = (
    "t_ms",
    "Ta_p5_kPa",
    "Ta_p50_kPa",
    "Ta_p95_kPa",
    *(
        name
        for celltype in CELLTYPES.values()
        for name in (f"Ta_median_{celltype}_kPa", f"lmbda_median_{celltype}")
    ),
)
#: The pseudo-ECG's leads, by ``ecg.csv`` column, as ``beat.ecg.Leads12`` names them.
LEADS = {
    "I": "I",
    "II": "II",
    "III": "III",
    "aVR": "aVR",
    "aVL": "aVL",
    "aVF": "aVF",
    **{f"V{i}": f"V{i}_" for i in range(1, 7)},
}
ECG_FIELDS = ("t_ms", *LEADS)
#: ``ecg.png``'s layout, physcardems' (``ecg.py``, ``LEAD_LAYOUT``): one row per limb
#: lead, then its augmented lead and two precordial leads.
ECG_LAYOUT = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))


# ---------------------------------------------------------------------------
# The run's summary and plot, from log.csv's columns
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


def phase_switches(columns: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Every switch of a cavity's phase, in order: the rows whose ``next_phase_*`` (the
    phase after the step) is not their ``phase_*`` (the phase it was solved under)."""
    switches = []
    for i, t in enumerate(columns["t_ms"]):
        for c in CHAMBERS:
            during, after = int(columns[f"phase_{c}"][i]), int(columns[f"next_phase_{c}"][i])
            if after != during:
                switches.append(
                    {
                        "chamber": c,
                        "from": Phase(during).name,
                        "to": Phase(after).name,
                        "t_ms": float(t),
                    },
                )
    return switches


def summarise(
    columns: Mapping[str, np.ndarray],
    failure: str | None,
    land_placement: Mapping[str, list[str]],
    t_end_ms: float,
) -> dict[str, Any]:
    """The acceptance numbers of the run so far, from ``log.csv``'s columns alone;
    ``t_end_ms`` is where it is to end, in whole steps.

    Row 0 is the unloaded solve, and every later row a step. ``pulse.cycle`` retries a
    failed solve at most once, so a step's first solve (``first_*``) and last solve are
    all of its solves.
    """
    t = columns["t_ms"]
    summary: dict[str, Any] = {"t_end_ms": float(t[-1]), "failure": failure}
    five_phases = [p.name for p in Phase]
    for c in CHAMBERS:
        P = columns[f"P_{c}_kPa"]
        sequence = phase_sequence(columns[f"phase_{c}"])
        summary[c] = {
            "ejection": ejection_fraction(columns[f"phase_{c}"], columns[f"V_{c}_mL"]),
            "peak_P_kPa": float(P.max()),
            "peak_P_mmHg": float(P.max() * 1e3 / 133.322),
            "t_peak_P_ms": float(t[np.argmax(P)]),
            "phases": sequence,
            "five_phases_in_order": sequence[:5] == five_phases,
        }
    summary["phase_switches"] = phase_switches(columns)

    # Every step after the unloaded solve (row 0).
    final_reasons = [str(int(r)) for r in columns["snes_reason"][1:]]
    first_reasons = [str(int(r)) for r in columns["first_reason"][1:]]
    attempts = columns["solve_attempts"][1:]
    reasons: dict[str, int] = {}
    for reason in final_reasons:
        reasons[reason] = reasons.get(reason, 0) + 1
    iterations = columns["newton_iterations"][1:]
    summary["newton"] = {
        "steps": int(iterations.size),
        "unloaded_solve": {
            "iterations": int(columns["newton_iterations"][0]),
            "reason": int(columns["snes_reason"][0]),
            "linear_iterations": int(columns["linear_iterations"][0]),
            "converged": True,
        },
        "iterations_min": int(iterations.min()) if iterations.size else None,
        "iterations_mean": float(iterations.mean()) if iterations.size else None,
        "iterations_max": int(iterations.max()) if iterations.size else None,
        # 2 and 3: converged on the residual (absolute, relative); 4: on the step size.
        "final_reasons": reasons,
        "retries": int(np.sum(attempts - 1)),
        "steps_with_retry_ms": [float(s) for s, n in zip(t[1:], attempts) if n > 1],
        "all_attempts_reasons": sorted(set(final_reasons) | set(first_reasons)),
    }
    detF = columns["detF_min"]
    summary["detF_min"] = float(detF.min())
    summary["t_detF_min_ms"] = float(t[np.argmin(detF)])
    summary["Ta_min_kPa"] = float(columns["Ta_min_kPa"].min())
    summary["Ta_max_kPa"] = float(columns["Ta_max_kPa"].max())
    summary["lmbda_min"] = float(columns["lmbda_min"].min())
    summary["lmbda_max"] = float(columns["lmbda_max"].max())
    summary["criteria"] = {
        "reached_t_end": bool(np.isclose(t[-1], t_end_ms)),
        "newton_converged_every_step": failure is None,
        "detF_positive_every_step": bool(np.all(columns["n_detF_nonpositive"] == 0)),
        **{f"{c}_five_phases_in_order": summary[c]["five_phases_in_order"] for c in CHAMBERS},
    }
    summary["land_placement"] = dict(land_placement)
    return summary


def plot(columns: Mapping[str, np.ndarray], summary: Mapping[str, Any], path: Path) -> None:
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


def read_columns(path: Path) -> dict[str, np.ndarray]:
    """``log.csv``'s columns as float arrays, by its own header."""
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    rows = CsvLog(path, header).read()
    return {key: np.array([row[key] for row in rows]) for key in header}


# ---------------------------------------------------------------------------
# The run's fields, from results.bp
# ---------------------------------------------------------------------------


def _at_quadrature_points(
    source: dolfinx.fem.Function,
    space: dolfinx.fem.FunctionSpace,
) -> np.ndarray:
    """``source`` interpolated at the points of the quadrature ``space``."""
    target = dolfinx.fem.Function(space)
    target.interpolate(dolfinx.fem.Expression(source, space.element.interpolation_points))
    return target.x.array.copy()


class ActiveStats:
    """``active_stats.csv``'s rows, from ``lmbda`` and ``tension`` (``tension_kPa``) on
    the backend's quadrature space: where the myocardium is, and each point's cell type,
    are computed once."""

    def __init__(self, case: Case, space: dolfinx.fem.FunctionSpace):
        self.in_myocardium = _at_quadrature_points(case.myocardium_mask, space) > 0.5
        mesh = space.mesh
        celltype = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)))
        celltype.x.array[:] = case.celltype
        self.celltype = np.rint(_at_quadrature_points(celltype, space)).astype(int)

    def row(
        self,
        t: float,
        lmbda: dolfinx.fem.Function,
        tension: dolfinx.fem.Function,
    ) -> dict[str, float]:
        Ta = tension.x.array[self.in_myocardium]
        lam = lmbda.x.array[self.in_myocardium]
        celltype = self.celltype[self.in_myocardium]
        row = {"t_ms": t}
        row |= dict(zip(ACTIVE_STATS_FIELDS[1:4], map(float, np.percentile(Ta, [5, 50, 95]))))
        for c, name in CELLTYPES.items():
            selected = celltype == c
            if not selected.any():
                row[f"Ta_median_{name}_kPa"] = row[f"lmbda_median_{name}"] = np.nan
                continue
            row[f"Ta_median_{name}_kPa"] = float(np.median(Ta[selected]))
            row[f"lmbda_median_{name}"] = float(np.median(lam[selected]))
        return row


class PseudoECG:
    """The pseudo-ECG at the case's electrodes, from ``v`` (P1) as it is: see the module
    docstring. ``beat.ECGRecovery``'s forms, one per electrode, are compiled once."""

    def __init__(self, v: dolfinx.fem.Function, positions: Mapping[str, np.ndarray]):
        # physcardems' G = I convention: no conductivity, sigma_b = C_m = 1.
        self.recovery = beat.ECGRecovery(v=v, sigma_b=1.0, C_m=1.0, M=1.0)
        self.forms = {name: self.recovery.eval(point) for name, point in positions.items()}
        self.comm = v.function_space.mesh.comm

    def row(self, t: float) -> dict[str, float]:
        self.recovery.solve()
        potentials = {
            name: self.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
            for name, form in self.forms.items()
        }
        leads = beat.ecg.Leads12(**potentials)
        return {"t_ms": t} | {column: float(getattr(leads, lead)) for column, lead in LEADS.items()}


def write_fields_and_traces(
    post_dir: Path,
    case: Case,
    case_dir: Path,
    functions: Mapping[str, dolfinx.fem.Function],
    saved: Mapping[str, Mapping[float, np.ndarray]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Write ``fields.bp``, ``active_stats.csv`` and ``ecg.csv`` into ``post_dir``, and
    return the rows of the last two.

    ``functions`` are :func:`demo_io.result_functions`'s, on ``case``'s mesh, and
    ``saved`` their values by name and time, as
    :func:`~simcardemsx.results.read_results` returns them.
    """
    lmbda, tension, v = (functions[name] for name in ("lmbda", "tension_kPa", "v"))
    stats = ActiveStats(case, lmbda.function_space)
    ecg = PseudoECG(v, electrodes(case_dir))
    stats_rows: list[dict[str, float]] = []
    ecg_rows: list[dict[str, float]] = []

    def at_each_time(t: float, p1: Mapping[str, dolfinx.fem.Function]) -> None:
        if t in saved["lmbda"] and t in saved["tension_kPa"]:
            stats_rows.append(stats.row(t, lmbda, tension))
        if t in saved["v"]:
            ecg_rows.append(ecg.row(t))

    write_p1_fields(
        post_dir / "fields.bp",
        functions,
        saved,
        {"u": "u", "v": "v", "lmbda": "lmbda", "Ta": "tension_kPa"},
        at_each_time,
    )
    for name, fields, rows in (
        ("active_stats.csv", ACTIVE_STATS_FIELDS, stats_rows),
        ("ecg.csv", ECG_FIELDS, ecg_rows),
    ):
        log = CsvLog(post_dir / name, fields)
        log.start()
        for row in rows:
            log.append(row)
    return stats_rows, ecg_rows


def plot_ecg(rows: Sequence[Mapping[str, float]], path: Path) -> None:
    """The twelve leads over time, in physcardems' 3 x 4 layout, on one shared scale so
    their amplitudes compare. Uses matplotlib's ``Figure`` API, with no pyplot."""
    from matplotlib.figure import Figure

    t = np.array([row["t_ms"] for row in rows])
    fig = Figure(figsize=(12.0, 6.5), layout="constrained")
    axes = fig.subplots(3, 4, sharex=True, sharey=True)
    for r, leads in enumerate(ECG_LAYOUT):
        for c, lead in enumerate(leads):
            ax = axes[r, c]
            ax.axhline(0.0, color=GRID, linewidth=1.0)
            ax.plot(t, [row[lead] for row in rows], color=SERIES, linewidth=1.5)
            ax.set_title(lead, loc="left", fontsize=10, color=INK)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_SECONDARY)
            ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_SECONDARY, labelsize=8)
            if r == len(ECG_LAYOUT) - 1:
                ax.set_xlabel("t (ms)", color=INK_SECONDARY)
    fig.suptitle("Pseudo-ECG (G = I, sigma_b = 1)", x=0.0, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)


def run_failure(run: Mapping[str, Any] | None) -> str | None:
    """The summary's ``failure``, from ``run.json`` (``None`` if there is none): its
    ``failure``, or, while its ``status`` is still ``running``, a note that the run did
    not finish. A run killed outright (SIGKILL, or the machine going down) never
    rewrites ``run.json``, and its summary must not say that Newton converged at every
    step of a run that stopped."""
    if run is None:
        return None
    if run.get("status") == "running":
        return "the run did not finish: run.json's status is still 'running'"
    return run.get("failure")


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

    from rodero_05 import main as rodero

    settings = read_resolved_settings(folder / "config.resolved.toml")
    columns = read_columns(folder / "log.csv")
    run = folder / "run.json"
    failure = run_failure(json.loads(run.read_text()) if run.exists() else None)
    case_dir = Path(settings["case_dir"]["resolved"])
    try:
        case = load_case(case_dir, tref_scale=settings["tref"])
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    functions = result_functions(case.geometry.mesh, settings, rodero.EP_RESULTS)
    saved = read_results(folder, functions)

    post_dir = folder / "post"
    shutil.rmtree(post_dir, ignore_errors=True)
    post_dir.mkdir()
    dt = settings["dt_mech"]
    summary = summarise(
        columns,
        failure,
        settings["land_placement"],
        t_end_ms=round(settings["t_end"] / dt) * dt,
    )
    (post_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _, ecg_rows = write_fields_and_traces(post_dir, case, case_dir, functions, saved)
    if plots_available():
        plot(columns, summary, post_dir / "pv_loops.png")
        plot_ecg(ecg_rows, post_dir / "ecg.png")


if __name__ == "__main__":
    main()
