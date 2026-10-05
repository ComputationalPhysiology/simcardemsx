"""What the demos in ``numerical_experiments/`` share: their output folder, restart and
post-processing.

Imported as ``import demo_io``, with ``numerical_experiments/`` on ``sys.path``: each demo
puts it there, as it does for ``scheme_comparison``.

The run side, in the order a demo's ``main`` uses it:

- :func:`prepare_run` makes every refusal before anything expensive is built and before
  any file is written. It checks the strides of the output options, each error naming
  its option, then the physics, then :func:`~simcardemsx.results.prepare_output`. A
  refusal exits through ``SystemExit`` with the reason.
- :func:`start_or_resume`: a fresh run starts ``log.csv`` and writes its rows at t = 0;
  a restart restores the checkpoint and resumes ``log.csv`` and ``results.bp``.
- :func:`steps_to_take` counts the steps left to ``t_end``, none if the checkpoint is at
  or past it.
- :func:`failure_at` gives what ``run.json`` records of an exception from the loop.
- :func:`finish` writes, in order: the end checkpoint (:func:`write_end_checkpoint`),
  the demo's own files, and ``run.json`` last, with the provenance, the checkpoint
  history and whether the run was a restart.

The post side, for each demo's ``post.py``:

- :func:`write_p1_fields`: VTX on P1 of a run's ``results.bp`` fields, the quadrature
  fields averaged and the others interpolated, with a callback at each time.
- :class:`Panel` and :func:`figure`: quantities over time, stacked over a shared time
  axis; :func:`plots_available` warns and says so when matplotlib is missing.
- :func:`cell_containing`: the cell holding a point, for point traces.
"""

import functools
import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import dolfinx
import numpy as np
from scheme_comparison.record import Recorder, failure_of, finish_after_artifacts

from simcardemsx.averaging import make_averager
from simcardemsx.checkpoint import Checkpointer
from simcardemsx.controller import SimulationController
from simcardemsx.provenance import provenance
from simcardemsx.results import CsvLog, ResultsWriter, prepare_output, stride

logger = logging.getLogger(__name__)

#: What a demo with a scheme-comparison ``Recorder`` writes besides
#: :data:`simcardemsx.results.ARTIFACTS`, and so what ``--overwrite`` also deletes:
#: ``steps.csv``, ``snapshots.npz`` and the recorder's checkpoint sidecars.
RECORDER_ARTIFACTS = ("steps.csv", "snapshots.npz", "restart_recorder_*.npz")


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------


def prepare_run(
    outdir: Path,
    *,
    restart: bool,
    overwrite: bool,
    strides: Mapping[str, tuple[float, float]],
    physics: Callable[[], dict[str, Any]],
    artifacts: Sequence[str],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Every refusal of a run, before anything expensive and before any file is written.

    In order: each of ``strides``, ``{option: (every_ms, dt_ms)}``, must be a whole
    number of steps (:func:`~simcardemsx.results.stride`); ``physics()`` is computed,
    and may raise ``FileNotFoundError`` for an input that is missing; and
    :func:`~simcardemsx.results.prepare_output` decides on ``outdir``, which it
    creates, or wipes of ``artifacts`` with ``overwrite``.

    Returns ``(steps, physics)``: the number of steps in each stride, by option, and the
    physics. Raises ``SystemExit`` with the reason for any ``ValueError`` or
    ``FileNotFoundError`` on the way; a stride's names its option.
    """

    def steps(option: str, every: float, dt: float) -> int:
        try:
            return stride(every, dt)
        except ValueError as error:
            raise ValueError(f"{option}: {error}") from error

    try:
        counts = {option: steps(option, every, dt) for option, (every, dt) in strides.items()}
        run_physics = physics()
        prepare_output(
            outdir,
            restart=restart,
            overwrite=overwrite,
            physics=run_physics,
            artifacts=artifacts,
        )
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    return counts, run_physics


def start_or_resume(
    checkpointer: Checkpointer,
    log: CsvLog,
    results: ResultsWriter,
    *,
    restart: bool,
    result_names: Iterable[str],
    write_initial: Callable[[], None],
) -> list[dict[str, float]]:
    """Start the run's output, or restore the run and resume its output.

    A fresh run starts ``log`` (its header) and calls ``write_initial()``, which writes
    the rows at t = 0 (``results.bp`` and ``log.csv``); it returns ``[]``. A restart
    restores the checkpoint (``checkpointer.restore()``), keeps the rows of ``log`` up
    to its time (:meth:`~simcardemsx.results.CsvLog.resume`), takes each of
    ``result_names``' last saved time from ``results.bp``
    (:meth:`~simcardemsx.results.ResultsWriter.resume`), and returns the rows kept, the
    first of them the run's row at t = 0.

    A refused or failed restore raises before any file is written: neither the end
    checkpoint nor ``run.json`` is reached.
    """
    if restart:
        t = checkpointer.restore()
        kept = log.resume(t)
        results.resume(result_names)
        logger.info(f"Restarted from the checkpoint at t = {t} ms")
        return kept
    log.start()
    write_initial()
    return []


def steps_to_take(controller: SimulationController, t_end_ms: float) -> int:
    """The steps left to ``t_end_ms``: the run takes ``round(t_end_ms / dt_mech)`` steps
    from t = 0, and the controller has taken ``mech_step_idx`` of them. None, logged,
    if a restart's checkpoint is at or past ``t_end_ms``."""
    remaining = round(t_end_ms / controller.dt_mech) - controller.mech_step_idx
    if remaining <= 0:
        logger.info(f"The checkpoint, at t = {controller.t} ms, is at or past t_end: no step")
    return max(remaining, 0)


def failure_at(controller: SimulationController, error: BaseException) -> tuple[str, float]:
    """``run.json``'s ``failure`` and ``t_fail_ms`` for an exception from a demo's loop,
    which is logged with its traceback. Call it in the ``except``.

    A step that raised before its ``mech_callback`` was rolled back: the controller's
    ``t`` is its start, and ``t_failed`` its end, which is ``t_fail_ms``. Anything
    raised after that, in ``mech_callback`` (the log row, ``results.bp``, the recorder)
    or in a periodic checkpoint, follows an accepted step: ``t_failed`` is None, and
    ``t_fail_ms`` is the controller's ``t``, that step's end, whose output may be half
    written.
    """
    t_failed = controller.t_failed
    if t_failed is not None:
        logger.exception(f"The coupled step ending at t = {t_failed} ms failed")
        return failure_of(error), t_failed
    logger.exception(f"Writing the output of the step ending at t = {controller.t} ms failed")
    return failure_of(error), controller.t


def write_end_checkpoint(checkpointer: Checkpointer, failure: str | None) -> None:
    """The checkpoint at the end of a run, at the controller's ``t``.

    Written at the end of a run that finished, or whose last step failed and was rolled
    back to its start (``t_failed`` is set). Not after a failure that followed an
    accepted step (``failure`` is set and ``t_failed`` is None): that step's log row,
    ``results.bp`` fields or recorder row may be missing, and a checkpoint at its end
    would leave them missing for good, so the last periodic checkpoint stays the
    restart point. Not while the backend has a step pending. At a time already
    checkpointed, only ``restart.json`` and the sidecars are new.
    """
    controller = checkpointer.controller
    if failure is not None and controller.t_failed is None:
        logger.warning(
            f"No checkpoint at t = {controller.t} ms: the failure followed an accepted "
            "step, whose output may be incomplete",
        )
        return
    # Both CoupledBackends have step_pending; the protocol does not say so.
    if cast(Any, controller.backend).step_pending:
        logger.warning(f"No checkpoint at t = {controller.t} ms: a step is pending")
        return
    checkpointer.write()


def finish(
    recorder: Recorder,
    checkpointer: Checkpointer,
    artifacts: Sequence[tuple[str, Callable[[], object]]],
    *,
    failure: str | None,
    t_fail_ms: float | None,
    timings: Mapping[str, float],
    here: Path,
    restart: bool,
) -> None:
    """The end of a run, for the ``finally`` of its loop.

    In order, each guarded so that a failure cannot stop the rest: the end checkpoint
    (:func:`write_end_checkpoint`), then the demo's own ``artifacts`` (``(name, write)``
    pairs). Then ``run.json``, last, through
    :func:`~scheme_comparison.record.finish_after_artifacts`, with ``provenance``
    (:func:`~simcardemsx.provenance.provenance` taken in ``here``), ``history`` (the
    checkpointer's: every process that wrote the run) and ``restart``.
    """
    finish_after_artifacts(
        recorder,
        [("checkpoint", lambda: write_end_checkpoint(checkpointer, failure)), *artifacts],
        failure=failure,
        t_fail_ms=t_fail_ms,
        timings=timings,
        extra={
            "provenance": provenance(here),
            "history": checkpointer.history,
            "restart": restart,
        },
    )


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------


def write_p1_fields(
    path: Path,
    functions: Mapping[str, dolfinx.fem.Function],
    saved: Mapping[str, Mapping[float, np.ndarray]],
    shown: Mapping[str, str],
    at_each_time: Callable[[float, Mapping[str, dolfinx.fem.Function]], None] | None = None,
) -> None:
    """Write VTX at ``path`` of a run's fields on P1, at every time any of ``functions``
    was saved at.

    ``functions`` are one Function per ``results.bp`` name, all on one mesh, and
    ``saved`` their values by name and time, as
    :func:`~simcardemsx.results.read_results` returns them. ``shown`` maps each name in
    the VTX file to the name it shows: a P1 Function (vector-valued if the field is)
    holds it, averaged (:func:`~simcardemsx.averaging.make_averager`) from a quadrature
    field and interpolated from any other.

    At each time, in order, every one of ``functions`` saved at it takes its saved
    values (the others keep theirs, so a field not saved at a time keeps its last saved
    value), the P1 Functions are refreshed and written, and ``at_each_time(t, p1)`` is
    called with them, by their VTX names.
    """
    p1: dict[str, dolfinx.fem.Function] = {}
    refresh: list[Callable[[], None]] = []
    for name, source_name in shown.items():
        source = functions[source_name]
        space = source.function_space
        shape = tuple(space.value_shape)
        target = dolfinx.fem.Function(
            dolfinx.fem.functionspace(space.mesh, ("P", 1, shape) if shape else ("P", 1)),
            name=name,
        )
        if space.ufl_element().family_name == "quadrature":
            refresh.append(make_averager(source, target))
        else:
            refresh.append(functools.partial(target.interpolate, source))
        p1[name] = target

    mesh = next(iter(p1.values())).function_space.mesh
    times = sorted({t for by_time in saved.values() for t in by_time})
    writer = dolfinx.io.VTXWriter(mesh.comm, path, list(p1.values()))
    try:
        for t in times:
            for name, f in functions.items():
                if t in saved[name]:
                    f.x.array[:] = saved[name][t]
            for update in refresh:
                update()
            writer.write(t)
            if at_each_time is not None:
                at_each_time(t, p1)
    finally:
        writer.close()


# The plots' colours: one series per panel, so one hue.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


class Panel(NamedTuple):
    """One quantity over time: a line, or with ``counts`` a dot per value on an integer
    axis. ``nan`` values (times the quantity was not saved at) are left out."""

    title: str
    t: np.ndarray
    values: np.ndarray
    counts: bool = False


def plots_available() -> bool:
    """Whether matplotlib can be imported; if not, a warning says the plots are skipped."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        warnings.warn("matplotlib is not available: the plots are skipped", stacklevel=3)
        return False
    return True


def figure(path: Path, panels: Sequence[Panel]) -> None:
    """The panels stacked over a shared time axis, one series each, saved at ``path``.

    Uses matplotlib's ``Figure`` API, with no pyplot and no global backend.
    """
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


def cell_containing(mesh: dolfinx.mesh.Mesh, point: np.ndarray) -> int:
    """A local cell of ``mesh`` that holds ``point`` (shape ``(1, 3)``). Raises
    ``ValueError`` if none does."""
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, point)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, point).links(0)
    if len(cells) == 0:
        raise ValueError(f"The point {point[0].tolist()} is not in the mesh")
    return int(cells[0])
