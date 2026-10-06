"""A per-run recorder for the scheme comparison.

Call ``Recorder.step(t_ms, newton_iterations)`` after each ``backend.post_solve()`` and
``finish`` once at the end, whether the run completed or not, after the example's own
files: ``run.json`` is what ``run.py`` takes as the mark of a finished run, so it is
written last (:func:`finish_after_artifacts` does both, in that order). Serial only;
writes on rank 0. Imported as ``from scheme_comparison.record import Recorder`` with
``numerical_experiments/`` on ``sys.path``.
"""

import contextlib
import csv
import datetime
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import dolfinx
import numpy as np
import ufl

from scheme_comparison import metrics
from simcardemsx import provenance

COLUMNS = [
    "t_ms",
    "newton_iterations",
    "lmbda_mean",
    "lmbda_min",
    "lmbda_max",
    "Ta_mean_kPa",
    "Ka_max_kPa",
    "KaDl_max_kPa",
    *[f"reversal_fraction_{f:g}" for f in metrics.FLOORS_PER_MS],
]


logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def _git_commit(cwd: Path = HERE) -> str | None:
    """The commit checked out in ``cwd``, or None."""
    return provenance.git_commit(cwd)


def _git_dirty(cwd: Path = HERE) -> bool | None:
    """Whether a tracked file differs from that commit (untracked files ignored), or None."""
    return provenance.git_dirty(cwd)


REQUIRED_RUN_INFO = ("geometry", "split", "scheme", "dt_mech_ms", "t_end_ms", "regime")
REQUIRED_REGIME = ("Kp_kPa", "eta_Pa_s", "rho_kg_m3", "h_m")


def _check_run_info(run_info: Mapping[str, Any]) -> None:
    """Refuse a ``run_info`` that ``finish`` could not write, before the run starts."""
    missing = [k for k in REQUIRED_RUN_INFO if k not in run_info]
    if "regime" in run_info:
        missing += [f"regime.{k}" for k in REQUIRED_REGIME if k not in run_info["regime"]]
    if missing:
        raise ValueError(f"run_info is missing {', '.join(missing)}")


class Recorder:
    """Records one run. ``KaDl_max_kPa`` is a magnitude, ``max |Ka (λ_{n+1} − λ_n)|``."""

    def __init__(
        self,
        backend,
        outdir: Path,
        *,
        run_info: Mapping[str, Any],
        snapshot_every_ms: float | None = None,
    ):
        _check_run_info(run_info)
        if backend.space.ufl_element().family_name != "quadrature":
            raise ValueError(
                "Recorder needs a backend with states on a quadrature space "
                f"(one weight per point), got {backend.space.ufl_element().family_name!r}",
            )
        self.backend = backend
        self.outdir = Path(outdir)
        self.run_info = dict(run_info)
        self.snapshot_every_ms = snapshot_every_ms
        self.comm = backend.mesh.comm
        # The run's provenance is the code it starts with.
        self.git_commit = _git_commit()
        self.git_dirty = _git_dirty()
        dx = ufl.dx(domain=backend.mesh, metadata={"quadrature_degree": backend.quadrature_degree})
        self.weights = dolfinx.fem.assemble_vector(
            dolfinx.fem.form(ufl.TestFunction(backend.space) * dx),
        ).array.copy()

        self.rows: list[list[float]] = []
        self.newton: list[int] = []
        self.snap_t: list[float] = []
        self.snaps: dict[str, list[np.ndarray]] = {
            "lmbda": [],
            "tension_kPa": [],
            "stiffness_kPa": [],
        }
        self._lmbda = backend.lmbda_prev.x.array.copy()
        self._d_prev: np.ndarray | None = None
        self._t = 0.0
        self._sidecar_t: float | None = None
        self.rows.append(self._row(0.0, 0, 0.0, [0.0] * len(metrics.FLOORS_PER_MS)))
        self._maybe_snapshot(0.0)

    def _row(self, t: float, iters: int, KaDl: float, reversals: list[float]) -> list[float]:
        b, w = self.backend, self.weights
        lmbda = b.lmbda_prev.x.array
        return [
            t,
            iters,
            metrics.weighted_mean(lmbda, w),
            float(lmbda.min()),
            float(lmbda.max()),
            metrics.weighted_mean(b.tension_kPa.x.array, w),
            float(b.stiffness_kPa.x.array.max()),
            KaDl,
            *reversals,
        ]

    def _maybe_snapshot(self, t: float) -> None:
        every = self.snapshot_every_ms
        if every is None:
            return
        k = round(t / every)
        if abs(t - k * every) > 1e-9 * max(1.0, abs(t)):
            return
        self.snap_t.append(t)
        self.snaps["lmbda"].append(self.backend.lmbda_prev.x.array.copy())
        self.snaps["tension_kPa"].append(self.backend.tension_kPa.x.array.copy())
        self.snaps["stiffness_kPa"].append(self.backend.stiffness_kPa.x.array.copy())

    def step(self, t_ms: float, newton_iterations: int) -> None:
        b = self.backend
        dt = t_ms - self._t
        lmbda = b.lmbda_prev.x.array.copy()
        d_curr = lmbda - self._lmbda
        if self._d_prev is None:
            reversals = [0.0] * len(metrics.FLOORS_PER_MS)
        else:
            reversals = [
                metrics.reversal_fraction(self._d_prev, d_curr, self.weights, f * dt)
                for f in metrics.FLOORS_PER_MS
            ]
        KaDl = float(np.max(np.abs(b.stiffness_kPa.x.array * d_curr)))
        self.rows.append(self._row(t_ms, newton_iterations, KaDl, reversals))
        self.newton.append(int(newton_iterations))
        self._maybe_snapshot(t_ms)
        self._d_prev = d_curr
        self._lmbda = lmbda
        self._t = t_ms

    # -- Checkpointer component (see simcardemsx.checkpoint.Checkpointable) --------------
    namespace = "recorder"

    def restart_functions(self) -> list:
        return []

    def restart_metadata(self) -> dict[str, Any]:
        return {"t_ms": self._t}

    def load_restart(self, functions, metadata: Mapping[str, Any]) -> None:
        """Take the time back. The rows come from :meth:`read_sidecar`."""
        if "t_ms" not in metadata:
            raise ValueError("The recorder's restart metadata has no 't_ms'")
        self._t = float(metadata["t_ms"])

    @staticmethod
    def _sidecar(folder: Path, t_ms: float) -> Path:
        return Path(folder) / f"restart_recorder_{t_ms!r}.npz"

    def write_sidecar(self, folder: Path, t_ms: float) -> None:
        """Write every row and snapshot so far to ``restart_recorder_<t_ms!r>.npz``.

        Keeps this file and the one written at the previous call (which the current
        ``restart.json`` names until the new one replaces it); deletes the others.
        Rows and snapshots are stored as float arrays, so they come back bit for bit.

        The file is written atomically: into a temporary file in ``folder`` (named
        ``.restart_recorder_tmp_<pid>.npz``, which the ``restart_recorder_*.npz`` glob
        does not match), which ``os.replace`` then moves onto it. A checkpoint at a time
        already checkpointed rewrites the file that ``restart.json`` names, and a kill
        meanwhile must not leave it truncated.
        """
        if self.comm.rank != 0:
            return
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "rows": np.array(self.rows, dtype=float).reshape(-1, len(COLUMNS)),
            "newton": np.array(self.newton, dtype=np.int64),
            "snap_t": np.array(self.snap_t, dtype=float),
            "lmbda": self._lmbda,
            "has_d_prev": np.array(self._d_prev is not None),
            "d_prev": np.empty(0) if self._d_prev is None else self._d_prev,
            "t": np.array(self._t),
        }
        for name, values in self.snaps.items():
            arrays[f"snap_{name}"] = np.array(values, dtype=float) if values else np.empty((0, 0))
        target = self._sidecar(folder, t_ms)
        tmp = folder / f".restart_recorder_tmp_{os.getpid()}.npz"
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
        keep = {target.name}
        if self._sidecar_t is not None:
            keep.add(self._sidecar(folder, self._sidecar_t).name)
        for path in folder.glob("restart_recorder_*.npz"):
            if path.name not in keep:
                path.unlink()
        self._sidecar_t = t_ms

    def read_sidecar(self, folder: Path, t_ms: float) -> None:
        """Take the rows and snapshots back from the file written at ``t_ms``.

        Raises ``ValueError`` if the file's time is not ``t_ms``. Everything is read
        into locals first, so a refusal leaves the recorder as it was.
        """
        with np.load(self._sidecar(folder, t_ms)) as data:
            t = float(data["t"])
            if t != t_ms:
                raise ValueError(f"The recorder's sidecar is of t = {t} ms, not {t_ms} ms")
            # Column 1 is the Newton count: an int, as step() wrote it.
            rows = [[float(row[0]), int(row[1]), *map(float, row[2:])] for row in data["rows"]]
            newton = [int(n) for n in data["newton"]]
            snap_t = [float(x) for x in data["snap_t"]]
            snaps = {name: list(data[f"snap_{name}"]) for name in self.snaps}  # [] if empty
            lmbda = data["lmbda"].copy()
            d_prev = data["d_prev"].copy() if bool(data["has_d_prev"]) else None
        self.rows, self.newton, self.snap_t, self.snaps = rows, newton, snap_t, snaps
        self._lmbda, self._d_prev, self._t = lmbda, d_prev, t
        self._sidecar_t = t_ms

    def finish(
        self,
        *,
        failure: str | None,
        t_fail_ms: float | None,
        timings: Mapping[str, float],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Write ``steps.csv``, ``snapshots.npz`` and, last, ``run.json``.

        ``extra`` is merged into ``run.json`` after everything else, so a key in it
        overrides the recorder's own.
        """
        if self.comm.rank != 0:
            return
        self.outdir.mkdir(parents=True, exist_ok=True)
        with open(self.outdir / "steps.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(self.rows)
        if self.snapshot_every_ms is not None:
            arrays = {k: np.array(v) for k, v in self.snaps.items()}
            arrays["t_ms"] = np.array(self.snap_t)
            arrays["weights"] = self.weights
            np.savez(self.outdir / "snapshots.npz", **arrays)  # type: ignore[arg-type]
        t_end = self.run_info["t_end_ms"]
        reached = failure is None and abs(self._t - t_end) <= 1e-9 * max(1.0, t_end)
        n = len(self.newton)
        info = {
            **self.run_info,
            "reached_t_end": reached,
            "failure": failure,
            "t_fail_ms": t_fail_ms,
            "newton": {
                "total": sum(self.newton),
                "mean": sum(self.newton) / n if n else 0.0,
                "max": max(self.newton, default=0),
                "steps": n,
            },
            "timings": dict(timings),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **(extra or {}),
        }
        (self.outdir / "run.json").write_text(json.dumps(info, indent=2))


def failure_of(exc: BaseException) -> str:
    """``repr`` of ``exc`` and of each exception behind it, outermost first, joined by " <- ".

    For ``run.json``'s ``failure``. petsc4py turns an exception raised in a SNES callback,
    a ``KeyboardInterrupt`` among them, into ``PETSc.Error(101)`` whose ``__cause__`` is
    the original. Without the chain an interrupted solve reads as a solver failure, and
    ``run.py`` would take the run as done.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(repr(current))
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


class RunFinisher(Protocol):
    """What :func:`finish_after_artifacts` writes ``run.json`` with: a :class:`Recorder`,
    or anything with the same ``finish``."""

    def finish(
        self,
        *,
        failure: str | None,
        t_fail_ms: float | None,
        timings: Mapping[str, float],
        extra: Mapping[str, Any] | None = None,
    ) -> None: ...


def finish_after_artifacts(
    recorder: RunFinisher,
    artifacts: Sequence[tuple[str, Callable[[], object]]],
    *,
    failure: str | None,
    t_fail_ms: float | None,
    timings: Mapping[str, float],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the example's own ``artifacts``, then ``recorder.finish`` (``run.json``) last.

    For the ``finally`` of an example's loop. ``artifacts`` are ``(name, write)`` pairs,
    written in order. Each is guarded: an ``Exception`` from one is logged, and the rest,
    and ``run.json``, are still written. If the loop raised (``failure`` is not None), its
    exception is the one that leaves the ``finally``; otherwise the first artifact's
    error is raised once ``run.json`` is written. ``finish`` itself is not guarded: if it
    fails, there is no ``run.json`` and the run is not done. ``extra`` is passed on to
    :meth:`Recorder.finish`.
    """
    errors: list[Exception] = []
    for name, write in artifacts:
        try:
            write()
        except Exception as exc:
            logger.exception(f"Writing {name} failed")
            errors.append(exc)
    recorder.finish(failure=failure, t_fail_ms=t_fail_ms, timings=timings, extra=extra)
    if errors and failure is None:
        raise errors[0]
