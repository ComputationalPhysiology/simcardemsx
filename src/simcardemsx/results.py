"""Results in the upstream CLIs' layout: ``results.bp``, ``log.csv``, resolved settings.

A run's folder holds ``config.resolved.toml``, ``run.json``, ``results.bp`` (Functions,
through io4dolfinx, in input-mesh order), ``log.csv`` (one row per logged step),
``restart.bp`` and ``restart.json`` (see :mod:`simcardemsx.checkpoint`) and ``post/``.
:func:`prepare_output` applies the CLIs' rules to such a folder before a run starts.
"""

from __future__ import annotations

import csv
import fnmatch
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import toml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: read with toml, which wrote the file
    tomllib = None  # type: ignore[assignment]

from .checkpoint import RESTART, RESTART_META, check_restart, write_json

__all__ = [
    "ARTIFACTS",
    "RESTART",
    "RESTART_META",
    "RESULTS",
    "CsvLog",
    "ResultsWriter",
    "prepare_output",
    "read_resolved_settings",
    "read_result_times",
    "read_results",
    "stride",
    "write_json",
    "write_resolved_settings",
]

RESULTS = "results.bp"
ARTIFACTS = (
    "config.resolved.toml",
    "run.json",
    RESULTS,
    "log.csv",
    RESTART,
    RESTART_META,
    "timings.json",
    "post",
)


def stride(every_ms: float, dt_ms: float) -> int:
    """The number of steps of ``dt_ms`` in ``every_ms``.

    Raises ``ValueError`` if ``every_ms`` is not positive or not a whole multiple of
    ``dt_ms`` (relative tolerance 1e-9).
    """
    if every_ms <= 0 or dt_ms <= 0:
        raise ValueError(f"every ({every_ms} ms) and dt ({dt_ms} ms) must be positive")
    n = max(1, round(every_ms / dt_ms))
    if not np.isclose(n * dt_ms, every_ms, rtol=1e-9, atol=0.0):
        raise ValueError(f"{every_ms} ms is not a whole multiple of the step {dt_ms} ms")
    return n


def _matches(folder: Path, artifacts: Sequence[str]) -> list[Path]:
    """Every path in ``folder`` that an entry of ``artifacts`` (a glob) names."""
    if not folder.is_dir():
        return []
    return [
        p
        for p in sorted(folder.iterdir())
        if any(fnmatch.fnmatchcase(p.name, pattern) for pattern in artifacts)
    ]


def _decide(
    folder: Path,
    restart: bool,
    overwrite: bool,
    physics: Mapping[str, Any],
    artifacts: Sequence[str],
) -> tuple[str, str]:
    """``(action, message)`` from the filesystem; ``action`` may be ``"error"``."""
    if restart and overwrite:
        return "error", "--restart and --overwrite are mutually exclusive"
    if restart:
        try:
            check_restart(folder, physics)
        except FileNotFoundError as e:
            return "FileNotFoundError", str(e)
        except ValueError as e:  # includes json.JSONDecodeError (a corrupt restart.json)
            return "error", str(e)
        return "restart", ""
    if _matches(folder, artifacts):
        if not overwrite:
            return "error", (
                f"Output folder {folder} already holds results. Use --overwrite to replace "
                "them or --restart to continue the run."
            )
        return "wipe", ""
    return "create", ""


def prepare_output(
    folder: str | Path,
    *,
    restart: bool,
    overwrite: bool,
    physics: Mapping[str, Any],
    artifacts: Sequence[str] = ARTIFACTS,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> str:
    """Decide what to do with ``folder`` and do it. Returns ``"create"``, ``"wipe"`` or
    ``"restart"``.

    ``artifacts`` are names or fnmatch-style globs; ``"wipe"`` deletes only what they
    match. The decision is taken on rank 0 and broadcast, so every rank raises together.
    A stale ``restart.bp`` is an artifact: a fresh run is refused beside it, and
    ``overwrite`` removes it, so a fresh run never appends to another run's checkpoint.

    Raises
    ------
    ValueError
        If both flags are given; if the folder holds artifacts and neither is; or if
        ``restart`` and the physics differ from the checkpoint's.
    FileNotFoundError
        If ``restart`` and the folder holds no ``restart.json``.
    """
    folder = Path(folder)
    decision = None
    if comm.rank == 0:
        try:
            decision = _decide(folder, restart, overwrite, physics, artifacts)
        except Exception as e:  # e.g. a corrupt restart.json: all ranks must raise
            decision = ("error", f"Cannot prepare output folder {folder}: {e!r}")
    action, message = comm.bcast(decision, root=0)
    if action == "FileNotFoundError":
        raise FileNotFoundError(message)
    if action == "error":
        raise ValueError(message)
    if action == "restart":
        return action
    if action not in ("create", "wipe"):
        raise ValueError(f"Cannot prepare output folder {folder}: unexpected action {action!r}")

    failure = None
    if comm.rank == 0:
        try:
            if action == "wipe":
                for p in _matches(folder, artifacts):
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p)
                    else:
                        p.unlink(missing_ok=True)
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            failure = repr(e)
    failure = comm.bcast(failure, root=0)
    if failure is not None:
        raise OSError(f"Cannot prepare output folder {folder}: {failure}")
    return action


def read_result_times(
    folder: str | Path,
    name: str,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> np.ndarray:
    """Sorted, unique saved times of ``name`` in ``folder / results.bp``."""
    times = io4dolfinx.read_timestamps(
        filename=Path(folder) / RESULTS,
        comm=comm,
        function_name=name,
    )
    return np.unique(np.asarray(times, dtype=float))


def read_results(
    folder: str | Path,
    functions: Mapping[str, dolfinx.fem.Function],
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> dict[str, dict[float, np.ndarray]]:
    """Every saved time of each name, read into that Function's space.

    Returns ``{name: {t: array}}``, the arrays being copies of the Function's values.
    The Functions are overwritten.
    """
    out: dict[str, dict[float, np.ndarray]] = {}
    for name, f in functions.items():
        out[name] = {}
        for t in read_result_times(folder, name, comm):
            io4dolfinx.read_function(Path(folder) / RESULTS, f, time=float(t), name=name)
            f.x.scatter_forward()
            out[name][float(t)] = f.x.array.copy()
    return out


class ResultsWriter:
    """Writes Functions to ``folder / results.bp``, each name at most once per time."""

    def __init__(self, folder: str | Path, comm: MPI.Intracomm = MPI.COMM_WORLD) -> None:
        self.folder = Path(folder)
        self.comm = comm
        self._last: dict[str, float] = {}

    def write(self, t_ms: float, functions: Mapping[str, dolfinx.fem.Function]) -> None:
        """Write each function at ``t_ms``, unless that name already has a time not
        earlier than ``t_ms`` by more than ``1e-9 * max(1, t_ms)``."""
        for name, f in functions.items():
            last = self._last.get(name)
            if last is not None and not t_ms > last + 1e-9 * max(1.0, t_ms):
                continue
            io4dolfinx.write_function_on_input_mesh(
                self.folder / RESULTS,
                f,
                time=float(t_ms),
                name=name,
            )
            self._last[name] = float(t_ms)

    def resume(self, names: Iterable[str]) -> None:
        """Take each name's last saved time from ``results.bp`` (none if it has none)."""
        if not (self.folder / RESULTS).exists():
            return
        for name in names:
            # io4dolfinx returns no times for a name never written to an existing file;
            # any real read failure propagates.
            times = read_result_times(self.folder, name, self.comm)
            if times.size:
                self._last[name] = float(times[-1])


class CsvLog:
    """A CSV log of floats whose first column is ``t_ms``. Written on rank 0 only."""

    def __init__(
        self,
        path: str | Path,
        fields: Sequence[str],
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        if not fields or fields[0] != "t_ms":
            raise ValueError(f"The first field must be 't_ms', got {list(fields)}")
        self.path = Path(path)
        self.fields = list(fields)
        self.comm = comm

    def start(self) -> None:
        """Write the header, replacing any file."""
        if self.comm.rank == 0:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.fields)
        self.comm.barrier()

    def append(self, row: Mapping[str, float]) -> None:
        """Append ``row`` with each value as ``repr(float(v))`` (exact round trip)."""
        if self.comm.rank == 0:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([repr(float(row[k])) for k in self.fields])

    def read(self) -> list[dict[str, float]]:
        """The rows as floats."""
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            return [{k: float(v) for k, v in r.items()} for r in reader]

    def resume(self, t_max_ms: float) -> list[dict[str, float]]:
        """Keep the rows with ``t_ms <= t_max_ms + 1e-9 * max(1, t)``, atomically, and
        return them. Raises ``ValueError`` if the file's header is not ``fields``."""
        with open(self.path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows or rows[0] != self.fields:
            raise ValueError(
                f"{self.path} has header {rows[0] if rows else None}, expected {self.fields}",
            )
        kept = []
        for r in rows[1:]:
            if not r:
                continue
            t = float(r[0])
            if t <= t_max_ms + 1e-9 * max(1.0, t):
                kept.append(r)
        if self.comm.rank == 0:
            tmp = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
            with open(tmp, "w", newline="") as f:
                csv.writer(f).writerows([rows[0], *kept])
            os.replace(tmp, self.path)
        self.comm.barrier()
        return [dict(zip(self.fields, map(float, r), strict=True)) for r in kept]


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def write_resolved_settings(path: str | Path, settings: Mapping[str, Any]) -> None:
    """Write ``settings`` as TOML, with ``Path`` values as ``str``."""
    Path(path).write_text(toml.dumps(_plain(settings)))


def read_resolved_settings(path: str | Path) -> dict[str, Any]:
    """Read a TOML file written by :func:`write_resolved_settings`: with the standard
    library's ``tomllib`` where there is one (Python >= 3.11), else with ``toml``."""
    if tomllib is None:
        return toml.load(path)
    with open(path, "rb") as f:
        return tomllib.load(f)
