"""Which code wrote a run: git commit, dirty flag, package versions, ranks and time."""

import datetime
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

from mpi4py import MPI

PACKAGES = (
    "simcardemsx",
    "fenics-dolfinx",
    "fenicsx-pulse",
    "fenicsx-beat",
    "gotranx",
    "crossbridge",
    "io4dolfinx",
    "numpy",
    "petsc4py",
)


def _git(args: list[str], cwd: Path) -> str | None:
    """``git <args>``'s stdout in ``cwd``, or None if git fails (best effort)."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def git_commit(cwd: Path) -> str | None:
    """The commit checked out in ``cwd``, or None."""
    out = _git(["rev-parse", "HEAD"], cwd)
    return (out.strip() or None) if out is not None else None


def git_dirty(cwd: Path) -> bool | None:
    """Whether a tracked file differs from that commit (untracked files ignored), or None.

    ``--no-optional-locks`` keeps the query from taking the index lock, so it cannot
    race a concurrent git command.
    """
    out = _git(["--no-optional-locks", "status", "--porcelain", "--untracked-files=no"], cwd)
    return bool(out.strip()) if out is not None else None


def versions() -> dict[str, str | None]:
    """The installed version of each of ``PACKAGES``, or None if it is not installed."""
    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def provenance(cwd: Path, comm: Any = MPI.COMM_WORLD) -> dict[str, Any]:
    """Git state of ``cwd``, package versions, the communicator's size and the UTC time."""
    return {
        "git_commit": git_commit(cwd),
        "git_dirty": git_dirty(cwd),
        "versions": versions(),
        "n_ranks": comm.size,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
