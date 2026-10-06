"""Provenance: which code wrote a run."""

import datetime
import subprocess
from pathlib import Path

from mpi4py import MPI

from simcardemsx import provenance as prov


def test_versions_names_every_package():
    v = prov.versions()
    assert tuple(v) == prov.PACKAGES
    assert v["simcardemsx"] is not None


def test_provenance_has_its_fields():
    p = prov.provenance(Path(__file__).resolve().parent, MPI.COMM_WORLD)
    assert set(p) == {"git_commit", "git_dirty", "versions", "n_ranks", "utc"}
    assert p["n_ranks"] == MPI.COMM_WORLD.size
    assert p["versions"] == prov.versions()
    stamp = datetime.datetime.fromisoformat(p["utc"])
    assert stamp.utcoffset() == datetime.timedelta(0)


def test_git_provenance_of_a_scratch_repository(tmp_path, monkeypatch):
    """``git_dirty`` looks at the tracked files only; both are None outside a repository."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (repo / "tracked.txt").write_text("a\n")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    sha = prov.git_commit(repo)
    assert sha is not None and len(sha) == 40
    assert prov.git_dirty(repo) is False
    (repo / "untracked.txt").write_text("b\n")
    assert prov.git_dirty(repo) is False
    (repo / "tracked.txt").write_text("changed\n")
    assert prov.git_dirty(repo) is True

    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert prov.git_commit(outside) is None
    assert prov.git_dirty(outside) is None
