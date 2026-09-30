"""A per-run recorder for the scheme comparison.

Call ``Recorder.step(t_ms, newton_iterations)`` after each ``backend.post_solve()`` and
``finish`` once at the end, whether the run completed or not. Serial only; writes on
rank 0. Imported as ``from scheme_comparison.record import Recorder`` with
``numerical_experiments/`` on ``sys.path``.
"""

import csv
import datetime
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dolfinx
import numpy as np
import ufl

from scheme_comparison import metrics

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


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


class Recorder:
    def __init__(
        self,
        backend,
        outdir: Path,
        *,
        run_info: Mapping[str, Any],
        snapshot_every_ms: float | None = None,
    ):
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
        KaDl = float(np.max(b.stiffness_kPa.x.array * d_curr))
        self.rows.append(self._row(t_ms, newton_iterations, KaDl, reversals))
        self.newton.append(int(newton_iterations))
        self._maybe_snapshot(t_ms)
        self._d_prev = d_curr
        self._lmbda = lmbda
        self._t = t_ms

    def finish(
        self,
        *,
        failure: str | None,
        t_fail_ms: float | None,
        timings: Mapping[str, float],
    ) -> None:
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
            "git_commit": _git_commit(),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (self.outdir / "run.json").write_text(json.dumps(info, indent=2))
