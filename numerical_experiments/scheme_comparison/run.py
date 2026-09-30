"""Batch runner for the slab and BiV studies of the scheme comparison.

Each run is one invocation of an example's ``main.py`` in a subprocess, written to
``<root>/<geometry>/<split>/<scheme>/dt<dt>/``, where the example leaves ``steps.csv``,
``run.json`` and, with snapshots, ``snapshots.npz``, and this runner adds ``stdout.log``
and ``launcher.json`` (``returncode``, ``wall_s``, ``argv``). A run is done once it has
a ``run.json``, so a re-run skips finished runs and redoes the rest. A non-zero return
does not stop the batch: the naive scheme is expected to fail.

Usage::

    python3 run.py --study {slab,biv} [--root DIR] [--dry-run]

Runs are ordered monolithic, stabilized, segregated, coarse time step first. Serial only.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent

SCHEMES = ("monolithic", "stabilized", "segregated")


@dataclass(frozen=True)
class Run:
    geometry: str
    split: str
    scheme: str
    dt_mech: float  # ms
    t_end: float  # ms
    snapshot_every: float  # ms

    def outdir(self, root: Path) -> Path:
        return root / self.geometry / self.split / self.scheme / f"dt{self.dt_mech:g}"

    def command(self, root: Path) -> tuple[list[str], Path]:
        """The argv and the working directory of this run."""
        outdir = str(self.outdir(root).resolve())
        common = [
            "--scheme",
            self.scheme,
            "--dt-mech",
            f"{self.dt_mech:g}",
            "--t-end",
            f"{self.t_end:g}",
            "--snapshot-every",
            f"{self.snapshot_every:g}",
        ]
        if self.geometry == "slab":
            # The slab's paths are relative to its own directory.
            return (
                [
                    sys.executable,
                    "main.py",
                    "--odefile",
                    f"../odefiles/ToRORd_dynCl_endo_{self.split}.ode",
                    *common,
                    "--output-dir",
                    outdir,
                ],
                EXAMPLES / "strong_coupling_zetasplit",
            )
        return (
            [sys.executable, "main.py", *common, "--outdir", outdir],
            EXAMPLES / "circulation_biv",
        )


def _runs(
    geometry: str,
    splits: tuple[str, ...],
    dts: tuple[float, ...],
    t_end: float,
    snapshot_every: float,
) -> tuple[Run, ...]:
    return tuple(
        Run(geometry, split, scheme, dt, t_end, snapshot_every)
        for split in splits
        for scheme in SCHEMES
        for dt in dts
    )


MATRIX: dict[str, tuple[Run, ...]] = {
    "slab": _runs("slab", ("zetasplit", "caisplit"), (2, 1, 0.5, 0.25, 0.05), 200, 1),
    "biv": _runs("biv", ("zetasplit",), (2, 1, 0.5, 0.25), 500, 10),
}


def pending_runs(root: Path, runs: tuple[Run, ...] | list[Run]) -> list[Run]:
    """The runs without a ``run.json``."""
    return [r for r in runs if not (r.outdir(root) / "run.json").exists()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", choices=sorted(MATRIX), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=HERE / "output",
        help="Where the runs go (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands of the pending runs and run nothing.",
    )
    args = parser.parse_args(argv)

    runs = pending_runs(args.root, MATRIX[args.study])
    print(f"{len(runs)} of {len(MATRIX[args.study])} runs pending")
    for i, run in enumerate(runs, 1):
        cmd, cwd = run.command(args.root)
        if args.dry_run:
            print(f"(cd {cwd} && {' '.join(cmd)})")
            continue
        outdir = run.outdir(args.root)
        outdir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(outdir / "stdout.log", "w") as log:
            returncode = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        wall_s = time.perf_counter() - start
        (outdir / "launcher.json").write_text(
            json.dumps({"returncode": returncode, "wall_s": wall_s, "argv": cmd}, indent=2),
        )
        print(f"[{i}/{len(runs)}] {outdir}: returncode {returncode} [{wall_s:.1f} s]")


if __name__ == "__main__":
    main()
