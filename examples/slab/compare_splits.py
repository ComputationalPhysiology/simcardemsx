"""Run the slab with each of the three splits of ToR-ORd + Land and compare them.

``python compare_splits.py [--overwrite] [--media] [--t-end 400] [--dt-mech 1]``, from
this directory (or anywhere: the runs are subprocesses started in it).

For each split in ``odefiles.composed.SPLITS`` it runs ``main.py --split <s>`` and then
``post.py`` into ``output/splits/<s>/``, unless that run is done
(``scheme_comparison.run.is_done``; ``--overwrite`` redoes all of them). It then writes

- ``output/splits/splits.png``: ``v``, ``cai``, ``Ta`` and the stretch, as volume means,
  with the three splits overlaid, and a fifth row with each split's difference from the
  zeta split in every quantity, relative to that quantity's peak in the zeta split;
- a table on stdout: what crosses for each split (:func:`~simcardemsx.transfer_plan.resolve`:
  EP to mechanics, mechanics to EP, and whether the stretch goes to EP), the run's
  ``loop_s``, and the largest relative difference from zeta in ``Ta`` and in the stretch.

With ``--media`` it also copies ``splits.png`` into ``media/`` and writes
``media/splits.gif`` (the three slabs side by side, warped by ``u`` and coloured by ``Ta``
on one range) and ``media/provenance.json``.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
sys.path.insert(0, str(EXAMPLES))
from odefiles.composed import SPLITS  # noqa: E402
from scheme_comparison.run import is_done  # noqa: E402

REFERENCE = "zeta"
SPLIT_FOLDER = HERE / "output" / "splits"
MEDIA = HERE / "media"
MAX_FRAMES = 100
#: The volume means compared, with the label and the axis of each.
QUANTITIES = {
    "v_mean_mV": "v (mV)",
    "cai_mean": "cai (mM)",
    "Ta_mean_kPa": "Ta (kPa)",
    "lmbda_mean": "stretch",
}
#: Distinguishable by colour and by line style, since the curves nearly coincide.
STYLES = {
    "zeta": ("#2a78d6", "-", 2.6),
    "cai": ("#d9531e", "--", 1.8),
    "catrpn": ("#2a9d5c", ":", 1.8),
}

Traces = Mapping[str, Mapping[str, np.ndarray]]


def difference_series(traces: Traces) -> dict[str, dict[str, np.ndarray]]:
    """``|x_split - x_zeta| / max|x_zeta|`` at every time, by split and quantity, for each
    split other than the reference. ``nan`` where either was not saved."""
    reference = traces[REFERENCE]
    result: dict[str, dict[str, np.ndarray]] = {}
    for split, values in traces.items():
        if split == REFERENCE:
            continue
        result[split] = {
            name: np.abs(values[name] - reference[name]) / np.nanmax(np.abs(reference[name]))
            for name in QUANTITIES
        }
    return result


def relative_differences(traces: Traces) -> dict[str, dict[str, float]]:
    """The largest of :func:`difference_series`, by split and quantity."""
    return {
        split: {name: float(np.nanmax(series)) for name, series in by_name.items()}
        for split, by_name in difference_series(traces).items()
    }


def pending_splits(root: Path, splits: Sequence[str], overwrite: bool = False) -> list[str]:
    """The splits whose run in ``root/<split>`` has to be (re)done."""
    return [s for s in splits if overwrite or not is_done(root / s)]


def read_traces(folder: Path) -> dict[str, np.ndarray]:
    """``post/traces.csv`` of a run, by column."""
    with (folder / "post" / "traces.csv").open() as f:
        header = f.readline().strip().split(",")
    data = np.atleast_2d(
        np.genfromtxt(folder / "post" / "traces.csv", delimiter=",", skip_header=1),
    )
    return {name: data[:, i] for i, name in enumerate(header)}


def write_figure(path: Path, traces: Traces) -> None:
    """Five rows over a shared time axis: the four volume means of every split, and the
    splits' differences from zeta relative to zeta's peak."""
    from demo_io import GRID, INK, INK_SECONDARY
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.0, 0.8 + 2.2 * 5), layout="constrained")
    axes = fig.subplots(5, 1, sharex=True)

    def style(ax, title: str) -> None:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_SECONDARY, labelsize=8)

    for ax, (name, label) in zip(axes[:4], QUANTITIES.items()):
        for split, values in traces.items():
            # The stretch is saved as 0 at t = 0, before the first mechanics step.
            keep = np.isfinite(values[name]) & (name != "lmbda_mean" or values[name] != 0.0)
            colour, line, width = STYLES[split]
            ax.plot(
                values["t_ms"][keep],
                values[name][keep],
                color=colour,
                linestyle=line,
                linewidth=width,
                label=split,
            )
        ax.ticklabel_format(axis="y", useOffset=False)
        style(ax, label)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[4]
    t = traces[REFERENCE]["t_ms"]
    for split, by_name in difference_series(traces).items():
        for name, series in by_name.items():
            keep = np.isfinite(series)
            ax.plot(
                t[keep],
                series[keep],
                color=STYLES[split][0],
                linestyle=STYLES[split][1],
                linewidth=1.0,
                alpha=0.9,
            )
    ax.set_yscale("symlog", linthresh=1e-6)
    style(ax, "|difference from zeta| / peak, each split and quantity")
    ax.set_xlabel("t (ms)", color=INK_SECONDARY)
    fig.savefig(path, dpi=110)


def run_split(split: str, folder: Path, dt_mech: float, t_end: float) -> None:
    """``main.py`` then ``post.py`` for ``split`` into ``folder``, as subprocesses in
    :data:`HERE`."""
    for argv in (
        [
            "main.py",
            "--split",
            split,
            "--dt-mech",
            f"{dt_mech:g}",
            "--t-end",
            f"{t_end:g}",
            "--output-dir",
            str(folder),
            "--overwrite",
        ],
        ["post.py", "--output-dir", str(folder)],
    ):
        subprocess.run([sys.executable, *argv], cwd=HERE, check=True)


def crossings(split: str) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """What ``resolve`` finds crossing for ``split``: EP to mechanics, mechanics to EP, and
    whether the stretch goes to EP. The generated code is the slab's own (cached)."""
    from odefiles.composed import torord_land

    from simcardemsx.ode_model import load_ode_modules
    from simcardemsx.transfer_plan import resolve

    odefile = torord_land(split)
    modules = load_ode_modules(odefile, HERE / "generated_odes" / odefile.stem)
    found = resolve(modules.ep, modules.mechanics)
    return tuple(found.forward), tuple(found.backward), bool(found.stretch_to_ep)


def table(root: Path, traces: Traces, splits: Sequence[str]) -> str:
    """The comparison as text: crossings, ``loop_s`` and the largest relative differences."""
    diffs = relative_differences(traces)
    rows = [
        (
            "split",
            "EP -> mechanics",
            "mechanics -> EP",
            "stretch to EP",
            "loop_s",
            "max dTa",
            "max dλ",
        ),
    ]
    for split in splits:
        forward, backward, stretch = crossings(split)
        loop_s = json.loads((root / split / "timings.json").read_text())["loop_s"]
        d = diffs.get(split)
        rows.append(
            (
                split,
                ", ".join(forward) or "-",
                ", ".join(backward) or "-",
                "yes" if stretch else "no",
                f"{loop_s:.1f}",
                "-" if d is None else f"{d['Ta_mean_kPa']:.3e}",
                "-" if d is None else f"{d['lmbda_mean']:.3e}",
            ),
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append("")
    lines.append(
        "differences are the largest |x - x_zeta| / max|x_zeta| of the volume means over time",
    )
    for split, d in diffs.items():
        lines.append(f"{split}: " + ", ".join(f"{n} {v:.3e}" for n, v in d.items()))
    return "\n".join(lines)


def write_media(root: Path, splits: Sequence[str], command: Sequence[str]) -> None:
    """Copy ``splits.png`` and write ``splits.gif`` and ``provenance.json`` into
    :data:`MEDIA`: the splits' slabs side by side, warped by ``u`` and coloured by ``Ta``
    on one range, one frame per saved mechanics time (at most :data:`MAX_FRAMES`)."""
    from demo_io import animate, media_available, write_media_provenance

    from simcardemsx.results import read_resolved_settings, read_results
    from slab import post

    MEDIA.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "splits.png", MEDIA / "splits.png")
    if media_available():
        meshes, all_frames = [], []
        for split in splits:
            folder = root / split
            settings = read_resolved_settings(folder / "config.resolved.toml")
            functions = post.result_functions(settings)
            saved = read_results(folder, functions)
            frames: list[tuple[float, np.ndarray, np.ndarray]] = []
            scratch = root / f".frames_{split}"
            shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir()
            try:
                post.write_fields_and_traces(
                    scratch,
                    functions,
                    saved,
                    settings["mechanics"]["quadrature_degree"],
                    frames,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            meshes.append(functions["v"].function_space.mesh)
            all_frames.append(frames)
        n = min(len(f) for f in all_frames)
        if any(len(f) != n or f[0][0] != all_frames[0][0][0] for f in all_frames):
            raise SystemExit("the splits saved Ta at different times: no common frames")
        step = -(-n // MAX_FRAMES)
        shown = list(range(0, n, step))
        top = max(max(float(f[i][2].max()) for i in shown) for f in all_frames)
        animate(
            MEDIA / "splits.gif",
            meshes,
            lambda k: [all_frames[s][shown[k]][1:] for s in range(len(splits))],
            len(shown),
            titles=[f"{s} split" for s in splits],
            scalar="Ta [kPa]",
            clim=(0.0, max(top, 1e-3)),
            caption=lambda k: f"t = {all_frames[0][shown[k]][0]:.0f} ms",
            size=(640, 270),
            zoom_out=1.8,
        )
    write_media_provenance(MEDIA, {s: root / s for s in splits}, command)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--overwrite", action="store_true", help="Redo runs that are done.")
    parser.add_argument("--media", action="store_true", help="Also write media/.")
    parser.add_argument("--t-end", type=float, default=400.0, help="End time, ms.")
    parser.add_argument("--dt-mech", type=float, default=1.0, help="Mechanics step, ms.")
    parser.add_argument("--root", type=Path, default=SPLIT_FOLDER, help="Where the runs go.")
    args = parser.parse_args(argv)
    splits = tuple(SPLITS)
    root: Path = args.root.resolve()
    for split in pending_splits(root, splits, args.overwrite):
        run_split(split, root / split, args.dt_mech, args.t_end)
    traces = {split: read_traces(root / split) for split in splits}
    from demo_io import plots_available

    if plots_available():
        write_figure(root / "splits.png", traces)
    print(table(root, traces, splits))
    if args.media:
        command = [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)]
        write_media(root, splits, command)


if __name__ == "__main__":
    main()
