"""Report for the scheme comparison: tables (``report.md``) and figures (PNG).

Reads a directory of runs laid out as ``<root>/<geometry>/<group>/<scheme>/dt<dt:g>/``
(``geometry`` is ``element``, ``slab`` or ``biv``; ``group`` is the study on the element
and the split on the slab and BiV) and writes ``report.md`` and one PNG per figure.
Every metric is the one defined in the spec, section 5, and the pure functions are the
ones in ``metrics.py``. A missing run, reference or file shows as "—" and never stops
the report.

Usage::

    python3 report.py --root DIR [--out DIR]     # --out defaults to <root>/report
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scheme_comparison import metrics  # noqa: E402

SCHEMES = ("monolithic", "segregated", "stabilized")
#: Fixed colour per scheme (dataviz categorical slots 1-3, in fixed order).
COLORS = {"monolithic": "#2a78d6", "segregated": "#eb6834", "stabilized": "#1baf7a"}
MARKERS = {"monolithic": "o", "segregated": "s", "stabilized": "^"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dc"

#: Onset rule of the static element (gate 1's criterion): spread of λ over the points.
SPREAD_ONSET = 1e-3
#: Dynamic element: more reversals of mean λ than this is an instability (gate D2's bound).
MAX_REVERSALS = 2
#: Reference-based orders need the finest tested dt to be at least this many times the
#: reference's.
MIN_REF_RATIO = 5.0
#: A step difference below this is round-off (tests/conftest.py ``_FLAT``).
_FLAT = 1e-12
#: Newton residual norms at or below this are round-off and carry no convergence order.
NEWTON_FLOOR = 1e-12


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    """A CSV with a header as a dict of columns; empty when absent or unreadable."""
    try:
        data = np.genfromtxt(path, delimiter=",", names=True)
    except (OSError, ValueError):
        return {}
    data = np.atleast_1d(data)
    if data.dtype.names is None:
        return {}
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


@dataclass
class RunData:
    path: Path
    geometry: str
    group: str
    scheme: str
    dt: float
    meta: dict[str, Any]
    timings: dict[str, Any]
    steps: dict[str, np.ndarray]
    snaps: dict[str, np.ndarray]
    log: dict[str, np.ndarray]
    summary: dict[str, Any]
    launcher: dict[str, Any]

    @property
    def reached(self) -> bool | None:
        return self.meta.get("reached_t_end")

    @property
    def failed(self) -> bool:
        return self.reached is False


def load_run(path: Path, geometry: str, group: str, scheme: str) -> RunData:
    meta = _read_json(path / "run.json")
    try:
        dt = float(meta.get("dt_mech_ms", path.name.removeprefix("dt")))
    except ValueError:
        dt = float("nan")
    snaps: dict[str, np.ndarray] = {}
    if (path / "snapshots.npz").exists():
        try:
            with np.load(path / "snapshots.npz") as z:
                snaps = {k: np.asarray(z[k]) for k in z.files}
        except (OSError, ValueError):
            snaps = {}
    return RunData(
        path=path,
        geometry=geometry,
        group=group,
        scheme=scheme,
        dt=dt,
        meta=meta,
        timings=_read_json(path / "timings.json") or dict(meta.get("timings", {})),
        steps=_read_csv(path / "steps.csv"),
        snaps=snaps,
        log=_read_csv(path / "log.csv"),
        summary=_read_json(path / "summary.json"),
        launcher=_read_json(path / "launcher.json"),
    )


Ladders = dict[tuple[str, str], dict[str, dict[float, RunData]]]


def discover(root: Path) -> Ladders:
    """Group the runs by ``(geometry, group)``, then ``scheme``, then dt."""
    out: Ladders = {}
    for run_json in sorted(root.glob("*/*/*/dt*/run.json")):
        dt_dir = run_json.parent
        scheme_dir, group_dir = dt_dir.parent, dt_dir.parent.parent
        geometry = group_dir.parent.name
        run = load_run(dt_dir, geometry, group_dir.name, scheme_dir.name)
        out.setdefault((geometry, group_dir.name), {}).setdefault(run.scheme, {})[run.dt] = run
    return out


# --------------------------------------------------------------------------------------
# Series and errors
# --------------------------------------------------------------------------------------


@dataclass
class Series:
    """A run's fields on its snapshot times (or its mean λ on its step times)."""

    t: np.ndarray
    lmean: np.ndarray
    lam: np.ndarray | None = None
    ta: np.ndarray | None = None
    w: np.ndarray | None = None


def series_of(run: RunData) -> Series | None:
    s = run.snaps
    if s and {"t_ms", "weights", "lmbda"} <= s.keys():
        w = s["weights"]
        lam = s["lmbda"]
        lmean = np.array([metrics.weighted_mean(row, w) for row in lam])
        ta = s.get("tension_kPa")
        return Series(np.asarray(s["t_ms"], float), lmean, lam, ta, w)
    st = run.steps
    if "t_ms" in st and "lmbda_mean" in st:
        return Series(st["t_ms"], st["lmbda_mean"])
    return None


def _common(ta: np.ndarray, tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, ia, ib = np.intersect1d(np.round(ta, 6), np.round(tb, 6), return_indices=True)
    return ia, ib


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.all(np.isfinite(a.reshape(len(a), -1)), axis=1)
    return ok


@dataclass
class Err:
    lam_rms: float = float("nan")
    ta_rms: float = float("nan")  # not normalized
    lam_mean: float = float("nan")
    n_common: int = 0


def compare(a: Series, b: Series) -> Err:
    """Max over the common times of the RMS and mean differences of ``a`` from ``b``.

    Times at which either side is not finite (a failed run's last iterate) are left out.
    """
    ia, ib = _common(a.t, b.t)
    if len(ia) == 0:
        return Err()
    ok = _finite_rows(a.lmean[ia], b.lmean[ib])
    ia, ib = ia[ok], ib[ok]
    if len(ia) == 0:
        return Err()
    e = Err(n_common=len(ia))
    e.lam_mean = float(np.max(np.abs(a.lmean[ia] - b.lmean[ib])))
    if a.lam is not None and b.lam is not None and a.lam.shape[1] == b.lam.shape[1]:
        assert a.w is not None
        e.lam_rms = float(
            max(metrics.weighted_rms(a.lam[i] - b.lam[j], a.w) for i, j in zip(ia, ib)),
        )
        if a.ta is not None and b.ta is not None:
            e.ta_rms = float(
                max(metrics.weighted_rms(a.ta[i] - b.ta[j], a.w) for i, j in zip(ia, ib)),
            )
    return e


def ta_norm(ref: Series) -> float:
    """The reference's max over time of the RMS of ``Ta``: what ``Ta`` errors divide by."""
    if ref.ta is None or ref.w is None:
        return float("nan")
    return float(max(metrics.weighted_rms(row, ref.w) for row in ref.ta))


def richardson_series(fine: Series, coarse: Series, ratio: float) -> Series | None:
    ia, ib = _common(fine.t, coarse.t)
    if len(ia) == 0:
        return None
    lmean = metrics.richardson(fine.lmean[ia], coarse.lmean[ib], ratio)
    out = Series(fine.t[ia], lmean)
    if fine.lam is not None and coarse.lam is not None and fine.lam.shape == coarse.lam.shape:
        out.lam = metrics.richardson(fine.lam[ia], coarse.lam[ib], ratio)
        out.w = fine.w
        if fine.ta is not None and coarse.ta is not None:
            out.ta = metrics.richardson(fine.ta[ia], coarse.ta[ib], ratio)
    return out


@dataclass
class Reference:
    name: str
    series: Series
    dt: float  # the reference's effective dt; 0 for an extrapolation
    run_dt: float | None = None  # the dt of the run that is the reference, if it is one
    logs: dict[str, np.ndarray] = field(default_factory=dict)  # BiV log.csv columns


def finest_monolithic(lad: dict[str, dict[float, RunData]]) -> RunData | None:
    mono = lad.get("monolithic", {})
    usable = [r for r in mono.values() if series_of(r) is not None]
    return min(usable, key=lambda r: r.dt) if usable else None


LOG_FIELDS = ("V_LV_mL", "V_RV_mL", "p_LV_mmHg", "p_RV_mmHg")


def make_references(geometry: str, lad: dict[str, dict[float, RunData]]) -> list[Reference]:
    refs: list[Reference] = []
    fin = finest_monolithic(lad)
    if fin is not None:
        s = series_of(fin)
        assert s is not None
        refs.append(Reference(f"monolithic dt {fin.dt:g} ms", s, fin.dt, fin.dt, dict(fin.log)))
    if geometry == "biv":
        mono = sorted(
            (r for r in lad.get("monolithic", {}).values() if series_of(r) is not None),
            key=lambda r: r.dt,
        )
        if len(mono) >= 2:
            fine, coarse = mono[0], mono[1]
            ratio = coarse.dt / fine.dt
            sf, sc = series_of(fine), series_of(coarse)
            assert sf is not None and sc is not None
            rs = richardson_series(sf, sc, ratio)
            if rs is not None:
                logs: dict[str, np.ndarray] = {}
                if fine.log and coarse.log and "t_ms" in fine.log and "t_ms" in coarse.log:
                    ia, ib = _common(fine.log["t_ms"], coarse.log["t_ms"])
                    logs["t_ms"] = fine.log["t_ms"][ia]
                    for k in LOG_FIELDS:
                        if k in fine.log and k in coarse.log:
                            logs[k] = metrics.richardson(fine.log[k][ia], coarse.log[k][ib], ratio)
                refs.append(
                    Reference(
                        f"Richardson of monolithic dt {coarse.dt:g} and {fine.dt:g} ms",
                        rs,
                        0.0,
                        None,
                        logs,
                    ),
                )
    return refs


# --------------------------------------------------------------------------------------
# Onsets
# --------------------------------------------------------------------------------------


def kind_of(geometry: str, group: str) -> str:
    if geometry == "element":
        return "dynamic" if group.startswith("dynamic") else "static"
    return "3d"


def reversals(trace: np.ndarray) -> int:
    """Changes of direction of a trace, ignoring differences below round-off.

    Copied by value from ``tests/conftest.py::_reversals``, the definition the dynamic
    gates use.
    """
    d = np.diff(trace)
    d = d[np.abs(d) >= _FLAT]
    return int(np.count_nonzero(np.diff(np.sign(d))))


def _floor_col(eps: float) -> str:
    return f"reversal_fraction_{eps:g}"


def choose_epsilon(lad: dict[str, dict[float, RunData]]) -> tuple[float | None, str]:
    """Smallest floor at which no monolithic run of the group has an onset."""
    mono = [r for r in lad.get("monolithic", {}).values() if r.steps]
    if not mono:
        return None, "no monolithic run, so no floor can be fixed"
    for eps in metrics.FLOORS_PER_MS:
        col = _floor_col(eps)
        if all(col in r.steps and metrics.onset_index(list(r.steps[col])) is None for r in mono):
            return eps, ""
    return None, (
        "even the largest floor "
        f"({metrics.FLOORS_PER_MS[-1]:g}) trips a monolithic run; no floor is chosen and "
        "the 3D onset is not reported"
    )


@dataclass
class Onset:
    onset_ms: float | None
    value: float  # the criterion's own value: spread, reversals or max fraction
    flagged: bool | None  # None when it cannot be judged


def onset_of(run: RunData, kind: str, eps: float | None) -> Onset:
    st = run.steps
    nan = float("nan")
    if not st or "t_ms" not in st:
        return Onset(None, nan, None)
    t = st["t_ms"]
    if kind == "static":
        spread = st["lmbda_max"] - st["lmbda_min"]
        bad = np.flatnonzero(spread > SPREAD_ONSET)
        return Onset(float(t[bad[0]]) if len(bad) else None, float(np.nanmax(spread)), True)
    if kind == "dynamic":
        n = reversals(st["lmbda_mean"])
        return Onset(None, float(n), n > MAX_REVERSALS)
    col = _floor_col(eps) if eps is not None else ""
    if eps is None or col not in st:
        return Onset(None, nan, None)
    frac = st[col]
    i = metrics.onset_index(list(frac))
    return Onset(float(t[i]) if i is not None else None, float(np.nanmax(frac)), True)


def onset_text(o: Onset, kind: str) -> str:
    if o.flagged is None:
        return "n/a"
    if kind == "dynamic":
        return f"unstable ({o.value:g} reversals)" if o.flagged else "none"
    return f"{o.onset_ms:g}" if o.onset_ms is not None else "none"


# --------------------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------------------

DASH = "—"


def fnum(x: Any, spec: str = ".3e") -> str:
    if x is None:
        return DASH
    try:
        if not np.isfinite(x):
            return DASH
    except TypeError:
        return str(x)
    return format(x, spec)


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_no data_\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def sorted_runs(lad: dict[str, dict[float, RunData]]) -> list[RunData]:
    """Runs ordered by scheme (fixed order), then coarse dt first."""
    out: list[RunData] = []
    for s in SCHEMES:
        out += [lad[s][dt] for dt in sorted(lad.get(s, {}), reverse=True)]
    for s in sorted(set(lad) - set(SCHEMES)):
        out += [lad[s][dt] for dt in sorted(lad[s], reverse=True)]
    return out


def flag(run: RunData) -> str:
    return "*" if run.failed else ""


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def accuracy_tables(
    lad: dict[str, dict[float, RunData]],
    refs: list[Reference],
) -> tuple[str, dict[tuple[str, float, str], Err]]:
    """The accuracy tables, and the errors against the first reference for the figures."""
    md = []
    store: dict[tuple[str, float, str], Err] = {}
    mono = lad.get("monolithic", {})
    runs = sorted_runs(lad)
    if not refs:
        md.append(
            "No monolithic run to serve as reference: the reference-based columns are empty.\n",
        )
    header = [
        "scheme",
        "dt (ms)",
        "coupling: RMS λ",
        "coupling: RMS Ta / ref max RMS Ta",
        "coupling: max abs Δ mean λ",
        "vs ref: RMS λ",
        "vs ref: RMS Ta / ref max RMS Ta",
        "vs ref: max abs Δ mean λ",
    ]
    norm0 = ta_norm(refs[0].series) if refs else float("nan")
    for ref_i, ref in enumerate(refs):
        norm = ta_norm(ref.series)
        rows = []
        for run in runs:
            s = series_of(run)
            cpl = Err()
            if run.scheme != "monolithic" and run.dt in mono and s is not None:
                sm = series_of(mono[run.dt])
                if sm is not None:
                    cpl = compare(s, sm)
            e = Err()
            is_ref = ref.run_dt is not None and run.scheme == "monolithic" and run.dt == ref.run_dt
            if s is not None and not is_ref:
                e = compare(s, ref.series)
            if ref_i == 0:
                store[(run.scheme, run.dt, "main")] = e
            rows.append(
                [
                    run.scheme,
                    f"{run.dt:g}" + flag(run),
                    fnum(cpl.lam_rms) if run.scheme != "monolithic" else "0 (by definition)",
                    fnum(cpl.ta_rms / norm) if run.scheme != "monolithic" else "0 (by definition)",
                    fnum(cpl.lam_mean) if run.scheme != "monolithic" else "0 (by definition)",
                    "0 (is the reference)" if is_ref else fnum(e.lam_rms),
                    "0 (is the reference)" if is_ref else fnum(e.ta_rms / norm),
                    "0 (is the reference)" if is_ref else fnum(e.lam_mean),
                ],
            )
        md.append(
            f"**Reference: {ref.name}.** (`*`: the run failed; the errors cover the "
            "finite common times only.)\n",
        )
        md.append(table(header, rows))
    del norm0
    # Orders
    if refs:
        md.append(orders_tables(lad, refs))
    return "\n".join(md), store


def _order_pairs(dts: list[float], errs: list[float]) -> list[tuple[float, float, float]]:
    """Observed order between successive dts (coarse to fine) whose errors are positive."""
    ok = [(d, e) for d, e in zip(dts, errs) if np.isfinite(e) and e > 0]
    if len(ok) < 2:
        return []
    p = metrics.reference_orders([d for d, _ in ok], [e for _, e in ok])
    return [(ok[i][0], ok[i + 1][0], p[i]) for i in range(len(ok) - 1)]


def orders_tables(lad: dict[str, dict[float, RunData]], refs: list[Reference]) -> str:
    md = []
    rows = []
    for ref in refs:
        for scheme in SCHEMES:
            runs = lad.get(scheme, {})
            tested = sorted(
                (
                    dt
                    for dt in runs
                    if not runs[dt].failed
                    and not (ref.run_dt is not None and scheme == "monolithic" and dt == ref.run_dt)
                ),
                reverse=True,
            )
            if not tested:
                continue
            finest = min(tested)
            ratio = finest / ref.dt if ref.dt > 0 else float("inf")
            if ratio < MIN_REF_RATIO:
                rows.append(
                    [
                        ref.name,
                        scheme,
                        f"finest {finest:g} / ref {ref.dt:g} = {ratio:.3g} < {MIN_REF_RATIO:g}",
                        DASH,
                        DASH,
                        DASH,
                    ],
                )
                continue
            errs = {dt: (series_of(runs[dt]),) for dt in tested}
            ec = {
                dt: compare(s[0], ref.series) if s[0] is not None else Err()
                for dt, s in errs.items()
            }
            for q, label in (("lam_rms", "RMS λ"), ("ta_rms", "RMS Ta"), ("lam_mean", "mean λ")):
                pairs = _order_pairs(tested, [getattr(ec[d], q) for d in tested])
                for a, b, p in pairs:
                    rows.append([ref.name, scheme, f"{a:g} → {b:g}", label, fnum(p, ".2f"), ""])
    md.append(
        "**Orders against the reference.** Failed runs are left out (their errors cover a "
        "truncated "
        "window). Reported only where the finest tested dt is at "
        f"least {MIN_REF_RATIO:g}× the reference's (the extrapolation counts as dt 0).\n",
    )
    md.append(
        table(
            ["reference", "scheme", "dt pair (ms)", "quantity", "observed order"],
            [r[:5] for r in rows],
        ),
    )
    # Self-convergence
    srows = []
    for scheme in SCHEMES:
        runs = lad.get(scheme, {})
        dts = sorted((d for d in runs if not runs[d].failed), reverse=True)
        for i in range(len(dts) - 2):
            d0, d1, d2 = dts[i : i + 3]
            s0, s1, s2 = (series_of(runs[d]) for d in (d0, d1, d2))
            if s0 is None or s1 is None or s2 is None:
                continue
            c, f = compare(s0, s1), compare(s1, s2)
            srows.append(
                [
                    scheme,
                    f"{d0:g}, {d1:g}, {d2:g}",
                    fnum(metrics.self_convergence_order((d0, d1, d2), c.lam_rms, f.lam_rms), ".2f"),
                    fnum(
                        metrics.self_convergence_order((d0, d1, d2), c.lam_mean, f.lam_mean),
                        ".2f",
                    ),
                ],
            )
    md.append(
        "**Self-convergence orders** (failed runs left out; from successive differences, "
        "no reference; "
        '"—" where undefined).\n',
    )
    md.append(table(["scheme", "dts (ms)", "order, RMS λ", "order, mean λ"], srows))
    return "\n".join(md)


def stability_table(
    lad: dict[str, dict[float, RunData]],
    kind: str,
    eps: float | None,
    eps_note: str,
) -> str:
    rule = {
        "static": f"onset: first t with λ_max − λ_min > {SPREAD_ONSET:g} (gate 1's criterion); "
        "criterion value: max spread",
        "dynamic": f"onset: more than {MAX_REVERSALS} reversals of mean λ (gate D2's bound); "
        "criterion value: reversals",
        "3d": "onset: first of 3 consecutive steps with reversal fraction > 0.5; "
        "criterion value: max reversal fraction",
    }[kind]
    lines = [rule + "\n"]
    if kind == "3d":
        if eps is not None:
            lines.append(
                f"**ε = {eps:g} per ms**, the smallest floor at which no monolithic run of this "
                "group has an onset (column `reversal_fraction_"
                f"{eps:g}`).\n",
            )
        else:
            lines.append(f"**No ε chosen: {eps_note}.**\n")
    rows = []
    for run in sorted_runs(lad):
        o = onset_of(run, kind, eps)
        rows.append(
            [
                run.scheme,
                f"{run.dt:g}",
                run.reached,
                fnum(run.meta.get("t_fail_ms"), "g"),
                onset_text(o, kind),
                fnum(o.value, ".3g"),
                (run.meta.get("failure") or DASH),
            ],
        )
    lines.append(
        table(
            [
                "scheme",
                "dt (ms)",
                "reached_t_end",
                "t_fail_ms",
                "onset (ms)",
                "criterion value",
                "failure",
            ],
            rows,
        ),
    )
    return "\n".join(lines)


def consistency_table(lad: dict[str, dict[float, RunData]]) -> str:
    runs = lad.get("stabilized", {})
    rows = []
    prev: tuple[float, float] | None = None
    for dt in sorted(runs, reverse=True):
        st = runs[dt].steps
        if not st or "KaDl_max_kPa" not in st:
            rows.append([f"{dt:g}", DASH, DASH])
            continue
        v = float(np.nanmax(np.abs(st["KaDl_max_kPa"])) / np.nanmax(np.abs(st["Ta_mean_kPa"])))
        ratio = DASH
        if prev is not None and v > 0:
            ratio = f"{prev[1] / v:.3g} (dt ratio {prev[0] / dt:.3g})"
        rows.append([f"{dt:g}" + flag(runs[dt]), fnum(v), ratio])
        prev = (dt, v)
    return (
        "Stabilized scheme: max abs Ka·Δλ / max mean Ta. R&Q §4.1 say it is O(dt), so the ratio "
        "between successive dts should equal the dt ratio.\n\n"
        + table(["dt (ms)", "max abs KaΔλ / max Ta_mean", "ratio to previous (coarser) dt"], rows)
    )


def cost_table(lad: dict[str, dict[float, RunData]], geometry: str) -> str:
    rows = []
    for run in sorted_runs(lad):
        nw = run.meta.get("newton", {}) or {}
        tm = run.timings
        total = nw.get("total", tm.get("newton_its"))
        mech = tm.get("mech_s")
        per = mech / total if isinstance(mech, (int, float)) and total else None
        rows.append(
            [
                run.scheme,
                f"{run.dt:g}" + flag(run),
                fnum(total, "g"),
                fnum(nw.get("mean"), ".3g"),
                fnum(nw.get("max"), "g"),
                fnum(mech, ".3g"),
                fnum(per, ".3g"),
                fnum(tm.get("setup_s"), ".3g"),
                fnum(tm.get("loop_s"), ".3g"),
                fnum(tm.get("total_s"), ".3g"),
            ],
        )
    note = (
        "Wall times are serial. The first run of each (geometry, scheme, dt) is JIT-cold, so "
        "its `setup_s` includes compilation. "
    )
    if geometry == "biv":
        note += (
            "Each BiV run compiles its own form, because dt is compiled into the "
            "`DynamicProblem`, so every `setup_s` here is cold. "
        )
    else:
        note += (
            "The slab and the element share a compiled form per scheme, so only the first "
            "dt of each scheme pays the compilation. "
        )
    note += (
        "The segregated schemes evaluate the ODE step at every assembly (spec §3); a "
        "production code would evaluate it once per step.\n"
    )
    return (
        note
        + "\n"
        + table(
            [
                "scheme",
                "dt (ms)",
                "Newton total",
                "mean",
                "max",
                "mech_s",
                "mech_s / Newton it.",
                "setup_s",
                "loop_s",
                "total_s",
            ],
            rows,
        )
    )


def threshold_of(run: RunData) -> float:
    reg = run.meta.get("regime")
    if not reg:
        return float("nan")
    try:
        return metrics.naive_threshold_kPa(
            run.dt,
            Kp_kPa=reg["Kp_kPa"],
            eta_Pa_s=reg["eta_Pa_s"],
            rho_kg_m3=reg["rho_kg_m3"],
            h_m=reg["h_m"],
        )
    except (KeyError, TypeError, ZeroDivisionError):
        return float("nan")


def ka_max_of(lad: dict[str, dict[float, RunData]], dt: float) -> float:
    """Peak Ka from the stabilized run of this dt; otherwise from any run that logged one."""
    order = ["stabilized"] + [s for s in lad if s != "stabilized"]
    for s in order:
        r = lad.get(s, {}).get(dt)
        if r is not None and r.steps.get("Ka_max_kPa") is not None and len(r.steps["Ka_max_kPa"]):
            v = float(np.nanmax(r.steps["Ka_max_kPa"]))
            if v > 0 or s == "stabilized":
                return v
    return float("nan")


def regime_table(
    lad: dict[str, dict[float, RunData]],
    kind: str,
    eps: float | None,
) -> tuple[str, dict[float, tuple[float, float]]]:
    dts = sorted({dt for s in lad.values() for dt in s}, reverse=True)
    rows = []
    pts: dict[float, tuple[float, float]] = {}
    for dt in dts:
        any_run = next(s[dt] for s in lad.values() if dt in s)
        thr = threshold_of(any_run)
        ka = ka_max_of(lad, dt)
        pts[dt] = (ka, thr)
        pred = (
            DASH
            if not (np.isfinite(ka) and np.isfinite(thr))
            else ("unstable" if ka > thr else "stable")
        )
        seg = lad.get("segregated", {}).get(dt)
        if seg is None:
            obs = DASH
        else:
            o = onset_of(seg, kind, eps)
            bits = []
            if seg.failed:
                bits.append(f"failed at {fnum(seg.meta.get('t_fail_ms'), 'g')} ms")
            if o.flagged and (o.onset_ms is not None or (kind == "dynamic" and o.flagged)):
                bits.append("onset " + onset_text(o, kind))
            obs = "; ".join(bits) if bits else ("stable" if o.flagged is not None else "n/a")
        rows.append([f"{dt:g}", fnum(ka, ".4g"), fnum(thr, ".4g"), pred, obs])
    txt = (
        "Indicator, not a criterion: the naive scheme is predicted unstable where "
        "max Ka exceeds Kp + η/dt + ρh²/dt² (Kp is a uniaxial Holzapfel-Ogden estimate; "
        "parameters from `run.json`'s `regime`).\n\n"
        + table(
            ["dt (ms)", "max Ka_max (kPa)", "threshold (kPa)", "predicted naive", "observed naive"],
            rows,
        )
    )
    return txt, pts


def newton_order_table(lad: dict[str, dict[float, RunData]]) -> str:
    rows = []
    for run in sorted_runs(lad):
        res = run.meta.get("newton_residuals")
        if not res:
            continue
        res = [float(x) for x in res]
        order = float("nan")
        # Residuals at round-off (below NEWTON_FLOOR) carry no order; drop them.
        above = [x for x in res if x > NEWTON_FLOOR]
        if len(above) >= 3:
            order = float(np.log(above[-1] / above[-2]) / np.log(above[-2] / above[-3]))
        rows.append(
            [
                run.scheme,
                f"{run.dt:g}",
                ", ".join(f"{x:.2e}" for x in res),
                fnum(order, ".2f"),
            ],
        )
    return (
        "Residual history of one mid-twitch step per run, and the order estimated from the last "
        f"three residuals above {NEWTON_FLOOR:g} (round-off is left out), "
        "log(r₃/r₂)/log(r₂/r₁); 2 is quadratic. With so few residuals it is a rough estimate.\n\n"
        + table(["scheme", "dt (ms)", "SNES residual norms", "observed order"], rows)
    )


def biv_tables(lad: dict[str, dict[float, RunData]], refs: list[Reference]) -> str:
    md = []
    runs = sorted_runs(lad)
    for vent in ("LV", "RV"):
        v_key, p_key = f"V_{vent}_mL", f"p_{vent}_mmHg"
        for ref in refs:
            rows = []
            for run in runs:
                lg = run.log
                if not lg or "t_ms" not in lg or not ref.logs or "t_ms" not in ref.logs:
                    dv = dp = float("nan")
                else:
                    ia, ib = _common(lg["t_ms"], ref.logs["t_ms"])
                    is_ref = (
                        ref.run_dt is not None
                        and run.scheme == "monolithic"
                        and run.dt == ref.run_dt
                    )

                    def md_(
                        k: str,
                        ia: np.ndarray = ia,
                        ib: np.ndarray = ib,
                        lg: Any = lg,
                    ) -> float:
                        if k not in lg or k not in ref.logs or len(ia) == 0:
                            return float("nan")
                        d = np.abs(lg[k][ia] - ref.logs[k][ib])
                        d = d[np.isfinite(d)]
                        return float(np.max(d)) if len(d) else float("nan")

                    dv, dp = (0.0, 0.0) if is_ref else (md_(v_key), md_(p_key))
                rows.append([run.scheme, f"{run.dt:g}" + flag(run), fnum(dv), fnum(dp)])
            md.append(
                f"**{vent}: max abs V − V_ref and p − p_ref on the log's grid, reference: "
                f"{ref.name}.**\n",
            )
            md.append(
                table(
                    ["scheme", "dt (ms)", "max abs V − V_ref (mL)", "max abs p − p_ref (mmHg)"],
                    rows,
                ),
            )
        rows = []
        for run in runs:
            s = (run.summary or {}).get(vent, {})
            opens = s.get("outflow_valve_open") or []
            ejected = sum(float(o.get("ejected_mL", 0.0)) for o in opens) if s else None
            rows.append(
                [
                    run.scheme,
                    f"{run.dt:g}" + flag(run),
                    fnum(ejected, ".4g") if s else DASH,
                    str(len(opens)) if s else DASH,
                    fnum(s.get("peak_p_mmHg"), ".4g"),
                    fnum(s.get("t_peak_p_ms"), ".4g"),
                    fnum(s.get("V_range_mL"), ".4g"),
                ],
            )
        md.append(
            f"**{vent}: ejected volume and pressure peak** (from `summary.json`; ejected is "
            "the sum over the outflow valve's open intervals).\n",
        )
        md.append(
            table(
                [
                    "scheme",
                    "dt (ms)",
                    "ejected (mL)",
                    "open intervals",
                    "peak p (mmHg)",
                    "t of peak p (ms)",
                    "V range (mL)",
                ],
                rows,
            ),
        )
    drift = [
        [
            r.scheme,
            f"{r.dt:g}",
            fnum((r.summary or {}).get("max_conservation_drift"), ".3g"),
            fnum((r.summary or {}).get("peak_Ta_mean_kPa"), ".4g"),
        ]
        for r in runs
    ]
    md.append("**Conservation and peak tension.**\n")
    md.append(table(["scheme", "dt (ms)", "max conservation drift", "peak mean Ta (kPa)"], drift))
    return "\n".join(md)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _style(ax: Any, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _legend(ax: Any) -> None:
    h, _ = ax.get_legend_handles_labels()
    if h:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)


def _save(fig: Any, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130, facecolor="white")
    plt.close(fig)
    return name


def _img(name: str | None, caption: str) -> str:
    return f"![{caption}]({name})\n\n*{caption}*\n" if name else ""


def fig_timeseries(lad: dict[str, dict[float, RunData]], out: Path, stem: str) -> str | None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    any_line = False
    for scheme in SCHEMES:
        runs = lad.get(scheme, {})
        usable = [dt for dt in runs if runs[dt].steps.get("t_ms") is not None]
        if not usable:
            continue
        picks = [(max(usable), "--"), (min(usable), "-")] if len(usable) > 1 else [(usable[0], "-")]
        for dt, ls in picks:
            st = runs[dt].steps
            for ax, key in zip(axes, ("lmbda_mean", "Ta_mean_kPa")):
                if key in st:
                    ax.plot(
                        st["t_ms"],
                        st[key],
                        ls,
                        color=COLORS[scheme],
                        lw=1.6,
                        label=f"{scheme}, dt {dt:g} ms",
                    )
                    any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _style(axes[0], "t (ms)", "mean λ (-)", "Mean stretch")
    _style(axes[1], "t (ms)", "mean Ta (kPa)", "Mean active tension")
    _legend(axes[1])
    return _save(fig, out, f"{stem}_timeseries.png")


def fig_error_dt(
    lad: dict[str, dict[float, RunData]],
    store: dict[tuple[str, float, str], Err],
    refs: list[Reference],
    out: Path,
    stem: str,
) -> str | None:
    if not refs:
        return None
    norm = ta_norm(refs[0].series)
    panels: list[tuple[str, Callable[[Err], float]]] = [
        ("RMS λ error", lambda e: e.lam_rms),
        ("RMS Ta error / ref max RMS Ta", lambda e: e.ta_rms / norm),
        ("max abs Δ mean λ", lambda e: e.lam_mean),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    any_line = False
    for ax, (label, f) in zip(axes, panels):
        for scheme in SCHEMES:
            pts = sorted(
                (dt, f(store[(scheme, dt, "main")]))
                for dt in lad.get(scheme, {})
                if (scheme, dt, "main") in store
            )
            pts = [(d, e) for d, e in pts if np.isfinite(e) and e > 0]
            if pts:
                ax.loglog(
                    *zip(*pts),
                    MARKERS[scheme] + "-",
                    color=COLORS[scheme],
                    lw=1.6,
                    ms=6,
                    label=scheme,
                )
                any_line = True
        xs = ax.get_xlim()
        _style(ax, "dt (ms)", label + " vs " + refs[0].name.split(" dt")[0].lower())
        ax.set_xlim(xs)
    if not any_line:
        plt.close(fig)
        return None
    _legend(axes[0])
    return _save(fig, out, f"{stem}_error_vs_dt.png")


def fig_work_precision(
    lad: dict[str, dict[float, RunData]],
    store: dict[tuple[str, float, str], Err],
    out: Path,
    stem: str,
) -> str | None:
    fig, ax = plt.subplots(figsize=(5.6, 4))
    any_line = False
    for scheme in SCHEMES:
        pts = []
        for dt, run in lad.get(scheme, {}).items():
            e = store.get((scheme, dt, "main"))
            mech = run.timings.get("mech_s")
            if e is not None and mech and np.isfinite(e.lam_rms) and e.lam_rms > 0:
                pts.append((mech, e.lam_rms, dt))
        pts.sort(key=lambda p: p[2], reverse=True)
        if pts:
            ax.loglog(
                [p[0] for p in pts],
                [p[1] for p in pts],
                MARKERS[scheme] + "-",
                color=COLORS[scheme],
                lw=1.6,
                ms=6,
                label=scheme,
            )
            for mech_t, err_v, dt_p in pts:
                ax.annotate(
                    f"{dt_p:g}",
                    (mech_t, err_v),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=INK_2,
                )
            any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _style(
        ax,
        "mechanics wall time mech_s (s); points labelled with dt (ms)",
        "RMS λ error vs reference",
        "Work-precision",
    )
    _legend(ax)
    return _save(fig, out, f"{stem}_work_precision.png")


def fig_newton(lad: dict[str, dict[float, RunData]], out: Path, stem: str) -> str | None:
    counts: dict[float, int] = {}
    for runs in lad.values():
        for dt, run_ in runs.items():
            if run_.steps.get("newton_iterations") is not None:
                counts[dt] = counts.get(dt, 0) + 1
    if not counts:
        return None
    dt = max(counts, key=lambda d: (counts[d], d))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for scheme in SCHEMES:
        one = lad.get(scheme, {}).get(dt)
        if one is not None and "newton_iterations" in one.steps:
            ax.plot(
                one.steps["t_ms"],
                one.steps["newton_iterations"],
                MARKERS[scheme] + "-",
                color=COLORS[scheme],
                lw=1.2,
                ms=4,
                label=scheme,
            )
    _style(ax, "t (ms)", "Newton iterations per step", f"Newton iterations, dt {dt:g} ms")
    _legend(ax)
    return _save(fig, out, f"{stem}_newton.png")


def fig_regime(pts: dict[float, tuple[float, float]], out: Path, stem: str) -> str | None:
    good = {dt: p for dt, p in pts.items() if np.isfinite(p[0])}
    if not good:
        return None
    dts = sorted(good)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.loglog(
        dts,
        [good[d][0] for d in dts],
        "^-",
        color=COLORS["stabilized"],
        lw=1.6,
        ms=6,
        label="max Ka_max over the run",
    )
    thr = [(d, good[d][1]) for d in dts if np.isfinite(good[d][1])]
    if thr:
        ax.loglog(*zip(*thr), "--", color=INK_2, lw=1.4, label="Kp + η/dt + ρh²/dt²")
    _style(ax, "dt (ms)", "active stiffness (kPa)", "Ka_max against the naive threshold")
    _legend(ax)
    return _save(fig, out, f"{stem}_regime.png")


def fig_biv_loops(
    lad: dict[str, dict[float, RunData]],
    out: Path,
    stem: str,
) -> tuple[str | None, str | None]:
    dts = [dt for dt in (2.0, 0.5) if any(dt in s and s[dt].log for s in lad.values())]
    if not dts:
        return None, None
    names = []
    for kind in ("pv", "pt"):
        fig, axes = plt.subplots(len(dts), 2, figsize=(9, 3.4 * len(dts)), squeeze=False)
        for i, dt in enumerate(dts):
            for j, vent in enumerate(("LV", "RV")):
                ax = axes[i][j]
                for scheme in SCHEMES:
                    r = lad.get(scheme, {}).get(dt)
                    if r is None or not r.log:
                        continue
                    lg = r.log
                    vk, pk = f"V_{vent}_mL", f"p_{vent}_mmHg"
                    if vk not in lg or pk not in lg:
                        continue
                    if kind == "pv":
                        ax.plot(lg[vk], lg[pk], "-", color=COLORS[scheme], lw=1.6, label=scheme)
                    else:
                        ax.plot(lg["t_ms"], lg[pk], "-", color=COLORS[scheme], lw=1.6, label=scheme)
                if kind == "pv":
                    _style(ax, f"V_{vent} (mL)", f"p_{vent} (mmHg)", f"{vent}, dt {dt:g} ms")
                else:
                    _style(ax, "t (ms)", f"p_{vent} (mmHg)", f"{vent}, dt {dt:g} ms")
                _legend(ax)
        names.append(
            _save(fig, out, f"{stem}_{'pv_loops' if kind == 'pv' else 'pressure_time'}.png"),
        )
    return names[0], names[1]


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def group_section(
    geometry: str,
    group: str,
    lad: dict[str, dict[float, RunData]],
    out: Path,
) -> str:
    kind = kind_of(geometry, group)
    stem = f"{geometry}_{group}"
    refs = make_references(geometry, lad)
    eps, eps_note = choose_epsilon(lad) if kind == "3d" else (None, "")
    n = sum(len(v) for v in lad.values())
    md = [
        f"## {geometry} / {group}\n",
        f"{n} runs: "
        + ", ".join(
            f"{s} (dt {', '.join(f'{d:g}' for d in sorted(lad[s], reverse=True))})"
            for s in sorted(lad, key=lambda s: SCHEMES.index(s) if s in SCHEMES else 9)
        )
        + ".\n",
    ]

    def guarded(title: str, fn: Callable[[], str]) -> None:
        md.append(f"### {title}\n")
        try:
            md.append(fn())
        except Exception:  # never crash the report; say so in it
            tb = traceback.format_exc()
            print(f"[{stem}] {title} failed:\n{tb}", file=sys.stderr)
            md.append(f"**This section failed to build:** `{tb.strip().splitlines()[-1]}`\n")

    store: dict[tuple[str, float, str], Err] = {}

    def acc() -> str:
        text, st = accuracy_tables(lad, refs)
        store.update(st)
        return text

    guarded("Accuracy", acc)
    guarded("Stability", lambda: stability_table(lad, kind, eps, eps_note))
    guarded("Consistency of the stabilization", lambda: consistency_table(lad))
    guarded("Cost", lambda: cost_table(lad, geometry))
    pts_holder: dict[float, tuple[float, float]] = {}

    def reg() -> str:
        text, pts = regime_table(lad, kind, eps)
        pts_holder.update(pts)
        return text

    guarded("Regime", reg)
    if kind != "3d":
        guarded("Newton convergence order", lambda: newton_order_table(lad))
    if geometry == "biv":
        guarded("Ventricles", lambda: biv_tables(lad, refs))

    figs: list[str] = []

    def add(label: str, fn: Callable[[], str | None], cap: str) -> None:
        try:
            name = fn()
        except Exception:
            tb = traceback.format_exc()
            print(f"[{stem}] figure {label} failed:\n{tb}", file=sys.stderr)
            figs.append(f"**Figure `{label}` failed to build:** `{tb.strip().splitlines()[-1]}`\n")
            plt.close("all")
            return
        figs.append(_img(name, cap) if name else f"_Figure `{label}`: no data._\n")

    add(
        "timeseries",
        lambda: fig_timeseries(lad, out, stem),
        "Mean λ and mean Ta in time; dashed: the coarsest dt of the scheme, solid: the finest.",
    )
    add(
        "error_vs_dt",
        lambda: fig_error_dt(lad, store, refs, out, stem),
        "Error against the reference, against dt (log-log), one line per scheme.",
    )
    add(
        "work_precision",
        lambda: fig_work_precision(lad, store, out, stem),
        "Work-precision: RMS λ error against the mechanics wall time.",
    )
    add("newton", lambda: fig_newton(lad, out, stem), "Newton iterations per step.")
    add(
        "regime",
        lambda: fig_regime(pts_holder, out, stem),
        "Peak Ka against the naive scheme's threshold, per dt.",
    )
    if geometry == "biv":
        try:
            pv, pt = fig_biv_loops(lad, out, stem)
        except Exception:
            print(f"[{stem}] BiV figures failed:\n{traceback.format_exc()}", file=sys.stderr)
            pv = pt = None
            plt.close("all")
        figs.append(
            _img(pv, "PV loops per scheme at dt 2 and 0.5 ms.")
            if pv
            else "_Figure `pv_loops`: no data._\n",
        )
        figs.append(
            _img(pt, "Ventricular pressure in time per scheme at dt 2 and 0.5 ms.")
            if pt
            else "_Figure `pressure_time`: no data._\n",
        )
    md.append("### Figures\n")
    md.extend(figs)
    return "\n".join(md)


def build_report(root: Path, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    ladders = discover(root)
    order = {"element": 0, "slab": 1, "biv": 2}
    md = [
        "# Scheme comparison report\n",
        f"Root: `{root}`. Generated {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M} UTC.\n",
        "Metrics are those of the spec, §5. Runs are compared at their common snapshot times; the "
        "BiV's V and p on `log.csv`'s grid. A `*` after a dt marks a run that did not reach "
        '`t_end`. A "—" is a value that does not exist (missing run, reference or file, or '
        "undefined).\n",
    ]
    if not ladders:
        md.append(
            f"**No runs found under `{root}`** (expected "
            "`<geometry>/<group>/<scheme>/dt<dt>/run.json`).\n",
        )
    for geometry, group in sorted(ladders, key=lambda k: (order.get(k[0], 9), k[1])):
        try:
            md.append(group_section(geometry, group, ladders[(geometry, group)], out))
        except Exception:
            tb = traceback.format_exc()
            print(f"[{geometry}/{group}] failed:\n{tb}", file=sys.stderr)
            md.append(
                f"## {geometry} / {group}\n\n**Failed to build:** "
                f"`{tb.strip().splitlines()[-1]}`\n",
            )
    text = "\n".join(md)
    (out / "report.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="directory of runs")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/report")
    args = ap.parse_args()
    out = args.out or args.root / "report"
    build_report(args.root, out)
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
