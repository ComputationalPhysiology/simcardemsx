"""The scheme comparison's report, ``numerical_experiments/scheme_comparison/report.py``.

Every run here is fabricated, in the real file formats (``steps.csv`` with
``record.COLUMNS``, ``snapshots.npz``, ``run.json``, ``timings.json``, and for the BiV
``log.csv`` and ``summary.json``) and the real layout
``<root>/<geometry>/<group>/<scheme>/dt<dt:g>/``, so that the loaders are tested along
with the arithmetic. Every quantity is ``X(t; dt) = X*(t) + k·C(t)·dt``, with ``k`` fixed
per scheme, so every error, order and extrapolation has a known value.
"""

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent / "numerical_experiments"


@pytest.fixture(scope="module")
def modules():
    saved = list(sys.path)
    sys.path.insert(0, str(EXAMPLES))
    try:
        return (
            importlib.import_module("scheme_comparison.report"),
            importlib.import_module("scheme_comparison.record"),
            importlib.import_module("scheme_comparison.metrics"),
        )
    finally:
        sys.path[:] = saved  # report.py puts its own parent on the path too


@pytest.fixture(scope="module")
def report(modules):
    return modules[0]


# --------------------------------------------------------------------------------------
# Fabricated runs
# --------------------------------------------------------------------------------------

#: Quadrature weights of the four points; |Ω| = 1.
W = np.array([0.1, 0.2, 0.3, 0.4])
PHASE = np.array([0.0, 0.7, 1.9, 2.8])
#: ``k`` per scheme.
K = {"monolithic": 1.0, "segregated": 5.0, "stabilized": 3.0}
REGIME = {"Kp_kPa": 1.0, "eta_Pa_s": 0.0, "rho_kg_m3": 0.0, "h_m": 0.01}


def lam_star(t: float) -> np.ndarray:
    return 1.0 - 0.1 * np.sin(0.3 * t + PHASE) ** 2


def lam_c(t: float) -> np.ndarray:
    return 0.02 * (1.5 + np.cos(0.5 * t + PHASE))


def ta_star(t: float) -> np.ndarray:
    return 20.0 * np.sin(0.2 * t + PHASE) ** 2


def ta_c(t: float) -> np.ndarray:
    return 2.0 * (1.2 + np.sin(0.4 * t - PHASE))


#: The BiV's V and p: X*(t) and C(t) per log column.
LOG = {
    "V_LV_mL": (lambda t: 100.0 + 10.0 * np.sin(0.1 * t), lambda t: 0.5 + 0.1 * t),
    "V_RV_mL": (lambda t: 80.0 + 8.0 * np.sin(0.1 * t), lambda t: 0.4 + 0.05 * t),
    "p_LV_mmHg": (lambda t: 10.0 + 70.0 * np.sin(0.05 * t) ** 2, lambda t: 1.0 + 0.2 * t),
    "p_RV_mmHg": (lambda t: 4.0 + 20.0 * np.sin(0.05 * t) ** 2, lambda t: 0.3 + 0.1 * t),
}


def _mean(x: np.ndarray) -> float:
    return float(np.sum(W * x) / np.sum(W))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(W * x**2) / np.sum(W)))


def write_run(
    record,
    root: Path,
    geometry: str,
    group: str,
    scheme: str,
    dt: float,
    *,
    t_end: float = 8.0,
    step: float | None = None,
    snap_every: float | None = None,
    t_fail: float | None = None,
    spike: tuple[float, float] | None = None,
    reversal: dict[float, list[float]] | None = None,
    biv: bool = False,
) -> Path:
    """Write one fabricated run and return its directory.

    - ``step``: a ``steps.csv`` row every ``step`` ms (default ``dt``) up to ``t_end``, or
      with ``t_fail`` only those before it, since a failed run records its converged
      steps only.
    - ``snap_every``: a snapshot every ``snap_every`` ms (default every row).
    - ``spike=(t, a)``: add ``a`` to λ, and ``100 a`` to ``Ta``, at every point at time
      ``t``, in ``steps.csv`` and in the snapshot if there is one then.
    - ``reversal``: a floor's ``reversal_fraction`` column, one value per row (default 0).
    """
    step = dt if step is None else step
    t = step * np.arange(round(t_end / step) + 1)
    if t_fail is not None:
        t = t[t < t_fail]
    k = K[scheme] * dt
    lam = np.array([lam_star(x) + k * lam_c(x) for x in t])
    ta = np.array([ta_star(x) + k * ta_c(x) for x in t])
    if spike is not None:
        (at,) = np.flatnonzero(np.isclose(t, spike[0]))
        lam[at] += spike[1]
        ta[at] += 100.0 * spike[1]

    out = root / geometry / group / scheme / f"dt{dt:g}"
    out.mkdir(parents=True)
    reversal = reversal or {}
    with open(out / "steps.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(record.COLUMNS)
        for i, x in enumerate(t):
            row = {
                "t_ms": float(x),
                "newton_iterations": 0 if i == 0 else 2,
                "lmbda_mean": _mean(lam[i]),
                "lmbda_min": float(lam[i].min()),
                "lmbda_max": float(lam[i].max()),
                "Ta_mean_kPa": _mean(ta[i]),
                "Ka_max_kPa": 10.0,
                "KaDl_max_kPa": 0.1 * dt,
            }
            for name in record.COLUMNS:
                if name.startswith("reversal_fraction_"):
                    floor = float(name.removeprefix("reversal_fraction_"))
                    row[name] = reversal.get(floor, [0.0] * len(t))[i]
            writer.writerow([row[name] for name in record.COLUMNS])

    every = step if snap_every is None else snap_every
    on = np.abs(t / every - np.round(t / every)) < 1e-9
    np.savez(
        out / "snapshots.npz",
        t_ms=t[on],
        weights=W,
        lmbda=lam[on],
        tension_kPa=ta[on],
        stiffness_kPa=np.full_like(lam[on], 10.0),
    )

    timings = {"setup_s": 0.5, "mech_s": 1.0, "loop_s": 2.0, "total_s": 2.5}
    (out / "timings.json").write_text(json.dumps(timings))
    reached = t_fail is None
    info = {
        "geometry": geometry,
        "split": group,
        "scheme": scheme,
        "dt_mech_ms": dt,
        "t_end_ms": t_end,
        "regime": REGIME,
        "newton_residuals": [1e-2, 1e-5, 1e-11],
        "reached_t_end": reached,
        "failure": None if reached else "RuntimeError('Newton failed')",
        "t_fail_ms": t_fail,
        "newton": {"total": 2 * (len(t) - 1), "mean": 2.0, "max": 2, "steps": len(t) - 1},
        "timings": timings,
        "git_commit": None,
        "utc": "2026-09-30T00:00:00+00:00",
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))

    if biv:
        logs = {name: np.array([x_(s) + k * c(s) for s in t]) for name, (x_, c) in LOG.items()}
        with open(out / "log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_ms", *LOG, "Ta_mean_kPa", "newton_iterations"])
            for i, x in enumerate(t):
                writer.writerow([x, *(logs[name][i] for name in LOG), _mean(ta[i]), 2])
        vent = {
            "EDV_mL": 110.0,
            "ESV_mL": 80.0,
            "V_range_mL": 30.0,
            "V_range_fraction": 0.27,
            "peak_p_mmHg": 85.0 + k,
            "t_peak_p_ms": 10.0,
            "outflow_valve_open": [
                {
                    "first_open_ms": 4.0,
                    "last_open_ms": 12.0,
                    "open_at_end": False,
                    "ejected_mL": 30.0 + k,
                },
            ],
            "ejects": True,
        }
        summary = {
            "LV": vent,
            "RV": vent,
            "max_conservation_drift": 1e-12,
            "peak_Ta_mean_kPa": max(_mean(row) for row in ta),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out


def _grid_max(f, times) -> float:
    return max(f(t) for t in times)


# --------------------------------------------------------------------------------------
# Orders, and the reference-based orders
# --------------------------------------------------------------------------------------


def test_orders_of_a_first_order_ladder_are_one(report, modules, tmp_path):
    """A3.1: with X(dt) = X* + k C dt every order is 1, for every quantity.

    The reference is a monolithic run at dt 1e-9, finer than any other by far, so its
    own error (k C 1e-9) moves the reference-based orders by less than 1e-8.
    """
    record = modules[1]
    for scheme in ("monolithic", "stabilized"):
        for dt in (2.0, 1.0, 0.5, 0.25):
            write_run(record, tmp_path, "element", "static-caisplit", scheme, dt)
    write_run(record, tmp_path, "element", "static-caisplit", "monolithic", 1e-9, step=0.25)
    lad = report.discover(tmp_path)[("element", "static-caisplit")]
    grid = report.grid_of(lad)
    assert grid == 2.0
    refs = report.make_references("element", lad)
    assert [ref.run_dt for ref in refs] == [1e-9]

    quantities = {q for q, _ in report.QUANTITIES}
    assert quantities == {"lam_rms", "ta_rms", "lam_mean", "ta_mean"}

    self_orders = report.self_orders(lad, grid)
    # Monolithic: 2, 1, 0.5; 1, 0.5, 0.25; 0.5, 0.25, 1e-9. Stabilized: the first two.
    assert len(self_orders) == 5 * len(quantities)
    for order in self_orders:
        assert order.value == pytest.approx(1.0, abs=1e-6), order

    orders, skipped = report.reference_orders(lad, refs, grid)
    assert skipped == []
    # Per scheme: dts 2, 1, 0.5, 0.25 are eligible (>= 5 x 1e-9), so three pairs.
    for scheme in ("monolithic", "stabilized"):
        mine = [o for o in orders if o.scheme == scheme]
        assert {o.quantity for o in mine} == quantities
        assert len(mine) == 3 * len(quantities)
    for order in orders:
        assert order.value == pytest.approx(1.0, abs=1e-6), order

    # The coupling errors themselves (A4's mean Ta among them), on the 2 ms grid.
    grid_times = np.arange(0.0, 8.0 + 1e-9, 2.0)
    coupling = report.coupling_errors(lad, grid)
    for dt in (2.0, 1.0, 0.5, 0.25):
        e = coupling[("stabilized", dt)]
        k = (K["stabilized"] - K["monolithic"]) * dt
        assert e.lam_rms == pytest.approx(k * _grid_max(lambda t: _rms(lam_c(t)), grid_times))
        assert e.ta_rms == pytest.approx(k * _grid_max(lambda t: _rms(ta_c(t)), grid_times))
        assert e.lam_mean == pytest.approx(
            k * _grid_max(lambda t: abs(_mean(lam_c(t))), grid_times),
        )
        assert e.ta_mean == pytest.approx(
            k * _grid_max(lambda t: abs(_mean(ta_c(t))), grid_times),
        )


# --------------------------------------------------------------------------------------
# The BiV's Richardson reference
# --------------------------------------------------------------------------------------


def test_richardson_reference_is_exact_for_a_first_order_ladder(report, modules, tmp_path):
    """A3.2: ``2X(0.25) - X(0.5)`` is X*, so the errors against it are k C dt.

    Snapshots every 10 ms and steps every dt, as on the BiV. Against the extrapolation
    only errors are reported, no orders (ruling R8).
    """
    record = modules[1]
    for scheme in ("monolithic", "stabilized"):
        for dt in (2.0, 1.0, 0.5, 0.25):
            write_run(
                record,
                tmp_path,
                "biv",
                "zetasplit",
                scheme,
                dt,
                t_end=20.0,
                snap_every=10.0,
                biv=True,
            )
    lad = report.discover(tmp_path)[("biv", "zetasplit")]
    grid = report.grid_of(lad)
    refs = report.make_references("biv", lad)
    assert [ref.run_dt for ref in refs] == [0.25, None]
    rich = refs[1]
    assert rich.dt == 0.0

    s = rich.series
    np.testing.assert_allclose(s.t, np.arange(0.0, 20.0 + 1e-9, 0.5))
    np.testing.assert_allclose(s.lmean, [_mean(lam_star(t)) for t in s.t], rtol=0, atol=1e-14)
    np.testing.assert_allclose(s.tamean, [_mean(ta_star(t)) for t in s.t], rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.ts, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(s.lam, [lam_star(t) for t in s.ts], rtol=0, atol=1e-14)
    np.testing.assert_allclose(s.ta, [ta_star(t) for t in s.ts], rtol=0, atol=1e-12)
    for name, (x_star, _) in LOG.items():
        np.testing.assert_allclose(
            rich.logs[name],
            [x_star(t) for t in rich.logs["t_ms"]],
            rtol=0,
            atol=1e-12,
        )

    step_grid = np.arange(0.0, 20.0 + 1e-9, 2.0)
    snap_grid = np.array([0.0, 10.0, 20.0])
    errors = report.reference_errors(lad, rich, grid)
    assert len(errors) == 8
    for (scheme, dt), e in errors.items():
        k = K[scheme] * dt
        assert e.lam_rms == pytest.approx(k * _grid_max(lambda t: _rms(lam_c(t)), snap_grid))
        assert e.ta_rms == pytest.approx(k * _grid_max(lambda t: _rms(ta_c(t)), snap_grid))
        assert e.lam_mean == pytest.approx(
            k * _grid_max(lambda t: abs(_mean(lam_c(t))), step_grid),
        )
        assert e.ta_mean == pytest.approx(
            k * _grid_max(lambda t: abs(_mean(ta_c(t))), step_grid),
        )

    orders, skipped = report.reference_orders(lad, refs, grid)
    assert all(o.ref != rich.name for o in orders)
    assert all(ref_name != rich.name for ref_name, _, _ in skipped)


# --------------------------------------------------------------------------------------
# The 3D onset's floor
# --------------------------------------------------------------------------------------


def test_choose_epsilon_picks_the_smallest_floor_no_monolithic_run_trips(
    report,
    modules,
    tmp_path,
):
    """A3.3: floor 0 trips a monolithic run (3 steps above 0.5), 1e-7 trips none.

    It reads the ``reversal_fraction_1e-07`` column by its name, which is what
    ``genfromtxt`` mangled before ``deletechars=""``.
    """
    record = modules[1]
    n = 9  # rows of a dt 1 run to t 8
    tripped = [0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0]
    below = [0.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0]
    write_run(record, tmp_path, "slab", "zetasplit", "monolithic", 1.0, reversal={0.0: tripped})
    write_run(
        record,
        tmp_path,
        "slab",
        "zetasplit",
        "monolithic",
        2.0,
        reversal={0.0: below[::2], 1e-7: [0.0] * 5},
    )
    write_run(
        record,
        tmp_path,
        "slab",
        "zetasplit",
        "segregated",
        1.0,
        reversal={0.0: [0.9] * n, 1e-7: [0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0]},
    )
    lad = report.discover(tmp_path)[("slab", "zetasplit")]
    eps, note = report.choose_epsilon(lad)
    assert eps == 1e-7
    assert note == ""
    onset = report.onset_of(lad["segregated"][1.0], "3d", eps)
    assert onset.onset_ms == 3.0

    text = report.build_report(tmp_path, tmp_path / "out")
    assert "**ε = 1e-07 per ms**" in text


# --------------------------------------------------------------------------------------
# The common grid
# --------------------------------------------------------------------------------------


def test_common_grid_keeps_a_spike_between_grid_points_out(report, modules, tmp_path):
    """A3.4 (ruling R7): only multiples of the group's coarsest dt enter the maxima.

    The stabilized dt 0.5 run has a spike at t = 3, between the grid points 2 and 4 of
    the 2 ms grid. No error of it may see the spike, although it is in both its
    ``steps.csv`` and its snapshots, and a comparison on its own times would.
    """
    record = modules[1]
    for scheme in ("monolithic", "stabilized"):
        write_run(record, tmp_path, "element", "static-caisplit", scheme, 2.0)
    write_run(record, tmp_path, "element", "static-caisplit", "monolithic", 0.5)
    write_run(record, tmp_path, "element", "static-caisplit", "stabilized", 0.5, spike=(3.0, 0.5))
    lad = report.discover(tmp_path)[("element", "static-caisplit")]
    grid = report.grid_of(lad)
    assert grid == 2.0
    np.testing.assert_array_equal(
        report.on_grid(np.array([0.0, 1.5, 2.0, 3.0, 4.0]), grid),
        [True, False, True, False, True],
    )

    grid_times = np.arange(0.0, 8.0 + 1e-9, 2.0)
    k = (K["stabilized"] - K["monolithic"]) * 0.5
    expected = {
        "lam_rms": k * _grid_max(lambda t: _rms(lam_c(t)), grid_times),
        "ta_rms": k * _grid_max(lambda t: _rms(ta_c(t)), grid_times),
        "lam_mean": k * _grid_max(lambda t: abs(_mean(lam_c(t))), grid_times),
        "ta_mean": k * _grid_max(lambda t: abs(_mean(ta_c(t))), grid_times),
    }
    coupling = report.coupling_errors(lad, grid)[("stabilized", 0.5)]
    (ref,) = report.make_references("element", lad)
    assert ref.run_dt == 0.5
    against_ref = report.reference_errors(lad, ref, grid)[("stabilized", 0.5)]
    for e in (coupling, against_ref):
        for q, value in expected.items():
            assert getattr(e, q) == pytest.approx(value), q

    # Not vacuous: over the run's own times, the spike is the maximum.
    s, s_mono = report.series_of(lad["stabilized"][0.5]), report.series_of(lad["monolithic"][0.5])
    assert report.compare(s, s_mono, None).lam_mean > 0.5
    assert report.compare(s, s_mono, None).lam_rms > 0.5


def test_mean_differences_come_from_every_step(report, modules, tmp_path):
    """A5: the mean-λ and mean-Ta differences are taken at every step on the grid.

    Snapshots every 10 ms, as on the BiV: a spike at t = 4, a grid point but not a
    snapshot time, must enter the mean differences, and cannot enter the RMS ones.
    """
    record = modules[1]
    kw = {"t_end": 20.0, "snap_every": 10.0, "biv": True}
    for dt in (2.0, 0.5):
        write_run(record, tmp_path, "biv", "zetasplit", "monolithic", dt, **kw)
    write_run(record, tmp_path, "biv", "zetasplit", "stabilized", 2.0, spike=(4.0, 0.5), **kw)
    lad = report.discover(tmp_path)[("biv", "zetasplit")]
    grid = report.grid_of(lad)
    refs = report.make_references("biv", lad)
    _, store = report.accuracy_tables(lad, refs, grid)

    e = store[("stabilized", 2.0, "main")]
    assert e.lam_mean > 0.5
    assert e.ta_mean > 50.0
    snap_grid = np.array([0.0, 10.0, 20.0])
    k = K["stabilized"] * 2.0 - K["monolithic"] * 0.5
    assert e.lam_rms == pytest.approx(k * _grid_max(lambda t: _rms(lam_c(t)), snap_grid))


# --------------------------------------------------------------------------------------
# The reference is a completed run
# --------------------------------------------------------------------------------------


def test_a_failed_monolithic_run_is_never_the_reference(report, modules, tmp_path):
    """A5: the finest monolithic run failed, so the next one is the reference."""
    record = modules[1]
    for dt in (2.0, 1.0):
        write_run(record, tmp_path, "slab", "zetasplit", "monolithic", dt)
    write_run(record, tmp_path, "slab", "zetasplit", "monolithic", 0.5, t_fail=5.0)
    write_run(record, tmp_path, "slab", "zetasplit", "stabilized", 0.5)
    lad = report.discover(tmp_path)[("slab", "zetasplit")]
    assert lad["monolithic"][0.5].failed
    (ref,) = report.make_references("slab", lad)
    assert ref.run_dt == 1.0
    assert report.finest_monolithic(lad) is lad["monolithic"][1.0]


def test_no_completed_monolithic_run_is_said(report, modules, tmp_path):
    """A5: with every monolithic run failed there is no reference, and the report says so."""
    record = modules[1]
    for dt in (2.0, 1.0):
        write_run(record, tmp_path, "slab", "zetasplit", "monolithic", dt, t_fail=5.0)
    write_run(record, tmp_path, "slab", "zetasplit", "stabilized", 1.0)
    lad = report.discover(tmp_path)[("slab", "zetasplit")]
    assert report.finest_monolithic(lad) is None
    assert report.make_references("slab", lad) == []
    text, _ = report.accuracy_tables(lad, [], report.grid_of(lad))
    assert "No monolithic run reached t_end" in text


# --------------------------------------------------------------------------------------
# The whole report
# --------------------------------------------------------------------------------------


def _small_root(record, root: Path, *, biv: bool) -> None:
    for scheme in ("monolithic", "stabilized", "segregated"):
        for dt in (2.0, 1.0, 0.5):
            write_run(record, root, "element", "static-caisplit", scheme, dt)
            if not biv:
                continue
            write_run(
                record,
                root,
                "biv",
                "zetasplit",
                scheme,
                dt,
                t_end=20.0,
                snap_every=10.0,
                biv=True,
            )


def test_report_on_a_clean_root(report, modules, tmp_path):
    """A3.5, A5: no load problems, no failed section, and Ta both raw and normalized."""
    _small_root(modules[1], tmp_path, biv=True)
    text = report.build_report(tmp_path, tmp_path / "out")
    assert "No load problems" in text
    assert "## Load problems" not in text
    assert "failed to build" not in text.lower()
    assert (tmp_path / "out" / "report.md").read_text() == text

    # The normalization constant, once per reference: the reference's max RMS Ta.
    sections = {s.split("\n", 1)[0]: s for s in text.split("\n## ")[1:]}
    ladders = report.discover(tmp_path)
    for (geometry, group), n_refs in (
        (("element", "static-caisplit"), 1),
        (("biv", "zetasplit"), 2),
    ):
        section = sections[f"{geometry} / {group}"]
        lad = ladders[(geometry, group)]
        refs = report.make_references(geometry, lad)
        assert len(refs) == n_refs
        for ref in refs:
            norm = report.ta_norm(ref.series, report.grid_of(lad))
            assert section.count(f"{norm:.4g} kPa") == 1, (geometry, ref.name)
        assert "RMS Ta (kPa)" in section
        assert "RMS Ta / ref max RMS Ta" in section
        assert "max abs Δ mean Ta (kPa)" in section
        assert "order, mean Ta" in section


def test_report_lists_a_truncated_run_json(report, modules, tmp_path):
    """A3.5: a run.json cut short is a load problem, named in the report."""
    _small_root(modules[1], tmp_path, biv=False)
    bad = tmp_path / "element" / "static-caisplit" / "stabilized" / "dt1" / "run.json"
    content = bad.read_text()
    bad.write_text(content[: len(content) // 2])
    text = report.build_report(tmp_path, tmp_path / "out")
    assert "## Load problems" in text
    assert "No load problems" not in text
    (line,) = [line for line in text.splitlines() if str(bad) in line]
    assert "unreadable JSON" in line
