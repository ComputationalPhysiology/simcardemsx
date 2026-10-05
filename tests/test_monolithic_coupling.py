"""The coupling gates, run through a real one-element ``pulse.StaticProblem``.

These are the tests the monolithic design rests on. The contraction model is
stepped inside Newton (``scheme="monolithic"``), or with the stretch frozen at
the last converged step (``scheme="segregated"``, the naive scheme). The setup is
quasistatic and the passive material soft, so active stiffness exceeds passive:
the regime in which Regazzoni & Quarteroni show the naive scheme is not
convergent. It is the FEM counterpart of ``tests/test_zero_d_coupling.py``.

- Gate 1: the monolithic scheme converges under time-step refinement against a
  dt = 0.01 ms monolithic reference, while the segregated one gets *worse*: it
  becomes unstable earlier as dt shrinks.
- S1: the stabilized segregated scheme converges at first order against the same
  reference, in the same regime.
- Gate 4: the zeta split, run monolithically, twitches without the period-2
  oscillation the old ``ZetaSplitUFL`` coupling showed.
"""

import pytest
from conftest import (
    _caisplit_inputs,
    _lmbda_error,
    _observed_orders,
    _reversals,
    _Run,
    _run,
    _zetasplit_inputs,
)


def _worse(fine: _Run, coarse: _Run) -> bool:
    """Whether the finer run is worse: it becomes unstable earlier."""
    return fine.t_unstable < coarse.t_unstable


@pytest.mark.slow
def test_monolithic_converges_and_segregated_does_not(
    split_modules,
    make_mechanics,
    caisplit_reference,
):
    """Gate 1: monolithic converges under dt refinement; the naive scheme gets worse."""
    t_end = 40.0
    args = (split_modules, make_mechanics, "caisplit")
    ref = caisplit_reference
    # The instability criterion must not trip on the monolithic scheme, or it would
    # not tell the two schemes apart.
    assert ref.t_unstable == t_end, (ref.t_fail, ref.spread.max())

    e = {}
    for dt in (1.0, 0.25, 0.05):
        run = _run(*args, "monolithic", dt, t_end, _caisplit_inputs)
        assert run.t_unstable == t_end, (dt, run.t_fail, run.spread.max())
        e[dt] = _lmbda_error(run, ref)

    orders = _observed_orders(e)
    assert orders[0] >= 0.8, e
    assert orders[1] >= 0.8, e

    seg = {dt: _run(*args, "segregated", dt, t_end, _caisplit_inputs) for dt in (1.0, 0.25, 0.05)}
    summary = {dt: (run.t_unstable, run.t_fail) for dt, run in seg.items()}
    assert _worse(seg[0.25], seg[1.0]), summary
    assert _worse(seg[0.05], seg[0.25]), summary


@pytest.mark.slow
def test_stabilized_converges_at_first_order(split_modules, make_mechanics, caisplit_reference):
    """S1: the stabilized scheme converges at first order where the naive one does not.

    Gate 1's inputs, regime and reference. Where the naive scheme becomes unstable
    earlier as dt shrinks, the stabilized one must never trip the instability
    criterion, and must converge at first order.
    """
    t_end = 40.0
    assert caisplit_reference.t_unstable == t_end

    e = {}
    for dt in (1.0, 0.25, 0.05):
        run = _run(
            split_modules,
            make_mechanics,
            "caisplit",
            "stabilized",
            dt,
            t_end,
            _caisplit_inputs,
        )
        assert run.t_unstable == t_end, (dt, run.t_fail, run.spread.max())
        e[dt] = _lmbda_error(run, caisplit_reference)

    orders = _observed_orders(e)
    assert orders[0] >= 0.8, e  # probe: 0.83
    assert orders[1] >= 0.8, e  # probe: 0.97


@pytest.mark.slow
def test_naive_zeta_split_becomes_unstable_earlier_as_dt_shrinks(split_modules, make_mechanics):
    """The control for gate 4: the naive segregated scheme, on gate 4's inputs, goes
    unstable within the run, and earlier the smaller dt is.

    The probe measured onsets of 20 / 15.75 / 7.7 ms at dt 1 / 0.25 / 0.05.
    """
    t_end = 80.0
    onsets = [
        _run(
            split_modules,
            make_mechanics,
            "zetasplit",
            "segregated",
            dt,
            t_end,
            _zetasplit_inputs,
        ).t_unstable
        for dt in (1.0, 0.25, 0.05)
    ]
    assert all(t < t_end for t in onsets), onsets
    assert onsets[0] > onsets[1] > onsets[2], onsets


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["monolithic", "stabilized"])
@pytest.mark.parametrize("dt", [1.0, 0.25, 0.05])
def test_zeta_split_has_no_period_two_oscillation(split_modules, make_mechanics, scheme, dt):
    """Gate 4: the zeta split, monolithic or stabilized, gives one clean twitch at every dt."""
    t_end = 80.0
    run = _run(
        split_modules,
        make_mechanics,
        "zetasplit",
        scheme,
        dt,
        t_end,
        _zetasplit_inputs,
    )
    assert run.t_fail == t_end
    # The naive scheme's unstable mode is spatial (spread over the quadrature
    # points), hidden from a reversal count on mean(lambda); the monolithic
    # spread measured here is ~5e-16, so this is not a vacuous check.
    assert run.t_unstable == t_end
    assert _reversals(run.trace) <= 2
