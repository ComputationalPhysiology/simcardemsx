"""
Reproduction of Regazzoni & Quarteroni (2021) Test Case 1, in 0D.

This is the test that justifies the whole stabilized-segregated design. The
claim being checked is stronger than "the naive scheme is less accurate":

    The naive segregated scheme is *not convergent* when the active stiffness
    exceeds the passive stiffness. Refining the time step does not help, and
    makes it worse.

`test_naive_scheme_does_not_converge` is where that is asserted. Everything
else establishes the conditions under which it is a meaningful statement --
that the monolithic run really is a reference, that all three schemes agree
where they should, and that the stabilized scheme converges at the first-order
rate its consistency argument predicts.

The instability lives in the **quasistatic** regime (M = sigma = 0), which is
what R&Q analyse in Sec. 4.3.1. With their dynamic parameters and a small dt
the inertia term M/dt^2 acts as an enormous additional stiffness -- at
dt = 1e-3 it is 1e5 Pa, dwarfing Kp -- and stabilizes the scheme by swamping
the feedback loop. That is a property of the test setup, not a defence of the
naive scheme.
"""

import numpy as np
import pytest

from simcardemsx import zero_d

#: Quasistatic, and soft enough that Ka exceeds Kp during the twitch.
UNSTABLE = zero_d.Tissue(Kp=2e4, M=0.0, sigma=0.0)
#: Same tissue model, stiff enough that Ka stays below Kp throughout.
STABLE = zero_d.Tissue(Kp=1e6, M=0.0, sigma=0.0)

T_END = 0.4
DT = 1e-3


@pytest.fixture(scope="module")
def unstable():
    return zero_d.compare(tissue=UNSTABLE, dt=DT, T=T_END)


@pytest.fixture(scope="module")
def stable():
    return zero_d.compare(tissue=STABLE, dt=DT, T=T_END)


# ---------------------------------------------------------------------------
# The reference has to be trustworthy before it can arbitrate
# ---------------------------------------------------------------------------


def test_calcium_transient_matches_the_paper():
    """R&Q Eq. (57): rest at c0, a single peak near cmax, decaying back."""
    t = np.linspace(0, 0.8, 2001)
    ca = zero_d.calcium_transient(t)

    assert np.allclose(ca[t < 0.1], 0.1), "should sit at c0 before t0"
    assert 1.4 < ca.max() < 1.7, f"peak {ca.max()} is not near cmax = 1.6"
    assert 0.1 <= t[ca.argmax()] <= 0.2, "peak is at the wrong time"
    assert ca[-1] < 0.3, "transient did not decay"


def test_monolithic_reference_is_smooth(unstable):
    """The reference must not oscillate, in the very regime where the naive
    scheme does -- otherwise it cannot be used to judge anything."""
    mono = unstable["monolithic"]
    assert mono.oscillation_metric < 0.02, (
        f"reference itself oscillates ({mono.oscillation_metric})"
    )
    # ~4.8% peak shortening in this regime; the bar is just "clearly contracting"
    assert mono.strain.min() < -0.04, "no appreciable contraction to compare against"


def test_monolithic_reference_is_dt_converged():
    """Halving dt must barely move the reference, or it is not one."""
    coarse = zero_d.solve("monolithic", tissue=UNSTABLE, dt=2e-3, T=T_END)
    fine = zero_d.solve("monolithic", tissue=UNSTABLE, dt=1e-3, T=T_END)
    diff = np.max(np.abs(coarse.strain[::2] - fine.strain[::4][: coarse.strain[::2].size]))
    assert diff < 0.02, f"reference moved by {diff} under refinement"


# ---------------------------------------------------------------------------
# The instability, and its absence
# ---------------------------------------------------------------------------


def test_naive_scheme_oscillates_when_active_stiffness_dominates(unstable):
    """The failure R&Q predict, in the regime they predict it."""
    naive = unstable["segregated"]
    stabilized = unstable["stabilized"]

    assert naive.Ka.max() > UNSTABLE.Kp, "test regime does not actually have Ka > Kp"
    assert naive.oscillation_metric > 0.1, (
        f"naive scheme did not oscillate ({naive.oscillation_metric}); "
        "the regime may no longer be unstable"
    )
    assert stabilized.oscillation_metric < 0.02, (
        f"stabilized scheme oscillated ({stabilized.oscillation_metric})"
    )


def test_all_schemes_agree_where_the_naive_one_is_stable(stable):
    """With Ka < Kp there is nothing to stabilize, and the stabilization must
    not be quietly changing the answer. Guards against it masking a real bug.
    """
    assert stable["segregated"].Ka.max() < STABLE.Kp
    for scheme in ("segregated", "stabilized"):
        assert stable[scheme].oscillation_metric < 0.02
        assert stable.error_against_monolithic(scheme) < 5e-3, (
            f"{scheme} disagrees with the reference in the stable regime"
        )


def test_naive_scheme_does_not_converge():
    """The sharp claim: refining dt does not fix the naive scheme.

    R&Q show its amplification factor tends to -Ka/Kp as dt -> 0, so for
    Ka > Kp it is not merely inaccurate but non-convergent -- refinement cannot
    buy accuracy. Asserted as: the oscillation does not die away, and the error
    against the reference does not shrink.
    """
    dts = (4e-3, 2e-3, 1e-3)
    runs = [zero_d.compare(tissue=UNSTABLE, dt=dt, T=T_END) for dt in dts]

    naive_osc = [r["segregated"].oscillation_metric for r in runs]
    naive_err = [r.error_against_monolithic("segregated") for r in runs]

    assert min(naive_osc) > 0.1, f"naive oscillation died away under refinement: {naive_osc}"
    assert naive_err[-1] >= naive_err[0] * 0.9, (
        f"naive error decreased under refinement, contradicting non-convergence: {naive_err}"
    )


def test_stabilized_scheme_converges_at_first_order():
    """And the stabilized one does what a consistent O(dt) scheme should.

    Same runs, same regime, opposite behaviour -- which is what makes the
    previous test a statement about the scheme rather than about the problem.
    """
    dts = (4e-3, 2e-3, 1e-3, 5e-4)
    errors = [
        zero_d.compare(tissue=UNSTABLE, dt=dt, T=T_END).error_against_monolithic("stabilized")
        for dt in dts
    ]

    assert all(b < a for a, b in zip(errors, errors[1:])), f"error did not decrease: {errors}"

    rates = [np.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(0.7 < r < 1.4 for r in rates), f"expected first-order convergence, got {rates}"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="unknown scheme"):
        zero_d.solve(scheme="staggered-ish")


@pytest.mark.parametrize("model", ["Land2017", "Lewalle2024"])
def test_runs_for_the_distortion_decay_models(model):
    """Both models with genuine strain-rate feedback must drive the reference."""
    result = zero_d.solve("stabilized", model=model, tissue=UNSTABLE, dt=DT, T=0.3)
    assert np.all(np.isfinite(result.strain))
    assert result.Ta.max() > 0.0, f"{model} generated no tension"


def test_reports_stretch_as_well_as_strain():
    """The 0D module works in strain; the FEM backends work in stretch. The
    conversion is exposed so comparisons cannot get it backwards."""
    result = zero_d.solve("stabilized", tissue=STABLE, dt=DT, T=0.2)
    np.testing.assert_allclose(result.stretch, 1.0 + result.strain)
    assert result.stretch[0] == 1.0
