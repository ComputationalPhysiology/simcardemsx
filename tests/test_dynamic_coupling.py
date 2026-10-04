"""Gate D1: ``GeneratedActivation`` evaluated at the true end of the step.

``pulse.DynamicProblem`` assembles the material form at the generalized-alpha
``alpha_f`` point. ``GeneratedActivation``'s active stress depends on states
advanced by one Rush-Larsen step, using the stretch and stretch *rate* of the
displacement it is handed -- not on the instantaneous deformation alone -- so
assembling it at ``alpha_f`` feeds it a blended stretch and a rate scaled by
``1 - alpha_f``, not the ones it actually advanced with.
``pulse.active_model.ActiveModel.evaluate_at_end_of_step`` tells
``DynamicProblem`` to assemble the active stress at the true end-of-step
displacement instead (see ``CLAUDE.md``, architecture 1).

This is checked against a reference built a completely different way: the
backend is *not* ``model.active`` (which is ``pulse.active_model.Passive()``
instead), and its ``S`` is added to the residual by hand, at the true
end-of-step displacement, by ``tests/conftest.py``'s ``_TrueUActive`` --
physcardems's own pattern for this. If the flag reproduces that by-hand
construction to round-off, while genuinely differing from the ``alpha_f``
evaluation, the flag does what it claims.

Gate D2 (below) is a weaker claim, about the zeta split with the flag set: one
twitch under ``DynamicProblem`` at dt 2, 1 and 0.5 ms that neither diverges, nor
goes non-finite, nor loses its damping (at most 2 reversals of λ), under the
monolithic and the stabilized scheme. It discriminates through its naive control:
the inputs are 10 times gate 4's, which puts ``Ka_max`` at 464-495 kPa against the
element's eta/dt + rho L^2/dt^2 = 75 / 200 / 600 kPa at 2 / 1 / 0.5 ms, and there the
naive segregated scheme fails or oscillates (dt 2 and 1). At 0.5 ms the naive scheme
is clean; that is recorded, not asserted. At the original inputs (x1) no scheme could
be told apart: every one gave 1 reversal at every dt. Telling end-of-step from alpha_f
apart is D1's job.
"""

import numpy as np
import pytest
from conftest import _reversals, _run_dynamic, _run_dynamic_until_failure, _zetasplit_inputs

#: Factor on :func:`~conftest._zetasplit_inputs` in D2 (XS peaks at 0.1, XW at 0.05).
D2_INPUT_SCALE = 10.0


def _d2_inputs(t: float) -> dict[str, float]:
    """The zeta split's inputs scaled by :data:`D2_INPUT_SCALE`: XS peaks at 0.1, XW at 0.05."""
    return {name: D2_INPUT_SCALE * value for name, value in _zetasplit_inputs(t).items()}


@pytest.mark.slow
def test_active_stress_is_evaluated_at_the_end_of_the_step(split_modules, make_dynamic_mechanics):
    """The flag matches the by-hand end-of-step reference to round-off, and
    genuinely differs from the ``alpha_f`` evaluation it replaces.

    A probe run measured ``max|flagged - alpha_f| = 7.7e-3`` in ``mean(lmbda)``
    (peak active tension 1.80 kPa flagged vs. 1.23 kPa alpha_f); the ``> 1e-3``
    threshold below is comfortably inside that margin.
    """
    _, mech = split_modules["caisplit"]

    flagged = _run_dynamic(make_dynamic_mechanics, mech, reference=False, end_of_step=True)
    reference = _run_dynamic(make_dynamic_mechanics, mech, reference=True, end_of_step=True)
    alpha_f = _run_dynamic(make_dynamic_mechanics, mech, reference=False, end_of_step=False)

    np.testing.assert_allclose(flagged, reference, rtol=1e-10, atol=0)
    assert np.max(np.abs(flagged - alpha_f)) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["monolithic", "stabilized"])
@pytest.mark.parametrize("dt", [2.0, 1.0, 0.5])
def test_zeta_split_has_no_period_two_oscillation_under_dynamic_problem(
    split_modules,
    make_dynamic_mechanics,
    scheme,
    dt,
):
    """Gate D2: the zeta split, under ``DynamicProblem`` with the flag set, gives
    one clean twitch at every dt, under the monolithic and the stabilized scheme: no
    divergence, no non-finite displacement, and at most 2 reversals of λ, i.e. the
    damping is not lost.

    The regime is chosen so the naive scheme fails here at dt 2 and 1 ms (see
    :func:`test_naive_scheme_oscillates_in_the_d2_regime`): the inputs are
    ``D2_INPUT_SCALE`` = 10 times gate 4's, which puts the active stiffness
    ``Ka_max`` at 464-495 kPa. That is above what the element supplies at dt 2 and
    1 ms, eta/dt + rho L^2/dt^2 = 75 and 200 kPa, but not at dt 0.5 ms, where it
    supplies 600 kPa: 200 of damping (eta/dt) and 400 of inertia (rho L^2/dt^2). The
    probe measured 1 reversal at every dt for both schemes here.
    """
    t_end = 150.0
    _, mech = split_modules["zetasplit"]
    trace = _run_dynamic(
        make_dynamic_mechanics,
        mech,
        reference=False,
        end_of_step=True,
        scheme=scheme,
        dt_ms=dt,
        t_end=t_end,
        inputs_of_t=_d2_inputs,
    )
    assert trace.size == round(t_end / dt)
    assert _reversals(trace) <= 2


@pytest.mark.slow
@pytest.mark.parametrize("dt", [2.0, 1.0])
def test_naive_scheme_oscillates_in_the_d2_regime(split_modules, make_dynamic_mechanics, dt):
    """The control for D2: the naive segregated scheme, in D2's regime, fails or
    oscillates.

    Without it D2 discriminated nothing. At ``D2_INPUT_SCALE`` = 1 (gate 4's inputs)
    every scheme gave 1 reversal at every dt, so the element's damping hid the
    instability. At 10 the probe measured a failure at 34 ms (9 reversals) at dt 2 and
    29 reversals at dt 1. At dt 0.5 ms the naive scheme is clean (the element then
    supplies 600 kPa, 200 of damping and 400 of inertia, above ``Ka_max``): recorded,
    not asserted.
    """
    _, mech = split_modules["zetasplit"]
    trace, t_fail = _run_dynamic_until_failure(
        make_dynamic_mechanics,
        mech,
        reference=False,
        end_of_step=True,
        scheme="segregated",
        dt_ms=dt,
        t_end=150.0,
        inputs_of_t=_d2_inputs,
    )
    assert t_fail is not None or _reversals(trace) > 2, (dt, _reversals(trace))
