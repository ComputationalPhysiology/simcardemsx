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
goes non-finite, nor loses its damping (at most 2 reversals of λ). It is not a
gate for the period-2 oscillation gate 4 (``test_monolithic_coupling.py``)
checks under ``StaticProblem``: in its pinned, damped regime it passes under the
alpha_f evaluation and under the naive segregated scheme too (1 reversal at every
dt), so it cannot tell either from the flagged monolithic scheme. Telling
end-of-step from alpha_f apart is D1's job.
"""

from collections.abc import Callable

import numpy as np
import pytest
from conftest import _reversals, _zetasplit_inputs, calcium

DT_MS = 1.0
T_END = 60.0


def _run(
    make_dynamic_mechanics,
    mech_module,
    *,
    reference: bool,
    end_of_step: bool,
    dt_ms: float = DT_MS,
    t_end: float = T_END,
    inputs_of_t: Callable[[float], dict[str, float]] = lambda t: {"cai": calcium(t)},
) -> np.ndarray:
    """Drive the pinned D1/D2 element exactly as ``_run`` in
    ``test_monolithic_coupling.py`` drives the one-element ``StaticProblem`` gates:
    set ``t``/``dt``, write the inputs at ``t_{n+1}``, solve, ``post_solve``,
    record ``mean(lmbda_prev)``.

    Parameters
    ----------
    reference, end_of_step:
        Passed on to ``make_dynamic_mechanics``: the ``_TrueUActive`` reference, and
        whether the flag is set (``False`` is the alpha_f evaluation).
    dt_ms:
        The time step in ms, of both the backend and the problem.
    t_end:
        The end time in ms: ``round(t_end / dt_ms)`` steps, from ``t = 0``.
    inputs_of_t:
        The backend's inputs at a time in ms, as ``{name: value}``; each value is
        written into all of ``backend.inputs[name]`` at ``t_{n+1}``. The default is the
        Ca_i split's prescribed ``calcium``.

    Returns
    -------
    np.ndarray
        ``mean(lmbda_prev)`` after each step.

    Raises
    ------
    AssertionError
        If a step leaves a non-finite displacement, or fails to converge; the
        message says which.
    """
    problem, backend = make_dynamic_mechanics(
        mech_module,
        dt_ms=dt_ms,
        reference=reference,
        end_of_step=end_of_step,
    )
    trace = []
    for n in range(round(t_end / dt_ms)):
        t_n, t_next = n * dt_ms, (n + 1) * dt_ms
        backend.t.value = t_n
        backend.dt.value = dt_ms
        for name, value in inputs_of_t(t_next).items():
            backend.inputs[name].x.array[:] = value

        ok = problem.solve(raise_on_failure=False)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = bool(np.all(np.isfinite(problem.u.x.array)))
        if not finite:
            raise AssertionError(
                f"step {n} (t={t_next} ms) left a non-finite displacement (Newton converged: {ok})",
            )
        if not ok:
            raise AssertionError(f"step {n} (t={t_next} ms) failed to converge")

        backend.post_solve()
        trace.append(float(np.mean(backend.lmbda_prev.x.array)))
    return np.array(trace)


@pytest.mark.slow
def test_active_stress_is_evaluated_at_the_end_of_the_step(split_modules, make_dynamic_mechanics):
    """The flag matches the by-hand end-of-step reference to round-off, and
    genuinely differs from the ``alpha_f`` evaluation it replaces.

    A probe run measured ``max|flagged - alpha_f| = 7.7e-3`` in ``mean(lmbda)``
    (peak active tension 1.80 kPa flagged vs. 1.23 kPa alpha_f); the ``> 1e-3``
    threshold below is comfortably inside that margin.
    """
    _, mech = split_modules["caisplit"]

    flagged = _run(make_dynamic_mechanics, mech, reference=False, end_of_step=True)
    reference = _run(make_dynamic_mechanics, mech, reference=True, end_of_step=True)
    alpha_f = _run(make_dynamic_mechanics, mech, reference=False, end_of_step=False)

    np.testing.assert_allclose(flagged, reference, rtol=1e-10, atol=0)
    assert np.max(np.abs(flagged - alpha_f)) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("dt", [2.0, 1.0, 0.5])
def test_zeta_split_has_no_period_two_oscillation_under_dynamic_problem(
    split_modules,
    make_dynamic_mechanics,
    dt,
):
    """Gate D2: the zeta split, under ``DynamicProblem`` with the flag set, gives
    one clean twitch at every dt: no divergence, no non-finite displacement, and at
    most 2 reversals of λ, i.e. the damping is not lost.

    This guards those failures only. It does not discriminate the period-2
    instability, as gate 4 (``test_monolithic_coupling.py::
    test_zeta_split_has_no_period_two_oscillation``) does under ``StaticProblem``: on
    this damped 1 cm element it passes under the alpha_f evaluation and under the
    naive segregated scheme as well (1 reversal at every dt). D1 is the gate that
    tells end-of-step from alpha_f apart.
    """
    t_end = 150.0
    _, mech = split_modules["zetasplit"]
    trace = _run(
        make_dynamic_mechanics,
        mech,
        reference=False,
        end_of_step=True,
        dt_ms=dt,
        t_end=t_end,
        inputs_of_t=_zetasplit_inputs,
    )
    assert trace.size == round(t_end / dt)
    assert _reversals(trace) <= 2
