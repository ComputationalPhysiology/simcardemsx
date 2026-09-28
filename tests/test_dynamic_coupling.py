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
"""

import numpy as np
import pytest
from conftest import calcium

DT_MS = 1.0
T_END = 60.0


def _run(make_dynamic_mechanics, mech_module, *, reference: bool, end_of_step: bool) -> np.ndarray:
    """Drive the pinned D1 element exactly as ``_run`` in
    ``test_monolithic_coupling.py`` drives the one-element ``StaticProblem`` gates:
    set ``t``/``dt``, write the Ca_i input at ``t_{n+1}``, solve, ``post_solve``,
    record ``mean(lmbda_prev)``. Raises if any step fails to converge or leaves a
    non-finite displacement.
    """
    problem, backend = make_dynamic_mechanics(
        mech_module,
        dt_ms=DT_MS,
        reference=reference,
        end_of_step=end_of_step,
    )
    trace = []
    for n in range(round(T_END / DT_MS)):
        t_n, t_next = n * DT_MS, (n + 1) * DT_MS
        backend.t.value = t_n
        backend.dt.value = DT_MS
        backend.inputs["cai"].x.array[:] = calcium(t_next)

        ok = problem.solve(raise_on_failure=False)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = bool(np.all(np.isfinite(problem.u.x.array)))
        if not ok or not finite:
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
