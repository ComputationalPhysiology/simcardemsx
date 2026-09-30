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

from collections.abc import Callable
from typing import Literal, NamedTuple

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from conftest import _mechanics, _reversals, _zetasplit_inputs, calcium

#: Spread of λ over the element's quadrature points above which a run is unstable.
#: The load and the boundary conditions are uniform, so the solution is uniform: the
#: monolithic runs stay below 1e-15. At the smaller time steps the naive scheme's
#: unstable mode is spatial, amplified from round-off, and the mean of λ hides it.
_UNSTABLE_SPREAD = 1e-3


def _caisplit_inputs(t: float) -> dict[str, float]:
    return {"cai": calcium(t)}


class _Run(NamedTuple):
    trace: np.ndarray  # mean(lmbda_prev) after each converged step, at dt, 2 dt, ...
    spread: np.ndarray  # max - min of lmbda_prev after each converged step
    dt: float
    t_fail: float  # end of the first step that failed, or t_end if none did
    t_end: float

    @property
    def t_unstable(self) -> float:
        """End of the first step after which λ is no longer uniform, or ``t_fail``.

        Instability is measured at its onset rather than by the Newton failure
        alone: when Newton gives up on the naive scheme depends on the solver
        settings (line search, tolerances), when the mode appears does not.
        """
        (unstable,) = np.nonzero(self.spread > _UNSTABLE_SPREAD)
        if unstable.size:
            return min(float((unstable[0] + 1) * self.dt), self.t_fail)
        return self.t_fail


def _run(
    split_modules,
    make_mechanics,
    split: str,
    scheme: Literal["monolithic", "segregated", "stabilized"],
    dt: float,
    t_end: float,
    inputs_of_t: Callable[[float], dict[str, float]],
) -> _Run:
    """Contract one element under the given inputs: λ after each step, and when it failed.

    Records the mean of λ over the element's quadrature points and its spread
    (max - min).

    Each step sets ``t = t_n`` and ``dt``, the inputs at ``t_{n+1}``, solves, and
    accepts the step with ``post_solve``. The run stops at the first step whose
    Newton solve fails or leaves a non-finite displacement.
    """
    _, mech = split_modules[split]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    problem, backend = make_mechanics(mech, mesh, scheme=scheme)

    trace: list[float] = []
    spread: list[float] = []
    for n in range(round(t_end / dt)):
        t_n, t_next = n * dt, (n + 1) * dt
        backend.t.value = t_n
        backend.dt.value = dt
        for name, value in inputs_of_t(t_next).items():
            backend.inputs[name].x.array[:] = value

        ok = problem.solve(raise_on_failure=False)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = bool(np.all(np.isfinite(problem.u.x.array)))
        if not ok or not finite:
            return _Run(np.array(trace), np.array(spread), dt, t_next, t_end)

        backend.post_solve()
        lmbda = backend.lmbda_prev.x.array
        trace.append(float(np.mean(lmbda)))
        spread.append(float(lmbda.max() - lmbda.min()))
    return _Run(np.array(trace), np.array(spread), dt, t_end, t_end)


def _at_whole_ms(trace: np.ndarray, dt: float) -> np.ndarray:
    """The entries of a trace recorded at t = 1, 2, ... ms."""
    per_ms = round(1.0 / dt)
    return trace[per_ms - 1 :: per_ms]


def _worse(fine: _Run, coarse: _Run) -> bool:
    """Whether the finer run is worse: it becomes unstable earlier."""
    return fine.t_unstable < coarse.t_unstable


@pytest.fixture(scope="module")
def caisplit_reference(split_modules) -> _Run:
    """The reference of gate 1 and S1: the Ca_i split, monolithic, at dt 0.01 ms for 40 ms.

    Built with ``conftest._mechanics`` itself, since the ``make_mechanics`` fixture
    is function-scoped and this one is shared by the module.
    """
    return _run(split_modules, _mechanics, "caisplit", "monolithic", 0.01, 40.0, _caisplit_inputs)


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
    lmbda_ref = _at_whole_ms(ref.trace, ref.dt)

    e = {}
    for dt in (1.0, 0.25, 0.05):
        run = _run(*args, "monolithic", dt, t_end, _caisplit_inputs)
        assert run.t_unstable == t_end, (dt, run.t_fail, run.spread.max())
        e[dt] = np.max(np.abs(_at_whole_ms(run.trace, dt) - lmbda_ref))

    assert np.log(e[1.0] / e[0.25]) / np.log(4) >= 0.8, e
    assert np.log(e[0.25] / e[0.05]) / np.log(5) >= 0.8, e

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
    lmbda_ref = _at_whole_ms(caisplit_reference.trace, caisplit_reference.dt)

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
        e[dt] = np.max(np.abs(_at_whole_ms(run.trace, dt) - lmbda_ref))

    assert np.log(e[1.0] / e[0.25]) / np.log(4) >= 0.8, e  # probe: 0.83
    assert np.log(e[0.25] / e[0.05]) / np.log(5) >= 0.8, e  # probe: 0.97


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
