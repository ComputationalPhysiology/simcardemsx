# Couple NumPy contraction models by a stabilized segregated scheme

Contraction models written in NumPy cannot be re-integrated inside a UFL Newton
iteration, so coupling one to mechanics is necessarily segregated. The naive
segregated scheme is not merely inaccurate for cardiac tissue — Regazzoni &
Quarteroni show its spectral radius tends to `-a_XB·mu0/Kp` as the step
shrinks, which is below -1 whenever active stiffness exceeds passive stiffness,
a routine condition in contracting myocardium. It is neither zero-stable nor
convergent, and refining the time step makes it worse. We therefore add their
consistent stabilization term, which requires every contraction model to report
its active stiffness.

## Considered options

Smaller time steps do not work: non-convergence is the whole point of the
result, and a test suite that only checks accuracy at one step size will not
see it. Sub-iterating each step to a fixed point works and is a true monolithic
reference, but costs a full mechanics solve per iteration. Re-expressing the
models symbolically in UFL recovers a monolithic scheme but discards the
bespoke integrators that are the models' actual value.

## Consequences

Active stiffness becomes a required part of a contraction model's interface,
not an optional extra. The stabilization term is only consistent if the stretch
it measures its increment against is exactly the stretch the model was advanced
with, which makes that identity a correctness invariant rather than a
convention. And because the term is precisely one Newton iteration of the
monolithic scheme, a monolithic backend and this one must converge to each
other — which is an assertion available to tests rather than an assumption.
