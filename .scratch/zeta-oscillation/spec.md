# The zeta-split FEM path oscillates at the time-step frequency

Status: ready-for-human

Deferred out of the coupler PR deliberately. Recorded here so the next attempt
does not have to re-derive the evidence.

## What happens

`numerical_experiments/strong_coupling_zetasplit` produces a fibre stretch that
reverses direction on almost every mechanics step — a period-2 limit cycle, not
a contraction:

```
0.91528, 0.90257, 0.91510, 0.90245, 0.91490, 0.90230 ...
```

Measured over 30 ms of simulated time on the shipped slab geometry: 270
direction reversals in 600 steps, amplitude ~0.015 in lambda.

## It predates the coupler work

The same experiment run at `main` (0bccb0d) with the pre-change code gives 267
reversals in 599 steps, amplitude and character identical. The refactor did not
introduce it, and slightly reduced the amplitude — peak active tension fell from
~12.4 kPa to ~4.3 kPa once distortion feedback actually reached the EP model.

## It is a segregated instability, not stiffness

Refining the time step makes it worse, which is the diagnostic that separates
the two. On the slab geometry over 20 ms:

| `dt_mech` | reversals/step |
|-----------|----------------|
| 0.05 ms   | 0.18           |
| 0.20 ms   | 0.04           |
| 0.80 ms   | 0.00           |

So no Newton tolerance and no smaller step will fix it. This is the Regazzoni &
Quarteroni result the 0D reference in `zero_d.py` already characterizes.

It is also not a soft-tissue edge case. On a single element, stiffening the
passive material by 20x and 50x leaves the reversal fraction near 0.75. The
amplitude is bounded by the contraction model's own `h(lambda)` force-length
clamp rather than by anything numerical, which is why it presents as a stable
oscillation rather than a divergence.

`demos/zeta_split_time_step.py` reproduces all of this on one element.

## Why this is surprising

`CLAUDE.md` and the design proposal both state that the zeta split is
"monolithic-in-Newton by construction", and that this is what makes it stable.
The zeta states *are* rebuilt inside every Newton iteration, so the claim is not
obviously wrong.

The suspected mechanism, **not yet proven**: `dLambda = (lmbda - lmbda_prev)/dt`
carries `lmbda_prev` explicitly, and its coefficient scales as `1/dt`. That
would make the effective active stiffness grow without bound as the step
shrinks, which matches the measured trend. Confirming or refuting this is the
first task, because the fix depends on it.

## Possible directions

- If the mechanism above is right, the stabilization already proven in 0D
  applies: `pulse.StabilizedActiveStress` is in place and `CrossbridgeSegregated`
  already uses it. `ZetaSplitUFL` supplies `S` directly and does not.
- Sub-iterating the mechanics solve to a fixed point within a step would confirm
  the diagnosis even if it is too slow to ship.
- The 0D reference cannot arbitrate directly: it drives crossbridge models, not
  `ZetaSplitUFL`.

## Related, and worth fixing alongside

The two backends report `active_tension` in **different units**.
`ZetaSplitUFL.Ta_current` holds Pa — its `Ta` multiplies by 1000 to match
pulse's internal Pa convention. `CrossbridgeSegregated.Ta_current` holds kPa and
declares that to pulse via `units.Variable`, which converts.

Each is internally correct, but `DataCollector` writes both under the name "Ta",
and any cross-backend comparison — the split-accuracy experiment the design
proposal calls for — would be wrong by a factor of 1000.
