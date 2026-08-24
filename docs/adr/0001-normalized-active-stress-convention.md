# Use the normalized (stretch) active-stress convention

Active tension can enter the first Piola-Kirchhoff stress either unnormalized,
`P_a = Ta·F f0⊗f0`, or normalized by the fibre stretch,
`P_a = Ta·F f0⊗f0/|F f0|`. We use the normalized form as the default
everywhere, and keep the unnormalized one selectable only to reproduce
pre-change results.

## Considered options

The unnormalized form is what simcardems used (its eq. 6) and what
`pulse.ActiveStress` defaults to, so it is the incumbent in both the published
record and the upstream library.

We chose the normalized form for three reasons. It is the convention Regazzoni
& Quarteroni derive both the per-model active stiffness formulas and the
stability argument in, so mixing conventions would make the stabilized scheme
inconsistent with its own justification. It is what falls out of writing the
active strain energy as a function of stretch rather than of `I4f`, which is
what makes the stabilizer a genuine potential. And decisively, under it the
delivered fibre traction equals `Ta` regardless of stretch — which is what the
contraction models' own calibration means by tension. Under the unnormalized
form, the tension a model delivers depends on how far the fibre has already
shortened.

## Consequences

The two forms differ by a factor of the stretch, so results are not comparable
across the change. Measured on a single-element contraction the peak stress
differs by ~31%, but peak shortening by only ~2.4%, because the passive law
stiffens exponentially. That ratio is a property of that geometry and that
passive law — it does not transfer. Archived zeta-split results need re-running
rather than rescaling.
