# Make activation backends `pulse.ActiveModel` implementations, with `S` as the contract

An activation backend supplies its active stress by implementing
`pulse.ActiveModel` directly, and is handed to `pulse.CardiacModel` unmodified.
The primary contract is the stress `S`, not the strain energy: `StaticProblem`
assembles `S(C)` and never reads `strain_energy`, and most backends have no
closed-form potential to differentiate. A backend without one is a first-class
citizen that raises rather than returning something plausible.

## Considered options

The alternative, and what the code did before, was a `MechanicsProblem`
subclass overriding the material form to add active stress. That works for one
activation model. It does not survive several, because each backend then needs
its own problem subclass and the choice of contraction model leaks into the
choice of solver.

Requiring `strain_energy` instead of `S` was the other option, and it is what
`ActiveModel` superficially suggests. It fails on two of the three backends:
for the zeta split, tension depends on stretch through the distortion states,
so the integral of tension over stretch has no closed form; for an
external-operator backend, tension is opaque by construction.

## Consequences

Swapping the coupling scheme, the split, and the contraction model is one
constructor call rather than a different solver. The mechanics problem subclass
was deleted, and the post-solve update that recomputed stretch and advanced the
contraction state moved onto the backend, where the state it updates lives.
