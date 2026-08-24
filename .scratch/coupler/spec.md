# The coupler: make the activation-backend protocol real

Status: resolved

## Problem Statement

A researcher who wants to run a cardiac electro-mechanics simulation with a
crossbridge contraction model cannot. The Ca_i-split activation backend exists,
is well tested, and produces correct active tension when handed calcium — but
there is no way to hand it calcium from an actual EP solve. The orchestration
layer predates the activation-backend interface and is wired directly to the
zeta split, so the only contraction model reachable from a real simulation is
the one that was there before.

Worse, the researcher who *is* running the zeta split is getting wrong answers
and has no way to know. The mechano-electric feedback return path is wired but
delivers zeros: the distortion states the backend computes never reach the EP
subsystem, so every zeta-split simulation has been running as though contraction
had no influence on electrophysiology at all. Nothing raises, nothing warns, and
the existing feedback test passes because it exercises the EP model directly
rather than the transfer that feeds it.

Underneath both symptoms is one structural fact: two activation backends
implement an interface that nothing consumes. The interface has been shaped by
what looked reasonable when writing backends, never by what a caller actually
needs, and the parts of it that are wrong have had no way to show it.

## Solution

Write the consumer.

The orchestration layer learns to ask a backend what it needs — which variables
must cross from EP, which must cross back, and in what units — and to move
exactly those, into and out of the Functions the backend owns. The split stops
being a property of the library and becomes a property of the backend, which is
what makes a single simulation configuration able to run either the zeta split
or the Ca_i split by changing one constructor call.

Because the ODE code generator already derives which variables cross, the
coupler can check the backend's declaration against it and refuse to run when
they disagree — turning the most likely configuration error, pairing a backend
with the wrong ODE file, from silently plausible numbers into an exception at
startup. The same treatment applies to units, once the generator records them.

Fixing the return path falls out of this rather than being patched: the source
of the backward transfer becomes the backend's own output Functions, so the
buffer that nothing ever filled ceases to exist.

## User Stories

1. As a researcher, I want to run a coupled simulation with a crossbridge
   contraction model, so that I can use a validated force-generation model
   instead of a hand-ported one.
2. As a researcher, I want to switch between the zeta split and the Ca_i split
   by changing which activation backend I construct, so that I can compare the
   two splits within one process rather than against archived numbers.
3. As a researcher, I want the distortion states my backend computes to actually
   reach the EP subsystem, so that mechano-electric feedback is present in my
   results.
4. As a researcher, I want to be told when my activation backend and my ODE file
   describe different splits, so that I do not silently simulate calcium as
   though it were a crossbridge population.
5. As a researcher, I want to be told when a transfer's units disagree with what
   the ODE file declares, so that a millimolar-for-micromolar mistake fails
   loudly instead of producing a plausible calcium transient.
6. As a researcher supplying my own ODE file, I want those same checks to apply,
   so that the guarantees are not limited to the files shipped with the package.
7. As a researcher, I want the troponin buffering flux to cross back to the EP
   subsystem under a Ca_i split, so that my calcium balance is not missing a
   sink.
8. As a researcher, I want to know that results produced before this change had
   no distortion feedback, so that I can decide what needs re-running rather
   than discovering the discrepancy later.
9. As a researcher, I want the one worked example in the repository to run
   against the current interface, so that I have something correct to copy.
10. As a researcher, I want to choose the function space the activation state
    lives on, so that I can trade interpolation error against cost for my
    problem.
11. As a researcher, I want that choice to default to what the package used
    before, so that this change does not silently move my results.
12. As a researcher comparing splits, I want both backends driven by the same
    orchestration code, so that a difference between them is attributable to the
    split rather than to two different code paths.
13. As a developer adding a third activation backend, I want the interface to
    have been validated by a real caller, so that I am implementing something
    known to be sufficient rather than something that merely looked reasonable.
14. As a developer, I want an activation backend to own the Functions its
    transfers land in, so that constructing a backend does not require an ODE
    model to exist first.
15. As a developer, I want both backends to agree about who owns those
    Functions, so that there is one rule rather than a per-backend convention.
16. As a developer, I want the interface to carry no fields that nothing reads,
    so that implementing a backend does not involve filling in values that have
    no effect.
17. As a developer, I want the mapping from a transfer's name to its position in
    the generated ODE interface to happen in one place, so that positional
    assumptions do not spread through calling code.
18. As a developer, I want to test transfer resolution without constructing a
    mesh, so that the checks are cheap enough to test exhaustively across ODE
    files and backends.
19. As a developer, I want the orchestration layer to name the backend it drives
    in its own signature, so that its most important collaborator is visible
    without reading the body.
20. As a developer, I want to be prevented from passing a backend that is not
    the one assembled into the mechanics form, so that the state I advance is
    the state the solve uses.
21. As a developer, I want the unit conversion for a contraction model to live
    next to that model, so that adding a model does not require editing a
    central conversion table.
22. As a maintainer, I want buffers that nothing writes and structures that
    nothing reads removed, so that the next reader does not have to determine
    which half of the data flow is real.
23. As a maintainer, I want the concern of *what crosses and in what units* kept
    separate from *how it physically moves between non-matching meshes*, so that
    each can be changed without disturbing the other.
24. As a maintainer, I want the tested non-matching-mesh interpolation machinery
    preserved rather than rewritten, so that this change does not put working
    code at risk.
25. As a maintainer, I want the two known bugs covered by regression tests, so
    that a future refactor cannot quietly reintroduce a return path that
    transfers zeros.
26. As a maintainer, I want the coupled test suite to stay fast, so that it keeps
    being run.
27. As a maintainer, I want at least one test against the real ToR-ORd model, so
    that the calcium balance being asserted against is not one we wrote
    ourselves.
28. As a reviewer, I want this phase to be a diff against a main branch that
    already contains the activation backends, so that the coupler is not buried
    among the things it builds on.
29. As a reviewer, I want the tests that encode the known bugs to have been
    written before the fix, so that I can see they failed for the right reason.
30. As a reviewer, I want the change to the worked example's results called out
    explicitly, so that a results change is not mistaken for a refactor.
31. As a future contributor, I want the architecture documentation to describe
    the coupling as it is after this change, so that the description and the
    code agree.

## Implementation Decisions

**The orchestration layer is generalized in place and keeps its name.** Its
shape — EP micro-steps, transfer in, one mechanics solve, backend post-solve,
transfer back — is already right. What is wrong is that two of those steps
bypass the activation backend. A rename would break the worked example and the
architecture documentation for no behavioural gain.

**The controller receives its activation backend explicitly** rather than
reaching it through the mechanics problem, and asserts at construction that the
backend it was given is the one assembled into the mechanics form. The
reach-through works but hides the collaborator, and nothing currently prevents a
caller advancing one backend while the solve uses another.

**A new transfer-resolution module owns what crosses and in what units.** It
builds a resolved plan from an activation backend and the pair of generated ODE
modules. Both generated modules already expose their missing variables as a
name-to-index mapping, which is precisely what is needed to bridge a
name-oriented backend interface to positionally-indexed transfer buffers.

**The plan refuses to resolve when declarations disagree.** If the backend's
declared transfers do not match what the ODE split derives, construction raises.
This is the error the design makes easiest to commit — pairing a Ca_i-split ODE
file with a zeta-split backend — and today it produces plausible wrong numbers
rather than an exception.

**The ODE code generator records units.** Generated modules currently carry no
unit metadata even though the ODE source declares it per quantity. The generator
holds the parsed ODE and already writes both modules, so it appends a unit
mapping to each. This is what allows the unit check to happen at construction
rather than by re-parsing the ODE source.

**A transfer declares the unit it expects to receive**, checked against that
mapping. Conversion itself stays inside the activation backend, on ingest, where
the model that knows its own unit conventions lives. A central conversion table
keyed by variable name is the action-at-a-distance this design exists to avoid.

**The transfer record loses its state-or-monitor field.** Its stated rationale —
that the two need different index lookups — does not hold. The forward path goes
through the generated missing-values function, which requires no lookup, and the
backward path writes positionally. The field is also wrong as currently used: the
troponin buffering flux is declared a monitor but is not one on the EP side; it
is a missing variable.

**The coupler interpolates directly into the Functions a backend owns**, and
reads directly from the ones it exposes for the return path. Backends therefore
always own their transfer Functions; the zeta-split backend loses its ability to
have them injected at construction, so both backends follow one rule. This costs
nothing at runtime, since interpolation writes straight into its target, and it
removes positional hand-wiring from calling code rather than automating it.

**This is what fixes the mechano-electric feedback bug.** The backward transfer's
source becomes the backend's output Functions, so the buffer that nothing ever
wrote is no longer in the path.

**The transfer buffers are thinned to what remains live.** Once the coupler
targets backend-owned Functions, the mechanics-side Functions and value arrays
are dead, as is the separate previous-values structure that is maintained every
step and read by nothing. The non-matching-mesh interpolation machinery is kept
intact — it is tested, it works, and it owns a genuinely different concern.

**The split becomes backend-driven.** The runtime ODE model stops assuming
anything about which variables cross; it serves whatever the resolved plan
describes. This absorbs what was planned as a separate phase, because it cannot
be separated from writing the coupler.

**The activation function space becomes configurable across both backends,
defaulting to what is used today.** The default deliberately does not change in
this phase: moving the activation space changes results, and this phase's purpose
is establishing that the coupling reproduces known references. A disagreement
must be attributable to one thing.

**No runtime assertion is added on the stretch-consistency invariant.** That the
stretch used to advance the contraction model is the one the stabilizer measures
its increment against is currently structural — one place sets both, and the
post-solve hook is deliberately empty. A runtime check would compare a value
against itself. This is worth revisiting only if a backend is added that admits
divergence.

**The worked example is ported in this phase.** It breaks the moment the
zeta-split backend stops accepting injected Functions, and it is the only
end-to-end example in the repository. Leaving it broken across a phase boundary
is how examples rot into being wrong.

## Testing Decisions

**What makes a good test here.** These tests assert what a simulation produces,
not how it is assembled. The existing suite is heavy on tests that pass whether
or not the coupling is right — a positive displacement, a non-zero crossbridge
population — and the bug this phase fixes survived all of them. Every test below
is chosen so that it fails when the coupling is wrong rather than when the code
is broken. In particular, no test may reach past the orchestration layer to
inject the value it then asserts on; that is precisely how the existing feedback
test missed a transfer path delivering zeros.

**Primary seam: the orchestration layer's step method.** All five gate tests sit
here. It is the highest seam available, it already exists as a test seam, and it
is the seam whose behaviour this phase changes. Prior art: the existing coupled
smoke test drives it with a stub EP solver and a mock ODE model, and that harness
is the starting point — extended so the mock carries the missing-variable
mappings the resolved plan now requires.

The five gate tests:

1. **Both backends through one coupler.** The zeta-split backend on a zeta-split
   ODE file and the Ca_i-split backend on a Ca_i-split ODE file, driven by the
   same orchestration code and the same configuration shape. An interface
   conformance test, cheap to keep green.
2. **Isometric clamp.** With stretch pinned to one, the coupled path reproduces a
   standalone contraction-model run driven by the same calcium, to tight
   tolerance. This cleanly separates a coupling bug from a model bug: if it
   fails, the fault is ours.
3. **The troponin buffering flux reaches EP.** Running with the return path
   produces a measurably different calcium transient from running without it.
   Guards the claim that a Ca_i split lacking this return path is silently wrong.
4. **The zeta return path delivers real values.** After stepping, the EP side
   holds the distortion states the backend computed, not zeros. Direct regression
   test for the bug found in this phase, on the backend where it occurred.
5. **Force–velocity monotonicity.** Peak active tension falls monotonically with
   shortening velocity. This catches a sign error in the rate of sarcomere-length
   change, which test 2 structurally cannot see because that rate is zero at
   constant stretch.

**Secondary seam: the transfer-resolution module.** Only the mismatch and unit
checks. Testing them through the controller would require two meshes and a
mechanics problem for what is a comparison between two mappings; keeping this
module constructible from a backend and a pair of generated modules is what makes
these cheap enough to test exhaustively. Prior art: the ODE model tests already
generate modules from both synthetic and real ODE files inside a test.

**Tests are written before the implementation.** Two of the five assert
behaviour that is currently wrong, so they are expected to go red against
existing code for the stated reason, not by import error or fixture mistake. The
failure output is recorded so that a later green result is known to mean
something.

**Fixture strategy.** Wiring tests use a minimal synthetic ODE file built during
the test, as the existing feedback test already does. One test runs the real
ToR-ORd Ca_i split and is marked slow. The suite currently completes in well
under a minute and that is worth protecting, but asserting the buffering-flux
behaviour only against a calcium balance we wrote ourselves would be close to
circular.

**Seams not extended.** The non-matching-mesh interpolation tests and the
backend-direct tests stand as they are. The backend-direct suite in particular
must keep passing unchanged — backends remain constructible and drivable without
any coupler.

## Out of Scope

- **MPI and ghost indexing.** Parallel correctness of the transfer buffers is its
  own phase with its own failure mode, and is the most likely way this design
  breaks without anyone noticing. Not addressed here.
- **The external-operator activation backend.** The monolithic route remains
  planned and is unaffected by this work.
- **Batched matrix exponentials and contraction-model checkpointing upstream.**
  Deferred until a performance benchmark exists to measure against; optimizing
  before the regression tripwire exists means optimizing blind.
- **Changing the default activation function space.** Made configurable here,
  deliberately not changed.
- **Pinning the upstream mechanics and contraction-model libraries.** Decided
  against for now.
- **Quantifying the accuracy difference between the two splits.** This phase
  makes that experiment possible by putting both splits behind one coupler; it
  does not run it.

## Further Notes

**Two bugs were found by auditing the transfer machinery for reads against
writes**, and neither was previously known. One buffer is read as an
interpolation source and written by nothing, which is the mechano-electric
feedback failure. Another structure is written every step and read by nothing,
costing an allocation of Functions on both meshes plus two interpolation
operators for no purpose. Both disappear as a consequence of the design rather
than being patched.

**This phase changes results.** The zeta-split experiment has been running
without distortion feedback into the EP subsystem. Re-running it after this
change will not reproduce what it produced before. That is a correction, but it
is a real change to a path that has been used, and it should be stated plainly in
the commit message rather than folded into a refactor.

**The decision to land the feedback fix inside this work rather than as a
standalone commit** means it is not cherry-pickable onto the main branch on its
own. This was chosen deliberately — a standalone fix would be written against
machinery this phase deletes — but it is worth knowing, because it is easy to
revisit now and awkward later.

**The interface is expected to change under this work, and that is the point.**
One field is already known to be dead. Writing the first real consumer is what
converts the interface from a plausible shape into a validated one, and doing it
with two backends in existence is cheaper than doing it with three.
