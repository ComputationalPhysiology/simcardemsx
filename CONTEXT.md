# simcardemsx

A cardiac electro-mechanics solver. An electrophysiology model and a mechanics
model, each on its own mesh, exchange state every mechanics time step; a
cellular force-generation model sits between them and turns calcium into
tension.

This file is a glossary. It records what the words mean, not how anything is
built.

## The two physics

**EP**:
The electrophysiology subsystem: a monodomain equation coupled to a cellular
ODE model, advanced on its own mesh at its own time step.
_Avoid_: electrics, electro

**Mechanics**:
The finite-elasticity subsystem: a quasi-static balance of momentum solved for
displacement on its own mesh.

**EP micro-step**:
One EP time step. Many of them run per mechanics step, since EP resolves a
faster time scale.

**Mechanics step**:
One advance of the coupled system. It is the unit of time at which the two
subsystems exchange state, and the step size the coupling scheme's stability is
argued about.

## Force generation

**Activation**:
The production of force by the cellular contraction machinery. Distinct from
mechanics, which is what the tissue does in response.
_Avoid_: contraction (ambiguous between the cause and the effect)

**Activation backend**:
An implementation of the whole path from EP state to active stress: which
variables cross between EP and mechanics, how the contraction model is
advanced, and how its tension enters the mechanics form. Backends are
interchangeable and differ in numerical scheme, not in the physics they claim
to model.
_Avoid_: active model (that is the narrower mechanics-side role a backend also
fills), contraction model

**Contraction model**:
The cellular model of force generation itself — Land2017, RDQ20MF, RDQ18,
Lewalle2024, or the Land variant expressed as an `.ode` file. A backend wraps
one; it is not one.

**Active tension** (`Ta`):
The tension a contraction model currently generates, per unit deformed fibre
cross-section, in kPa.

**Active stiffness** (`Ka`):
The sensitivity of active tension to the rate of stretch, ∂Ṫa/∂λ̇, reported by a
contraction model in the same units as its active tension. Zero for models with
no force–velocity behaviour.

**Distortion state** (`Zetas`, `Zetaw`):
The crossbridge-distortion variables of the Land model, which carry its
dependence on the rate of stretch.
_Avoid_: zeta (bare)

## Kinematics

**Stretch** (`λ`):
Fibre stretch, √(f₀·C f₀). Dimensionless, 1 in the reference configuration.
The mechanics subsystem's measure of fibre length.
_Avoid_: lambda (reserved for the Python keyword sense), elongation

**Sarcomere length** (`SL`):
Fibre length as a contraction model measures it, in µm. Related to stretch by a
chosen reference length.

**Reference sarcomere length** (`SL_ref`):
The sarcomere length the mechanics reference configuration corresponds to. A
property of the simulated tissue.

**Model reference length** (`SL0`):
The sarcomere length a contraction model normalizes its own force–length curve
against. A property of the model. Not every model defines one.

**Normalized sarcomere length** (`Λ`):
`SL/SL0` — the length variable a contraction model's own equations are written
in, and the one it reports active stiffness per unit of. Distinct from stretch,
which is normalized against `SL_ref` instead.

## The split

**Split**:
Where the cellular ODE system is cut between the EP subsystem and the
activation backend. The cut determines which variables have to cross, in both
directions.
_Avoid_: partition, decomposition

**Ca_i split**:
The cut at intracellular calcium: calcium crosses to activation, and the
troponin buffering flux must cross back.

**zeta split**:
The cut at the crossbridge states: `XS` and `XW` cross to activation, and the
distortion states cross back.

**CaTrpn split**:
The cut at troponin-bound calcium: `CaTrpn` crosses to activation and nothing
crosses back.

**Transfer**:
One variable crossing between EP and activation, together with the unit the
producing side emits it in. A bare name is not a transfer: it carries neither
direction nor unit.

**Troponin buffering flux** (`J_TRPN`):
The rate at which the contraction model is binding calcium. It is the return
path of a Ca_i split: without it the EP model's calcium balance is missing a
sink and its transient is wrong.

**Mechano-electric feedback**:
The influence of the mechanics state on the EP solution, carried by whatever a
split sends back toward EP.

## Coupling schemes

**Monolithic**:
A coupling in which activation is re-evaluated at the current mechanics iterate,
so the Newton solve sees the dependence of tension on stretch.

**Segregated**:
A coupling in which activation is advanced once per step against a frozen
stretch, and its tension enters the mechanics solve as fixed data.
_Avoid_: partitioned, staggered, explicit

**Naive segregated**:
The segregated scheme with no correction. Not convergent once active stiffness
exceeds passive stiffness — refining the time step makes it worse. Retained
only to demonstrate the failure.

**Stabilized segregated**:
The segregated scheme with a consistent term that lets the mechanics solve see
active tension as a spring rather than a dead load. Equivalent to one Newton
iteration of the monolithic scheme.

**Active-stress formulation**:
Which of two conventions relates active tension to active stress. Under
`stretch`, the delivered fibre traction equals `Ta` at any stretch; under
`invariant` it is larger by a factor of the stretch. Results produced under one
are not comparable to the other.

**0D tissue model**:
A mass–spring–dashpot stand-in for the mechanics subsystem, with no mesh. It
makes a coupling scheme's behaviour observable without a finite-element solve,
and admits a genuinely monolithic reference solution.
