---
title: How simcardemsx works
---

A simulation couples two solvers, each on its own mesh and with its own time step, and
exchanges state between them once every mechanics step. This page names the pieces and
shows how a time step goes. The pages after it describe the [models](models.md), the
[coupling schemes](coupling.md), and [mechanics, circulation and
output](mechanics.md).

## The pieces

| Piece | What it is | Where it comes from |
| --- | --- | --- |
| Cell model | One gotranx `.ode` file with EP and contraction, one component marked `"mechanics"` | Your own, or `examples/odefiles/` |
| Generated modules | The EP half as NumPy code and the contraction half as UFL, written by `load_ode_modules` | `simcardemsx.ode_model` |
| EP solver | A `beat.MonodomainSplittingSolver`, whose ODE half steps the EP module at every node | fenicsx-beat |
| Activation backend | Owns the contraction states, and turns them into an active stress for pulse | `simcardemsx.backends` |
| Mechanics problem | A `pulse.StaticProblem` or `pulse.DynamicProblem` whose active model is the backend | fenicsx-pulse |
| Mechanics driver | How one mechanics step is taken: a plain solve, a circulation clock, or a cycle controller | `simcardemsx.mechanics` |
| Transfer plan | Moves what crosses between the EP mesh and the mechanics mesh | `simcardemsx.transfer_plan` |
| Controller | The time loop over all of the above | `simcardemsx.controller` |

## What crosses, and how it is decided

Nothing in a simulation script lists the variables that cross between EP and mechanics.
Each generated module carries three dictionaries, read off the `.ode` file:

- `missing`: what the module needs from the other side;
- `provides`: what its own `missing_values()` hands over;
- `parameter`: its parameters.

`simcardemsx.transfer_plan.resolve(ep_module, backend)` pairs each side's `missing`
against the other's `provides`. If either side needs something the other does not
produce, it raises a `ValueError` naming everything that is missing in both
directions. This guards against a mismatched split or stale generated code. The fibre
stretch $\lambda$ is not produced by the ODE but by the mechanics solve, so it is sent
back to EP whenever the EP module has a parameter called `lmbda`.

The `TransferPlan` then moves the values at run time:

- **forward** (EP → mechanics): it evaluates the EP module's `missing_values()` on beat's
  states and interpolates them to the backend's `inputs`, at the quadrature points of
  the mechanics mesh.
- **backward** (mechanics → EP): it *averages* each backend output onto a P1 or DG0
  space on the mechanics mesh, interpolates that to the EP space, and writes it in
  place into beat's `missing_variables` and `parameters` arrays. Values at quadrature
  points are discontinuous between cells, so pointwise interpolation would take each
  node's value from whichever neighbouring cell happened to be visited last.

## One time step

`SimulationController.step()` advances everything from $t_n$ to
$t_{n+1} = t_n + \Delta t_{mech}$:

```{mermaid}
sequenceDiagram
    participant C as Controller
    participant EP as EP (beat)
    participant T as TransferPlan
    participant B as Backend
    participant M as Mechanics (pulse)
    C->>C: snapshot of every component
    loop Δt_mech / Δt_ep times
        C->>EP: step (ODE, then PDE)
    end
    C->>T: forward(t_{n+1})
    T->>B: inputs
    C->>B: begin_step(t_n, Δt_mech)
    C->>M: driver.advance(t_n, Δt_mech)
    M->>B: S(C) during Newton
    C->>B: post_solve()
    C->>T: backward()
    T->>EP: outputs, λ
```

Nothing in the backend is accepted before `post_solve`. If anything before the
mechanics callback raises, or the solve does not converge, the controller restores the
snapshot. EP's states, the transfer arrays, the displacement, any cavity pressures and
circuit states, and the backend are then exactly as they were before the step. It then
raises. The caller can retry the step, for example with a smaller time step.

When the controller is built it checks that:

- the backend is the mechanics problem's active model;
- $\Delta t_{mech}$ is a whole multiple of $\Delta t_{ep}$;
- a `DynamicProblem`'s own time step equals $\Delta t_{mech}$;
- the backend's quadrature degree is the mechanics form's;
- the EP solver's ODE solver is a `beat.odesolver.DolfinODESolver`.

## Units

simcardemsx works in **milliseconds** throughout, like beat and the `.ode` files. The
exceptions are converted at the edges:

- pulse's `DynamicProblem` time step and `pulse.cycle` are in seconds. The `Cycle` and
  `CirculationClock` drivers convert.
- Stresses in pulse are in pascals. Active tension is reported in kPa
  (`backend.tension_kPa`).
- The crossbridge models work in seconds and µM. `simcardemsx.units` holds the
  conversions, which `CrossbridgeSegregated` applies.
- `simcardemsx.zero_d` works in seconds and pascals, following the paper it reproduces.
