---
title: Mechanics, circulation and output
---

## Mechanics problems

simcardemsx uses pulse's problems unmodified, with the activation backend as the
`active` model of the `pulse.CardiacModel`:

- `pulse.StaticProblem`, quasistatic, for a slab or a single element;
- `pulse.DynamicProblem`, with inertia and damping (generalized-$\alpha$), for a beating
  heart with cavities and a circulation.

Solver options belong to whoever builds the problem. The examples tighten pulse's
absolute Newton tolerance `snes_atol` to 1e-9. At resting calcium the first residual is
already at round-off, which a purely relative tolerance cannot reduce further.

A `DynamicProblem`'s own time step, `parameters["dt"]`, is in seconds and compiled into
the form. The controller refuses a problem whose time step differs from `dt_mech`.

## Mechanics drivers

The controller never calls `problem.solve()` itself. It calls `advance(t_n, dt)`, in ms,
on a *mechanics driver* (`simcardemsx.mechanics`). The driver says how one mechanics step
is taken:

| Driver | Use it for |
| --- | --- |
| `Solve(problem)` | A plain solve. A bare problem passed to the controller is wrapped in this |
| `CirculationClock(problem, …)` | A closed-loop circulation solved with the mechanics. Sets the circuit's own clock, in the `.ode` file's time unit |
| `Cycle(controller)` | An open-loop cardiac cycle run by `pulse.cycle.CycleController`, which is in SI units |

### Closed loop

`pulse.circulation.GotranxCirculation` solves Regazzoni's 0D circulation model in the
same Newton system as the displacement. Each 3D cavity replaces the corresponding
chamber of the circuit. Its volume is the volume of the deformed cavity, and its
pressure is the Lagrange multiplier of the constraint. Blood goes round the loop, the
total volume is conserved, and each beat starts where the last one left the circuit.
The [BiV example](../applications/circulation-biv.md) runs one beat.

### Open loop

A controlled cavity (`pulse.problem.CavityControl`) has one pressure unknown whose
equation can be switched at run time: a prescribed volume, a prescribed pressure, or a
pressure tied to the volume as in a Windkessel. `pulse.cycle.CycleController` switches
each cavity through preload, isovolumic contraction, ejection, isovolumic relaxation
and filling. Ejected blood leaves the model, and filling is prescribed. The [Rodero
example](../applications/rodero.md) runs one beat.

The cavity multiplier is the wall pressure only when the cavity surface is closed, or
when its open rim stays in a plane through the origin. Close an open base with valve
plugs or constrain it.

## Rollback and checkpoints

Every component that carries state from one step to the next is *checkpointable*
(`simcardemsx.checkpoint.Checkpointable`): EP's states, the transfer arrays, the
mechanics problem, the cycle controller and the backend. What a beat, pulse or
crossbridge object must save is listed by that library's own restart API.

- `controller.snapshot()` and `controller.restore(snapshot)` copy the whole state in
  memory. The controller takes a snapshot before every step and restores it when the
  step fails.
- `Checkpointer(controller, folder, physics=...)` writes the same state to `restart.bp`
  and `restart.json`, and `restore()` reads it back in a new process.

```python
from simcardemsx.checkpoint import Checkpointer

checkpointer = Checkpointer(controller, outdir, physics=settings)
...
checkpointer.write()  # between steps

# later, in a new process, after building the same objects:
t = checkpointer.restore()
```

`physics` is a dictionary of everything that defines the physics, and its hash is
stored with the checkpoint. A restore refuses a checkpoint written with other physics,
so a restart can change the run's length, output and solver options, but nothing else.
Under pulse's default direct solver a restarted run continues bit for bit. With a lagged
preconditioner, whose factorization cannot be saved, it agrees with the uninterrupted
run to within the solver tolerances.

## Results

`simcardemsx.results` writes output in the layout of the pulse and beat command-line
tools:

| File | Contents |
| --- | --- |
| `config.resolved.toml` | Every setting of the run |
| `results.bp` | Fields over time (`ResultsWriter`), readable by ParaView and `read_results` |
| `log.csv` | One row per step (`CsvLog`) |
| `restart.bp`, `restart.json` | The latest checkpoint |
| `run.json` | How the run ended, with the software versions and git commit (`simcardemsx.provenance`) |
| `timings.json` | Where the time went |

`prepare_output(folder, restart=..., overwrite=...)` refuses to write into a folder that
already holds results, unless the run is a restart of the same physics or is told to
overwrite them. The [applications](../applications/examples.md) all use this layout, and
each has a `post.py` that replots a finished run from these files.
