---
title: simcardemsx
---

Cardiac electro-mechanics in [FEniCSx](https://fenicsproject.org).

simcardemsx couples a model of cardiac electrophysiology (EP) to a model of tissue
mechanics, with the contraction of each cell in between. It connects two solvers:

- [fenicsx-beat](https://github.com/finsberg/fenicsx-beat) for EP (the monodomain model);
- [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) for the mechanics, from a
  single element to a beating heart in a circulation model.

The cell model, EP and contraction alike, is a single
[gotranx](https://github.com/finsberg/gotranx) `.ode` file. simcardemsx splits it in two
and generates code for each half: NumPy for EP, and UFL for the contraction, so that the
contraction model can be stepped *inside* the mechanics Newton iteration. That removes
the instability the usual staggered coupling suffers from when the tissue is soft.

```{mermaid}
flowchart LR
    ode[".ode file<br/>(EP + contraction)"] -->|split + generate| ep["EP half<br/>NumPy"]
    ode -->|split + generate| mech["contraction half<br/>UFL"]
    ep --> beat["beat<br/>monodomain"]
    mech --> backend["activation backend"]
    backend --> pulse["pulse<br/>mechanics"]
    beat <-->|"calcium, crossbridges ⇄ stretch, …"| backend
```

## Where to start

::::{grid} 1 1 2 2

:::{card} Installation
:link: installation.md
Get DOLFINx and install simcardemsx.
:::

:::{card} Tutorials
:link: tutorials/01_cell_model.ipynb
From one cell to a coupled slab, in four steps.
:::

:::{card} User guide
:link: guide/overview.md
How the pieces fit together, and the models behind them.
:::

:::{card} API reference
:link: api/simcardemsx.md
Every module, class and function.
:::

::::

## Features

- One `.ode` file for the whole cell, split anywhere between EP and contraction. Which
  variables cross between the two sides is read off the generated code, not configured
  by hand.
- Separate meshes and time steps for EP and mechanics.
- Three coupling schemes: monolithic (the default), stabilized segregated, and the naive
  segregated scheme for comparison.
- Static and dynamic mechanics, closed-loop circulation and open-loop cardiac cycles,
  through pulse.
- Rollback of a failed time step, and checkpoints that a run continues from in a new
  process.
