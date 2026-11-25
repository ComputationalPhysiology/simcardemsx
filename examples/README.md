# Examples

Install the example dependencies first (matplotlib, cardiac-geometriesx, the
circulation model, …):

```bash
python3 -m pip install "simcardemsx[demo]"
```

## Tutorials

Short scripts that build up the ideas one at a time. Each is a
[jupytext](https://jupytext.readthedocs.io) percent-format script, so it runs as
`python3 <script>.py` or opens as a notebook. All of them are executed when the
documentation is built. Tutorials 7 and 8 also need [odezoo](https://github.com/ComputationalPhysiology/odezoo)
and gotranx ≥ 2.11.0, which the `docs` extra installs.

| Tutorial | What it shows | Run time |
| --- | --- | --- |
| [`01_cell_model.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/01_cell_model.py) | The ToR-ORd + Land cell model on its own, and how it is split between EP and mechanics | ~1 min (mostly code generation, cached afterwards) |
| [`02_coupling_schemes_0d.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/02_coupling_schemes_0d.py) | Monolithic, naive segregated and stabilized coupling in 0D, and why the naive scheme does not converge | ~10 s |
| [`03_single_element.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/03_single_element.py) | The same three schemes on one finite element, with `GeneratedActivation` and pulse | ~10 s |
| [`04_coupled_slab.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/04_coupled_slab.py) | EP and mechanics coupled on a slab through `SimulationController` | ~2 min |
| [`05_zero_mech_reference.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/05_zero_mech_reference.py) | The single element against an independent 0D solution from [zero-mech](https://github.com/ComputationalPhysiology/zero-mech) | ~15 s |
| [`06_thick_cylinder.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/06_thick_cylinder.py) | Inflation and active contraction of a thick-walled tube against its exact solution, and second-order mesh convergence | ~45 s |
| [`07_reaction_eikonal_slab.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/07_reaction_eikonal_slab.py) | Reaction-eikonal EP (R-E+ and R-E−) on the Niederer slab against coarse and fine monodomain, then coupled to mechanics, with TT06 + Land composed from [odezoo](https://github.com/ComputationalPhysiology/odezoo) models | ~2.5 min |
| [`08_composed_two_way_coupling.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/08_composed_two_way_coupling.py) | ToR-ORd + Land with two-way coupling (Margara et al.), composed from odezoo models and split two ways with `ODE.split`, against the hand-made files in one cell and coupled on the slab of tutorial 7 | ~4 min |

## Applications

Larger programs with a command line, output in the layout of the pulse and beat CLIs
(`results.h5`, `log.csv`, …), checkpoints and restarts, and a `post.py` that replots a
finished run. Run each from its own directory.

| Example | What it is |
| --- | --- |
| [`slab/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/slab) | EP and mechanics on a slab, for any split, scheme or crossbridge model |
| [`circulation_biv/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/circulation_biv) | A biventricular heart beating in a closed-loop circulation (Regazzoni et al.) |
| [`rodero_05/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/rodero_05) | A Rodero-cohort biventricular heart driven through the five phases of the cardiac cycle |
| [`scheme_comparison/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/scheme_comparison) | Batch runs and a report comparing the three coupling schemes |

`--ep` selects the electrophysiology in the first three: `monodomain` (slab), `synchronous`
(BiV) or `lat-stimulus` (rodero_05) by default, or beat's reaction-eikonal EP with
(`re+`) or without (`re-`) the monodomain PDE. It is physics, so a restart must use the
same one. See each application's README.

## Shared files

- [`odefiles/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/odefiles): the ToR-ORd + Land cell model, split three ways. Each file
  marks a different set of states as the `mechanics` component (see tutorial 1).
  `tt06_land.glue.ode` is the glue with which tutorial 7 composes TT06 and Land
  (`gotranx.compose`), and `torord_land.glue.ode` the glue with which tutorial 8
  composes ToR-ORd and Land, coupled both ways.
- [`demo_io.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/demo_io.py): the output folders, restarts and run records the
  applications share.
