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
documentation is built. Tutorials 7 and 8 also need [odezoo](https://github.com/ComputationalPhysiology/odezoo),
gotranx ≥ 2.11.0, pyvista, imageio and trame-vtk, which the `docs` extra installs, and they
render off screen: if VTK cannot open a window, see `docs/README.md`.

| Tutorial | What it shows | Run time |
| --- | --- | --- |
| [`01_cell_model.py`](https://computationalphysiology.github.io/simcardemsx/cell-model/) | The ToR-ORd + Land cell model on its own, and how it is split between EP and mechanics | ~1 min (mostly code generation, cached afterwards) |
| [`02_coupling_schemes_0d.py`](https://computationalphysiology.github.io/simcardemsx/coupling-schemes-0d/) | Monolithic, naive segregated and stabilized coupling in 0D, and why the naive scheme does not converge | ~10 s |
| [`03_single_element.py`](https://computationalphysiology.github.io/simcardemsx/single-element/) | The same three schemes on one finite element, with `GeneratedActivation` and pulse | ~10 s |
| [`04_coupled_slab.py`](https://computationalphysiology.github.io/simcardemsx/coupled-slab/) | EP and mechanics coupled on a slab through `SimulationController` | ~2 min |
| [`05_zero_mech_reference.py`](https://computationalphysiology.github.io/simcardemsx/zero-mech-reference/) | The single element against an independent 0D solution from [zero-mech](https://github.com/ComputationalPhysiology/zero-mech) | ~15 s |
| [`06_thick_cylinder.py`](https://computationalphysiology.github.io/simcardemsx/thick-cylinder/) | Inflation and active contraction of a thick-walled tube against its exact solution, and second-order mesh convergence | ~45 s |
| [`07_reaction_eikonal_slab.py`](https://computationalphysiology.github.io/simcardemsx/reaction-eikonal-slab/) | Reaction-eikonal EP (R-E+ and R-E−) on the Niederer slab against coarse and fine monodomain, then coupled to mechanics, with TT06 + Land composed from [odezoo](https://github.com/ComputationalPhysiology/odezoo) models | ~4–5 min |
| [`08_composed_two_way_coupling.py`](https://computationalphysiology.github.io/simcardemsx/composed-two-way-coupling/) | ToR-ORd + Land with two-way coupling (Margara et al.), composed from odezoo models and split two ways with `ODE.split`, against the hand-made files in one cell and coupled on the slab of tutorial 7 | ~4 min |

## Applications

Larger programs with a command line, output in the layout of the pulse and beat CLIs
(`results.h5`, `log.csv`, …), checkpoints and restarts, and a `post.py` that replots a
finished run. Run each from its own directory.

| Example | What it is |
| --- | --- |
| [`slab/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/slab) | EP and mechanics on a slab: ToR-ORd + Land composed from odezoo models and run with any of three splits (`--split`), any scheme or crossbridge model; `compare_splits.py` runs all three splits and `--media` makes the GIFs and figures shown in its README |
| [`rodero_05/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/rodero_05) | A Rodero-cohort biventricular heart driven through the five phases of the cardiac cycle |
| [`scheme_comparison/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/scheme_comparison) | Batch runs and a report comparing the three coupling schemes (`report.py --media` writes the committed figures) |
| [`circulation_biv/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/circulation_biv) | A biventricular heart beating in a closed-loop circulation (Regazzoni et al.) |

The slab's defaults are the composed zeta split, `--dt-mech 1` and `--t-end 400` (one beat).
Its first run fetches the two cell models from odezoo, which needs network access and the
`demo` extra (odezoo, gotranx ≥ 2.11.0); later runs use the cached code. The media options
(`post.py --media`, `compare_splits.py --media`) need pyvista and imageio and render off
screen.

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
