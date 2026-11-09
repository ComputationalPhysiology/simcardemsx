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
documentation is built.

| Tutorial | What it shows | Run time |
| --- | --- | --- |
| [`01_cell_model.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/01_cell_model.py) | The ToR-ORd + Land cell model on its own, and how it is split between EP and mechanics | ~1 min (mostly code generation, cached afterwards) |
| [`02_coupling_schemes_0d.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/02_coupling_schemes_0d.py) | Monolithic, naive segregated and stabilized coupling in 0D, and why the naive scheme does not converge | ~10 s |
| [`03_single_element.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/03_single_element.py) | The same three schemes on one finite element, with `GeneratedActivation` and pulse | ~10 s |
| [`04_coupled_slab.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/04_coupled_slab.py) | EP and mechanics coupled on a slab through `SimulationController` | ~2 min |

## Applications

Larger programs with a command line, output in the layout of the pulse and beat CLIs
(`results.bp`, `log.csv`, …), checkpoints and restarts, and a `post.py` that replots a
finished run. Run each from its own directory.

| Example | What it is |
| --- | --- |
| [`slab/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/slab) | EP and mechanics on a slab, for any split, scheme or crossbridge model |
| [`circulation_biv/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/circulation_biv) | A biventricular heart beating in a closed-loop circulation (Regazzoni et al.) |
| [`rodero_05/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/rodero_05) | A Rodero-cohort biventricular heart driven through the five phases of the cardiac cycle |
| [`scheme_comparison/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/scheme_comparison) | Batch runs and a report comparing the three coupling schemes |

## Shared files

- [`odefiles/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/odefiles): the ToR-ORd + Land cell model, split three ways. Each file
  marks a different set of states as the `mechanics` component (see tutorial 1).
- [`demo_io.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/demo_io.py): the output folders, restarts and run records the
  applications share.
