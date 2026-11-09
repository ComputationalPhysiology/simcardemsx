# simcardemsx

Cardiac electro-mechanics in [FEniCSx](https://fenicsproject.org).

simcardemsx couples a model of cardiac electrophysiology (EP) to a model of tissue
mechanics, with the contraction of each cell in between. It does not solve either
physics itself. It connects two solvers that do:

- [fenicsx-beat](https://github.com/finsberg/fenicsx-beat) for EP (the monodomain
  model);
- [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) for the mechanics, from a
  single element to a beating heart in a circulation model.

The cell model, EP and contraction alike, is a single
[gotranx](https://github.com/finsberg/gotranx) `.ode` file. simcardemsx splits it in two
and generates code for each half. The EP half is NumPy code for beat. The contraction
half is UFL, which the mechanics solver can differentiate through, so the contraction
model can be stepped *inside* the mechanics Newton iteration. That removes the
instability the usual staggered coupling suffers from when the tissue is soft
(Regazzoni & Quarteroni, 2021). Models that cannot be written in UFL, such as the NumPy
models of [crossbridge](https://github.com/ComputationalPhysiology/crossbridge), are
coupled with the stabilized scheme of the same paper instead.

📖 **Documentation**: <https://computationalphysiology.github.io/simcardemsx>

## Features

- One `.ode` file for the whole cell, split anywhere between EP and contraction.
  Which variables cross between the two sides is read off the generated code, not
  configured by hand.
- Separate meshes and time steps for EP and mechanics.
- Three coupling schemes: monolithic (the default), stabilized segregated, and the
  naive segregated scheme for comparison.
- Static and dynamic mechanics, closed-loop circulation (Regazzoni's 0D circuit) and
  open-loop cardiac cycles, through pulse.
- Rollback of a failed time step, and checkpoints that a run continues from in a new
  process (bit for bit under pulse's default direct solver).

## Installation

simcardemsx needs [DOLFINx](https://github.com/FEniCS/dolfinx), which is not
installable from PyPI on its own. The simplest way to get it is the DOLFINx container:

```bash
docker run -ti -v $(pwd):/home/shared -w /home/shared ghcr.io/fenics/dolfinx/dolfinx:stable
```

Inside it, install simcardemsx with pip:

```bash
python3 -m pip install simcardemsx
```

or from source, with the dependencies of the examples:

```bash
git clone https://github.com/ComputationalPhysiology/simcardemsx.git
cd simcardemsx
python3 -m pip install -e ".[demo]"
```

The extras are `demo` (the examples), `test`, `docs` and `dev`.

## A first look

Generate the two halves of a cell model and see what crosses between them:

```python
from pathlib import Path

from simcardemsx.ode_model import load_ode_modules

odefile = Path("examples/odefiles/ToRORd_dynCl_endo_zetasplit.ode")
modules = load_ode_modules(odefile, Path("generated_odes"))

print(modules.mechanics.missing)  # what contraction needs from EP: {'XS': 0, 'XW': 1}
print(modules.ep.missing)  # what EP needs back: {'Zetas': 0, 'Zetaw': 1}
```

The contraction half becomes the active stress of a pulse model:

```python
from simcardemsx.backends import GeneratedActivation

backend = GeneratedActivation(modules.mechanics, mesh, f0, quadrature_degree=4)
model = pulse.CardiacModel(material=material, active=backend, compressibility=compressibility)
```

and `SimulationController` steps EP and mechanics together:

```python
from simcardemsx.controller import SimulationController

controller = SimulationController(
    mechanics=problem,  # a pulse.StaticProblem or DynamicProblem using `model`
    ep_solver=ep_solver,  # a beat.MonodomainSplittingSolver
    backend=backend,
    ode_modules=modules,
    dt_mech=1.0,  # ms
    dt_ep=0.05,  # ms
)
while controller.t < 200.0:
    controller.step()
```

The [tutorials](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples) build this up step by step, from one cell to a
coupled slab.

## Examples

[`examples/`](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples) holds four short tutorials and four larger
applications:

- an electro-mechanical slab, for any split, scheme or contraction model;
- a biventricular heart beating in a closed-loop circulation;
- a biventricular heart from the Rodero cohort, driven through the cardiac cycle;
- a convergence study of the coupling schemes.

## Development

```bash
python3 -m pip install -e ".[dev,test]"
python3 -m pytest                 # the test suite
python3 -m pytest -m "not slow"   # the fast part of it
pre-commit run --all              # ruff, mypy, cspell, …
```

The documentation is built with [MyST](https://mystmd.org). See
[`docs/README.md`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/docs/README.md).

## References

- F. Regazzoni and A. Quarteroni. An oscillation-free fully staggered algorithm for
  velocity-dependent active models of cardiac mechanics. *Computer Methods in Applied
  Mechanics and Engineering* 373, 113506 (2021).
- J. Tomek et al. Development, calibration, and validation of a novel human ventricular
  myocyte model in health, disease, and drug block. *eLife* 8, e48890 (2019).
- S. Land et al. A model of cardiac contraction based on novel measurements of tension
  development in human cardiomyocytes. *Journal of Molecular and Cellular Cardiology*
  106, 68–83 (2017).

## Authors

Henrik Finsberg (<henriknf@simula.no>)

## License

MIT. See [LICENSE](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/LICENSE).
