---
title: Installation
---

## DOLFINx first

simcardemsx is built on [DOLFINx](https://github.com/FEniCS/dolfinx), which is not
installable from PyPI on its own. The simplest way to get it is the official container
image:

```bash
docker run -ti -v $(pwd):/home/shared -w /home/shared ghcr.io/fenics/dolfinx/dolfinx:stable
```

DOLFINx is also available from conda-forge (`conda install -c conda-forge fenics-dolfinx`)
and Spack. See the [DOLFINx installation
instructions](https://github.com/FEniCS/dolfinx#installation).

simcardemsx assumes that dolfinx, mpi4py, petsc4py, basix and ufl are already
installed, and that they are recent enough for the versions of fenicsx-pulse and
fenicsx-beat it requires.

## simcardemsx

From PyPI:

```bash
python3 -m pip install simcardemsx
```

From source:

```bash
git clone https://github.com/ComputationalPhysiology/simcardemsx.git
cd simcardemsx
python3 -m pip install -e .
```

One dependency, [scifem](https://github.com/scientificcomputing/scifem), is compiled
against the DOLFINx you have. If pip fails to build it, install its build requirements
and build it without isolation first:

```bash
python3 -m pip install --group build
python3 -m pip install scifem --no-build-isolation
```

## Extras

| Extra | What it adds |
| --- | --- |
| `demo` | What the [examples](applications/examples.md) need: matplotlib, cardiac-geometriesx, the circulation model, and the mesh and fibre generators of the BiV example |
| `test` | pytest, and what the tests read |
| `docs` | MyST and jupytext, to build this site |
| `dev` | pre-commit, mypy and the type stubs |
| `all` | All of the above |

```bash
python3 -m pip install "simcardemsx[demo]"
```

## The development container

The repository has a [dev container](https://containers.dev) in `.devcontainer/`, built on
the DOLFINx image, which installs everything (`pip install -e .[all]`). VS Code and GitHub
Codespaces open the repository in it directly.

## Checking the installation

```bash
python3 -c "import simcardemsx; print(simcardemsx.__version__)"
python3 -m pytest -m "not slow"     # from a source checkout, with the test extra
```
