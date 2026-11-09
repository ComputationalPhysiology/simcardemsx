# Comparing the coupling schemes

The three coupling schemes of [tutorial 2](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/02_coupling_schemes_0d.py),
compared on the same problems at several time steps:

- **monolithic**: the contraction model stepped inside the mechanics Newton iteration;
- **segregated**: the naive staggered scheme, which is not convergent once the active
  stiffness exceeds the passive stiffness;
- **stabilized**: segregated plus Regazzoni & Quarteroni's consistent stabilization
  term.

## The scripts

| Script | What it does |
| --- | --- |
| [`element.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/scheme_comparison/element.py) | One-element studies with prescribed inputs and no EP: `static-caisplit`, `static-zetasplit` and `dynamic-zetasplit`, each against a monolithic reference at a much smaller time step |
| [`run.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/scheme_comparison/run.py) | Runs the [slab](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/strong_coupling_zetasplit) or [BiV](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/circulation_biv) under every scheme and time step, one subprocess per run, into `<root>/<geometry>/<split>/<scheme>/dt<dt>/`. A re-run skips the runs that finished |
| [`report.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/scheme_comparison/report.py) | Reads such a folder of runs and writes `report.md` and figures: the error of each run against the monolithic reference, and the observed orders of convergence |
| [`record.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/scheme_comparison/record.py), [`metrics.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/scheme_comparison/metrics.py) | The per-step recorder the slab and BiV write `steps.csv` with, and the error metrics |

## Running

```bash
cd examples/scheme_comparison
python3 element.py --output-dir results
python3 run.py --study slab --root results
python3 report.py --root results
```

`run.py --dry-run` lists the runs without starting them. The naive scheme is expected
to fail on some runs, and its failures are part of the result.
