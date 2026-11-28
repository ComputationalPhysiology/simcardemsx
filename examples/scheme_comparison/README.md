# Comparing the coupling schemes

The three coupling schemes of [tutorial 2](https://computationalphysiology.github.io/simcardemsx/coupling-schemes-0d/),
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
| [`run.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/scheme_comparison/run.py) | Runs the [slab](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/slab) or [BiV](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/circulation_biv) under every scheme and time step, one subprocess per run, into `<root>/<geometry>/<split>/<scheme>/dt<dt>/`. A re-run skips the runs that finished |
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

## Results

The three one-element studies (`static-caisplit`, `static-zetasplit` and
`dynamic-zetasplit`) were run at several time steps under every scheme, each against a
monolithic reference at a smaller step, in about 90 s:

```bash
cd examples/scheme_comparison
python3 element.py --output-dir output
python3 report.py --root output --media media
```

`report.py --media` copies each group's error-against-dt and time-series figures into
`media/` and writes `media/provenance.json`: the git commit and whether the tree was
dirty, the library versions, the command, the date, and the `run.json` of every run (the
one-element runs have no `config.resolved.toml`). The naive `segregated` scheme (open
squares: the run failed and was truncated) does not converge, while `stabilized` converges
at first order and `monolithic` is the most accurate. See the
[single-element tutorial](https://computationalphysiology.github.io/simcardemsx/single-element/)
for the one-element problem.

Error against the monolithic reference, against dt (left to right: RMS λ, RMS Ta,
maximum of the mean λ difference):

![static-caisplit: error against dt](media/element_static-caisplit_error_vs_dt.png)
![static-zetasplit: error against dt](media/element_static-zetasplit_error_vs_dt.png)
![dynamic-zetasplit: error against dt](media/element_dynamic-zetasplit_error_vs_dt.png)

Mean λ and mean Ta in time:

![static-caisplit: time series](media/element_static-caisplit_timeseries.png)
![static-zetasplit: time series](media/element_static-zetasplit_timeseries.png)
![dynamic-zetasplit: time series](media/element_dynamic-zetasplit_timeseries.png)

The slab and BiV studies (`run.py --study slab` and `--study biv`, 30 and 12 runs, the
dt 0.05 ones taking hours) are not run here, so they have no figures.
