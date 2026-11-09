# A biventricular heart through the cardiac cycle

[`main.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/rodero_05/main.py) runs one beat of a coarse biventricular mesh from the Rodero et al.
(2021) virtual cohort, whose cavities are closed by valve plugs. Each ventricle is
driven through the five phases of the cardiac cycle by `pulse.cycle.CycleController`:

1. **preload**: a pressure ramp;
2. **isovolumic contraction**: volume held at end-diastolic volume;
3. **ejection**: a three-element Windkessel;
4. **isovolumic relaxation**: volume held at end-systolic volume;
5. **filling**: a prescribed inflow.

This is an *open loop*: unlike the [closed loop](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/circulation_biv), ejected
blood leaves the model and filling is prescribed.

The run reproduces the `em_tref7` run of the `rodero_05` case in physcardems, Jenny
Wang's electromechanics stack, on simcardemsx and pulse. The module docstring of
`main.py` lists the known differences from it.

## The set-up

- **EP**: beat's monodomain on P1, with the EP half of the zeta split. Every node
  starts from the steady state of its cell type (endo, mid or epi) and is activated by
  the cell model's own stimulus at its local activation time, read from the case.
- **Activation**: `GeneratedActivation`, monolithic, with the tension masked out of
  the valve plugs through `tension_scale`.
- **Mechanics**: `pulse.DynamicProblem` with a 2 ms step, Holzapfel-Ogden (three times
  stiffer in the plugs), slightly compressible and viscous, with an epicardial spring
  and damper. `Cycle` runs the cycle controller as the controller's mechanics driver.

## Running

The case data is not part of this repository. The example reads it from a checkout of
physcardems, by default `third-party/physcardems/cases/rodero_05`.

```bash
cd examples/rodero_05
python3 main.py --t-end 20               # a quick check
python3 main.py                          # one beat, 800 ms; takes hours
python3 post.py --output-dir output
```

| Option | Meaning |
| --- | --- |
| `--case-dir` | The physcardems case directory |
| `--tref` | Scale on Land's reference tension of 120 kPa (default 7) |
| `--t-end` | End time, in ms (default 800) |
| `--output-dir` | Where the results go (default `output/`) |
| `--save-every`, `--save-every-ep`, `--checkpoint-every`, `--restart`, `--overwrite` | As for the [slab](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/strong_coupling_zetasplit) |

## Output

Besides the files every application writes, `pv_loops.png` and `summary.json`, the
latter with the cycle's phases and whether every step converged. `post.py` also writes
tension statistics per cell type (`active_stats.csv`) and a pseudo-ECG (`ecg.csv`,
`ecg.png`).

The run is not bit-reproducible across processes: compare runs to round-off.
