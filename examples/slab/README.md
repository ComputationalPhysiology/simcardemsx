# Electro-mechanics on a slab

[`main.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/slab/main.py) couples EP and mechanics on a 2 × 1 × 0.5 mm slab of tissue. It is
[tutorial 4](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/tutorials/04_coupled_slab.py) grown into a program: the split, the
coupling scheme and the contraction model are chosen on the command line, and the run
writes its results for ParaView, checkpoints, and restarts.

- **EP**: beat's monodomain model on P1, with the EP half of the chosen `.ode` file. A
  stimulus current in one corner starts the action potential, and the cell model's own
  stimulus is switched off.
- **Mechanics**: pulse's quasistatic problem, Holzapfel-Ogden and incompressible, with
  rollers on the three faces through the origin.
- **Activation**: `GeneratedActivation` of the `.ode` file's `mechanics` component, or,
  with `--crossbridge`, a crossbridge model through `CrossbridgeSegregated`.

The slab is generated with cardiac-geometriesx on first use, into `meshes/`.

## Running

```bash
cd examples/slab
python3 main.py                                   # zeta split, monolithic, 40 ms
python3 main.py --odefile ../odefiles/ToRORd_dynCl_endo_caisplit.ode --scheme stabilized
python3 main.py --odefile ../odefiles/ToRORd_dynCl_endo_caisplit.ode \
    --crossbridge Land2017 --scheme stabilized --dt-mech 1
python3 post.py --output-dir output/ToRORd_dynCl_endo_zetasplit
```

| Option | Meaning |
| --- | --- |
| `--odefile` | The `.ode` file, and so the split (default: the zeta split) |
| `--scheme` | `monolithic` (default), `stabilized` or `segregated` |
| `--crossbridge` | `Land2017`, `RDQ18`, `RDQ20MF` or `Lewalle2024`; needs the Ca_i split and `--scheme stabilized` or `segregated` |
| `--sl-ref` | Reference sarcomere length in µm (required for `RDQ18`) |
| `--dt-mech`, `--t-end` | Mechanics time step and end time, in ms |
| `--output-dir` | Where the results go (default `output/<odefile stem>/`) |
| `--save-every`, `--save-every-ep`, `--checkpoint-every` | Output intervals, in ms |
| `--restart`, `--overwrite` | Continue from the checkpoint in the output folder, or replace its results |
| `--snapshot-every` | Field snapshots for the scheme comparison, in ms |

## Output

In `--output-dir`:

- `results.bp`: `v` and `cai`, and `u`, `lmbda`, `tension_kPa` and `stiffness_kPa`;
- `log.csv`: one row per mechanics step, with the Newton iterations, mean stretch and
  mean tension;
- `restart.bp` and `restart.json`: the checkpoint that `--restart` continues from;
- `config.resolved.toml`, `run.json` and `timings.json`: the settings, how the run
  ended, and where the time went.

`post.py` writes `post/`: `fields.bp` (everything on P1, for ParaView), traces at a
corner and as volume means, and plots.

A restart may change the run's length and output options, but not its physics. A
restart with another `--scheme`, for example, is refused.
