# A beating biventricular heart in a closed-loop circulation

[`main.py`](https://github.com/ComputationalPhysiology/simcardemsx/blob/main/examples/circulation_biv/main.py) runs one heart beat of a biventricular (BiV) geometry. EP drives
the contraction, and the two ventricles are closed by Regazzoni's 0D circulation
(Regazzoni et al., 2022). The circulation is solved in the same Newton system as the
displacement, so blood goes round the loop and the total volume is conserved.

It is pulse's `demo/time_dependent/monolithic_3d0d_biv.py`, with the prescribed
tension replaced by simcardemsx's EP-driven activation.

## The set-up

1. **Geometry**: the mean shape of the UK Biobank atlas at end diastole, clipped at the
   valve plane, in metres, with rule-based (LDRB) fibres.
2. **Circulation**: Regazzoni's circuit runs alone to a limit cycle from the mesh's
   volumes, which gives the end-diastolic pressures.
3. **Unloading**: the mesh is prestressed to those pressures to recover its unloaded
   reference, then inflated back to the end-diastolic volumes.
4. **The coupled run**: beat's monodomain on P1 with the EP half of the zeta split
   (the cell model's own stimulus fires everywhere at t = 0), `GeneratedActivation`
   for the contraction, and a `pulse.DynamicProblem` whose cavities are coupled to the
   circuit. `CirculationClock` keeps the circuit's clock, in seconds, in step with the
   controller's, in milliseconds. Mechanics steps are 2 ms and EP steps 0.05 ms, at a
   heart rate of 1 Hz.

## Running

Generating the mesh needs `ukb-atlas`, `fenicsx-ldrb` and pyvista, which come with the
`demo` extra.

```bash
cd examples/circulation_biv
python3 main.py                      # one beat, 1000 ms
python3 main.py --t-end 20           # a quick check
python3 post.py --output-dir output
```

| Option | Meaning |
| --- | --- |
| `--t-end` | End time, in ms (default 1000) |
| `--tref` | Scale on Land's reference tension of 120 kPa (default 3) |
| `--scheme`, `--dt-mech` | Coupling scheme and mechanics time step |
| `--outdir` | Where the results go (default `output/`) |
| `--save-every`, `--save-every-ep`, `--checkpoint-every`, `--restart`, `--overwrite` | As for the [slab](https://github.com/ComputationalPhysiology/simcardemsx/tree/main/examples/slab) |

The mesh, the circulation's operating point and the prestress are cached in
`meshes/ukb_mean_ed_clipped/`. Delete that folder after changing the mesh resolution
or the heart rate.

## Output

Besides the files every application writes (`results.bp`, `log.csv`, checkpoints,
`run.json`), it writes `pv_loops.png`, with the pressure-volume loops and the
pressures, volumes and tension over time, and `summary.json`, which has each
ventricle's volume range, the intervals in which its outflow valve is open and the
volume ejected in each, and the drift in total blood volume.
