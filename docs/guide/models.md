---
title: Models
---

## The cell model

The shipped cell models, in `examples/odefiles/`, join two published models in one
[gotranx](https://github.com/finsberg/gotranx) `.ode` file:

- **ToR-ORd** (Tomek et al., 2019), a model of the human ventricular action potential and
  calcium handling, in its dynamic-chloride variant for endocardial cells;
- **Land** (Land et al., 2017), a model of human cardiac contraction.

In the Land model, cytosolic calcium binds to troponin (`CaTrpn`), and the fraction of
blocked tropomyosin (`TmB`) falls. Crossbridges then form, first weakly bound (`XW`) and
then strongly bound (`XS`). Two distortion states, `Zetas` and `Zetaw`, track how far the
attached crossbridges are strained by the stretch rate $\dot\lambda$ of the fibre:

$$
\dot\zeta_s = A_s\,\dot\lambda - c_s\,\zeta_s, \qquad
\dot\zeta_w = A_w\,\dot\lambda - c_w\,\zeta_w .
$$

The active tension is

$$
T_a = h(\lambda)\,\frac{T_{ref}}{r_s}\Big[X_S\,(\zeta_s + 1) + X_W\,\zeta_w\Big],
$$

where $h(\lambda)$ is the length dependence and $T_{ref}$ (120 kPa by default) the
reference tension. The tension therefore depends on the stretch *rate*, and this
velocity dependence is what makes the coupling scheme matter ([Coupling](coupling.md)).

## Splits

Every `.ode` file marks one component `"mechanics"`. Its states are stepped by the
activation backend at the quadrature points of the mechanics mesh, and everything else
is stepped by EP. `simcardemsx.ode_model.load_ode_modules(odefile, output_dir)` cuts the
file there and generates two modules, both with gotranx's generalized Rush-Larsen
scheme:

- `ep_model.py`, NumPy code, for `beat.odesolver.DolfinODESolver`;
- `mechanics_model.py`, the same scheme printed as **UFL** expressions, so the step can
  be evaluated symbolically on the mechanics mesh and differentiated by Newton.

The generated code is cached in `output_dir`, with a stamp of the `.ode` file's hash,
gotranx's version and the generator's own source. It is regenerated only when one of
them changes.

The three shipped splits of ToR-ORd + Land:

| File | Mechanics states | EP → mechanics | Mechanics → EP |
| --- | --- | --- | --- |
| `…_zetasplit.ode` | `Zetas`, `Zetaw`, `Cd` | `XS`, `XW` | `Zetas`, `Zetaw`, $\lambda$ |
| `…_catrpnsplit.ode` | the Land model except troponin | `CaTrpn` | $\lambda$ |
| `…_caisplit.ode` | the whole Land model | `cai` | `J_TRPN` |

The further the cut moves towards EP, the more of the contraction model is evaluated at
the true stretch of the mechanics mesh. On the Ca$_i$ split, EP sees the contraction
only through the troponin buffering flux `J_TRPN`. That split is also the one the NumPy
crossbridge models of `CrossbridgeSegregated` plug into.

[Tutorial 1](../tutorials/01_cell_model.ipynb) runs the whole cell, and
then each split.

### Writing your own split

Any gotranx `.ode` file works if:

- one component is named `"mechanics"`;
- that component has the parameters `lmbda` (fibre stretch) and `dLambda` (stretch
  rate), which the backend sets from the displacement;
- it has a monitor for the tension (`Ta` by default, in kPa, or set `tension=` and
  `tension_unit=` on `GeneratedActivation`).

## Active stress

The tension acts along the fibre direction $f_0$. By default (`formulation="stretch"`)
simcardemsx uses the normalization of Regazzoni & Quarteroni (2021),

$$
\mathbf{P}_a = T_a\,\frac{\mathbf{F} f_0 \otimes f_0}{|\mathbf{F} f_0|},
\qquad |\mathbf{P}_a f_0| = T_a ,
$$

under which the tension is the force per reference area along the fibre whatever the
stretch. The historical simcardems form, `formulation="invariant"`
($\mathbf{P}_a = T_a\,\mathbf{F} f_0 \otimes f_0$), is larger by a factor of $\lambda$. It is
kept only to reproduce old results, which are not comparable with new ones.

`tension_scale=` multiplies $T_a$ in the stress and nothing else. A DG0 mask of 0 and 1
takes the active stress out of part of the domain, for example the valve plugs of the
`rodero_05` example, while the contraction states there keep their own dynamics.

## Passive mechanics

The passive material, compressibility and viscosity are pulse's. The examples use the
Holzapfel-Ogden model (Holzapfel & Ogden, 2009), either incompressible or with pulse's
compressibility penalties. See the [pulse
documentation](https://finsberg.github.io/fenicsx-pulse) for the full list.

## References

- J. Tomek et al. Development, calibration, and validation of a novel human ventricular
  myocyte model in health, disease, and drug block. *eLife* 8, e48890 (2019).
- S. Land et al. A model of cardiac contraction based on novel measurements of tension
  development in human cardiomyocytes. *J. Mol. Cell. Cardiol.* 106, 68–83 (2017).
- G. A. Holzapfel and R. W. Ogden. Constitutive modelling of passive myocardium: a
  structurally based framework for material characterization. *Phil. Trans. R. Soc. A*
  367, 3445–3475 (2009).
- F. Regazzoni and A. Quarteroni. An oscillation-free fully staggered algorithm for
  velocity-dependent active models of cardiac mechanics. *Comput. Methods Appl. Mech.
  Eng.* 373, 113506 (2021).
