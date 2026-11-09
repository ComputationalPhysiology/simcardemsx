---
title: Coupling schemes and backends
---

## The problem

The active tension depends on the fibre stretch $\lambda$ and its rate $\dot\lambda$,
and the stretch depends on the tension through the mechanics. Each mechanics step has
to resolve that loop somehow.

The obvious way is to take turns. Step the contraction model with the stretch of the
last step, then solve mechanics with the resulting tension held fixed. Regazzoni and
Quarteroni (2021) showed that this *segregated* scheme has an amplification factor that
tends to $-K_a/K_p$ as $\Delta t \to 0$, where $K_a = \partial T_a/\partial \lambda$ is the
active stiffness and $K_p$ the passive one. Once $K_a > K_p$, which happens in soft,
contracting tissue, the scheme is not convergent: it oscillates from one step to the
next, and a smaller time step makes it worse. The oscillation is easy to mistake for a
Newton tolerance problem or a mesh problem.

[Tutorial 2](../tutorials/02_coupling_schemes_0d.ipynb) shows this in 0D and
[tutorial 3](../tutorials/03_single_element.ipynb) on a finite element.

## The three schemes

| Scheme | What the mechanics solve sees | Converges |
| --- | --- | --- |
| **monolithic** | $T_a$ stepped from the last accepted states with $\lambda(\mathbf{u})$ and $(\lambda(\mathbf{u}) - \lambda_n)/\Delta t$ of the current Newton iterate | yes |
| **stabilized** | $T_a$ stepped with the frozen $\lambda_n$, plus $K_a\,(\lambda(\mathbf{u}) - \lambda_n)$ | yes, first order |
| **segregated** | $T_a$ stepped with the frozen $\lambda_n$, held fixed | no, once $K_a > K_p$ |

The stabilization term of the stabilized scheme is $\mathcal{O}(\Delta t)$, so the
scheme is consistent. It treats the crossbridges as the springs they are during the
solve.

## The backends

An activation backend is a `pulse.ActiveModel`, given to `pulse.CardiacModel` as its
`active`. It owns the contraction states, and its `S(C)` is the active stress. The
`SimulationController` drives any backend that implements the `CoupledBackend`
protocol of `simcardemsx.backends.base`:

| Member | Meaning |
| --- | --- |
| `missing`, `provides` | Name → index dictionaries, as in the generated modules |
| `inputs`, `outputs` | Functions written by the transfer plan before the step, and read after it |
| `quadrature_degree`, `space` | Where the states live |
| `active_tension` | $T_a$ averaged onto P1, in kPa |
| `begin_step(t_n, dt)` | Start a step (times in ms) |
| `post_solve()` | Accept the converged step |

### `GeneratedActivation`

The contraction half of the `.ode` file, generated as UFL. `scheme=` picks
`"monolithic"` (the default), `"stabilized"` or `"segregated"`.

Under `monolithic` the generalized Rush-Larsen step of the whole contraction model sits
inside the stress, as a UFL function of the displacement, so Newton differentiates
through all of it. This works for whatever the `mechanics` component contains, with
nothing ported by hand. Under `stabilized`, $K_a$ is the UFL derivative of the stepped
tension with respect to the step's end stretch. It includes the $1/\Delta t$ of the
rate, as in Regazzoni & Quarteroni's Eq. (44).

```python
backend = GeneratedActivation(
    modules.mechanics,  # from load_ode_modules
    mesh,  # the mechanics mesh
    f0,  # fibre direction
    quadrature_degree=4,  # must be the mechanics form's
    scheme="monolithic",
    parameters={"Tref": 360.0},  # override any parameter of the component
)
```

Besides `inputs` and `outputs`, it exposes `lmbda_prev` (the stretch of the last
accepted step), `tension_kPa` and `stiffness_kPa` ($T_a$ and $K_a$ at the quadrature
points, in kPa), and `reset_stretch()`, which takes the current displacement as the
rest state when a simulation starts from a deformed configuration.

### `CrossbridgeSegregated`

A NumPy contraction model from the
[crossbridge](https://github.com/ComputationalPhysiology/crossbridge) package (`Land2017`,
`RDQ18`, `RDQ20MF` or `Lewalle2024`), on the Ca$_i$ split. A NumPy model cannot be
evaluated inside Newton, so this backend is segregated, and stabilized by default.

```python
backend = CrossbridgeSegregated(f0, mesh, "RDQ20MF", quadrature_degree=4, SL_ref=2.0)
```

`begin_step` advances a copy of the model, and `post_solve` commits it, so a failed
step leaves the model untouched. `RDQ18` has no reference sarcomere length of its own,
so it needs `SL_ref`.

### `ZetaSplitUFL` (deprecated)

The hand-ported Land model of the zeta split. It is kept for code that still imports
it. Use `GeneratedActivation` with the zeta split instead.

## Dynamic problems

`pulse.DynamicProblem` evaluates the stress at the generalized-$\alpha$ point between the
old and new displacement. For a contraction model stepped over the time step this would
blend two stretches and underestimate the stretch rate. All three backends therefore set
`evaluate_at_end_of_step = True`, and pulse evaluates their active stress at the true
end-of-step displacement.

## Reference

F. Regazzoni and A. Quarteroni. An oscillation-free fully staggered algorithm for
velocity-dependent active models of cardiac mechanics. *Comput. Methods Appl. Mech.
Eng.* 373, 113506 (2021).
