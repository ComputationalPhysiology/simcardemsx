# # Why the coupling scheme matters: stabilized vs naive segregation
#
# When a cell-level contraction model is coupled to tissue mechanics, the two are
# usually advanced in turn: step the contraction model, then solve mechanics with
# the resulting active tension held fixed. This is a *segregated* (staggered)
# scheme, and it is the natural thing to do — it lets each physics keep its own
# solver, its own mesh, and its own time step.
#
# It also fails, in a way that is easy to mistake for a solver problem.
#
# Regazzoni & Quarteroni showed that this scheme's amplification factor tends to
# $-K_a/K_p$ as $\Delta t \to 0$, where $K_a$ is the *active* stiffness of the
# tissue and $K_p$ the passive one. Once $K_a > K_p$ — routine in contracting
# myocardium — that is less than $-1$: the scheme is not merely inaccurate, it is
# **not convergent**. Refining the time step makes it worse.
#
# This demo reproduces their Test Case 1 in 0D, so it runs in seconds and needs no
# mesh, and compares three ways of coupling the same contraction model to the same
# tissue:
#
# - **monolithic** — contraction and mechanics solved together. The reference.
# - **segregated** — the naive staggered scheme.
# - **stabilized** — segregated, plus one consistent extra term.
#
# The fix is small. Instead of treating active tension as a dead load during the
# mechanics solve, treat it as what it physically is — crossbridges behaving as
# springs:
#
# $$
# P_{act} = \left[T_a + K_a\,(\lambda - \lambda_{prev})\right]
#           \frac{\mathbf{F} f_0 \otimes f_0}{|\mathbf{F} f_0|}
# $$
#
# The added term is $\mathcal{O}(\Delta t)$ and vanishes in the limit, so it does
# not change the problem being solved. It just makes the scheme stable.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simcardemsx import zero_d

# ## Setting up
#
# The tissue is R&Q's 0D model, in the quasistatic regime they analyse. `Kp` is
# the knob that decides everything: with a stiff tissue the naive scheme is fine,
# and with a soft one it is not. We will run both.
#
# Note this module works in **seconds and pascals**, following the paper. The rest
# of simcardemsx uses milliseconds.

soft = zero_d.Tissue(Kp=2e4, M=0.0, sigma=0.0)  # Ka will exceed Kp
stiff = zero_d.Tissue(Kp=1e6, M=0.0, sigma=0.0)  # Ka stays below Kp

T_END = 0.4
DT = 1e-3

# The drive is R&Q Eq. (57), an idealised calcium transient peaking at 1.6 µM.

t = np.linspace(0, T_END, 400)
fig, ax = plt.subplots(figsize=(6, 2.4))
ax.plot(t, zero_d.calcium_transient(t), color="#12706B")
ax.set(xlabel="time [s]", ylabel="[Ca²⁺]$_i$ [µM]", title="Stimulus")
fig.tight_layout()

# ## The stable case
#
# With a stiff tissue, all three schemes agree. Nothing here suggests a problem —
# which is the trap.

stable = zero_d.compare(tissue=stiff, dt=DT, T=T_END)
for scheme in zero_d.SCHEMES:
    err = stable.error_against_monolithic(scheme)
    print(
        f"{scheme:12s} peak Ka/Kp = {stable[scheme].Ka.max() / stiff.Kp:5.2f}   error = {err:.2e}",
    )

# ## The unstable case
#
# Soften the tissue so the active stiffness overtakes the passive stiffness during
# the twitch, and the naive scheme comes apart.

unstable = zero_d.compare(tissue=soft, dt=DT, T=T_END)
for scheme in zero_d.SCHEMES:
    r = unstable[scheme]
    ratio = r.Ka.max() / soft.Kp
    print(f"{scheme:12s} peak Ka/Kp = {ratio:5.2f}   oscillation = {r.oscillation_metric:.3f}")

# +
STYLE = {
    "monolithic": dict(color="#16191C", lw=2.4, label="monolithic (reference)"),
    "segregated": dict(color="#B33F0C", lw=1.2, label="segregated (naive)"),
    "stabilized": dict(color="#12706B", lw=1.6, ls="--", label="stabilized"),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
for col, (name, comparison, tissue) in enumerate(
    [("stiff tissue: $K_a < K_p$", stable, stiff), ("soft tissue: $K_a > K_p$", unstable, soft)],
):
    for scheme in zero_d.SCHEMES:
        r = comparison[scheme]
        axes[0, col].plot(r.t, r.Ta / 1000.0, **STYLE[scheme])
        axes[1, col].plot(r.t, r.stretch, **STYLE[scheme])
    axes[0, col].set_title(name)
    axes[1, col].set_xlabel("time [s]")

axes[0, 0].set_ylabel("active tension [kPa]")
axes[1, 0].set_ylabel("fibre stretch $\\lambda$ [-]")
axes[0, 1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "stabilized_vs_naive.png", dpi=150)
# -

# The right-hand column is the point. The naive scheme oscillates every step,
# while the stabilized one tracks the monolithic reference closely enough to be
# hard to distinguish from it.
#
# ## Refinement does not save the naive scheme
#
# This is the part that makes the instability worth understanding rather than just
# avoiding. A scheme that is merely inaccurate gets better as you refine. This one
# does not.

print(f"{'dt':>8} | {'naive osc':>10} {'naive err':>10} | {'stab osc':>9} {'stab err':>10}")
errors: dict[str, list[float]] = {"segregated": [], "stabilized": []}
dts = [4e-3, 2e-3, 1e-3, 5e-4]
for dt in dts:
    c = zero_d.compare(tissue=soft, dt=dt, T=T_END)
    for name in errors:
        errors[name].append(c.error_against_monolithic(name))
    print(
        f"{dt:8.0e} | {c['segregated'].oscillation_metric:10.3f} {errors['segregated'][-1]:10.2e} "
        f"| {c['stabilized'].oscillation_metric:9.3f} {errors['stabilized'][-1]:10.2e}",
    )

# +
fig, ax = plt.subplots(figsize=(5.5, 4))
for scheme in ("segregated", "stabilized"):
    ax.loglog(dts, errors[scheme], "o-", color=STYLE[scheme]["color"], label=STYLE[scheme]["label"])
ax.loglog(dts, np.array(dts) * errors["stabilized"][0] / dts[0], "k:", lw=1, label="first order")
ax.set(xlabel="$\\Delta t$ [s]", ylabel="max strain error vs monolithic")
ax.legend(frameon=False)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(Path(__file__).parent / "convergence.png", dpi=150)
# -

# The stabilized scheme converges at first order, as its consistency argument
# says it should. The naive one flattens out, and its error actually *grows* as
# the step shrinks — there is no time step small enough to fix it.
#
# ## What this means in practice
#
# Any contraction model that responds to shortening *velocity* — which is most of
# them, since that is what produces the force-velocity relationship — will do this
# when coupled naively to a soft enough tissue. The symptom is a solution that
# oscillates at the time-step frequency, which is easy to misread as a Newton
# tolerance problem or a mesh issue.
#
# In simcardemsx the stabilization is on by default:
#
# ```python
# from simcardemsx.backends import CrossbridgeSegregated
#
# backend = CrossbridgeSegregated(f0=f0, mesh=mesh, model="Land2017")
# ```
#
# `stabilized=False` exists only to reproduce the failure shown above.
#
# The alternative is to keep the coupling monolithic, which the
# `ZetaSplitUFL` backend does by rebuilding the contraction model symbolically in
# UFL so that Newton re-linearises it every iteration. That is stable without any
# extra term, but it requires the contraction model to be expressible in UFL —
# which a NumPy model like those in `crossbridge` is not.

plt.show()
