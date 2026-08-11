# Demos

Runnable examples. Each is a [jupytext](https://jupytext.readthedocs.io) light-format
script — run it directly with `python`, or open it as a notebook.

They need the `demo` extra:

```bash
python3 -m pip install -e ".[demo]"
```

## [Why the coupling scheme matters](stabilized_vs_naive_coupling.py)

Reproduces Test Case 1 of Regazzoni & Quarteroni (2021) in 0D, comparing three ways
of coupling a contraction model to tissue mechanics: monolithic, naive segregated,
and stabilized segregated.

The point it makes is a practical one. Coupling a velocity-sensitive contraction
model to mechanics by simply alternating the two — the obvious approach — produces a
solution that oscillates at the time-step frequency once the tissue is soft enough,
and **refining the time step does not fix it**. The symptom looks like a solver
problem and is not one. One consistent extra term in the active stress removes it
entirely.

Runs in about a minute, needs no mesh, and is the reference the finite-element
backends are validated against.
