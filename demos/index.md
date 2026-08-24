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

## [Declaring what crosses](declaring_what_crosses.py)

What the coupler verifies before it will run: that the activation backend and the
`.ode` file describe the same split, and that they agree about units.

Both failures are silent without a check. The transfer buffers are positional while
backends are named, so a mismatched pair writes calcium into a crossbridge
population and converges happily. A unit mistake between the three libraries that
meet here is a factor of a thousand, and produces a calcium transient that looks
entirely reasonable.

Shows each failure, what the error says, and how to opt out of the unit half if you
would rather not annotate. Needs no mesh; runs in seconds.

## [The return path](mechano_electric_feedback.py)

Coupling is usually described in one direction — calcium drives contraction — and
the path back is easy to leave out. The simulation runs either way.

Under a Ca_i split it is not optional: the contraction model owns troponin, so the
EP model's calcium balance is missing a sink. This runs the same coupled problem
with and without the buffering flux and measures the difference, which is over 12%
of the calcium concentration within 15 ms and grows from there.

This is also the shape of a bug that shipped here for some time: the return path was
wired but transferred zeros, and nothing reported it.

## [When refining the time step makes things worse](zeta_split_time_step.py)

If a coupled run oscillates, the reflex is to shorten the time step. For a
segregated scheme that reflex is wrong — and this demo shows the finite-element
zeta-split path failing to improve as the step is refined, on a single element.

It is the symptom, in the solver you would actually run;
`stabilized_vs_naive_coupling.py` is the diagnosis and the fix, in 0D where a true
monolithic reference is available. Takes a few minutes.
