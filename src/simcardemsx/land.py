"""Deprecated alias for :mod:`simcardemsx.backends.zeta_split`.

The Land model is now one activation backend among several rather than *the*
active model, so it lives with the others. This module re-exports it under its
old name; import from :mod:`simcardemsx.backends` instead.

Note that ``ZetaSplitUFL`` also absorbed what ``MechanicsProblem.post_solve``
used to do, so it is no longer necessary -- or correct -- to pair it with a
custom problem class. Use ``pulse.StaticProblem`` and call
``backend.post_solve()`` after each solve.
"""

from __future__ import annotations

import warnings

from .backends.zeta_split import Scheme, ZetaSplitUFL, _parameters, _Zeta

__all__ = ["LandModel", "Scheme", "_Zeta", "_parameters"]


class LandModel(ZetaSplitUFL):
    """Deprecated. Use :class:`simcardemsx.backends.ZetaSplitUFL`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "simcardemsx.land.LandModel is deprecated; use "
            "simcardemsx.backends.ZetaSplitUFL instead. Note that it now provides "
            "S/P directly, so pulse.StaticProblem can be used without a custom "
            "MechanicsProblem subclass.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
