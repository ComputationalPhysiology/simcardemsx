"""Activation backends.

Each backend generates active tension and hands it to the mechanics solve, but
they differ in where the EP/mechanics split is cut and in how force generation
is coupled to mechanics. See :mod:`simcardemsx.backends.base` for the interface
and why the differences matter.

============================  ==================================  ============
Backend                       Coupling                            Split
============================  ==================================  ============
:class:`ZetaSplitUFL`         monolithic in Newton                zeta's
============================  ==================================  ============

Segregated and external-operator crossbridge backends are planned; they share
this interface.
"""

from .base import ActivationBackend, Transfer
from .zeta_split import Scheme, ZetaSplitUFL

__all__ = [
    "ActivationBackend",
    "Transfer",
    "ZetaSplitUFL",
    "Scheme",
]
