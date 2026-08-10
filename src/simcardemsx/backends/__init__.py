"""Activation backends.

Each backend generates active tension and hands it to the mechanics solve, but
they differ in where the EP/mechanics split is cut and in how force generation
is coupled to mechanics. See :mod:`simcardemsx.backends.base` for the interface
and why the differences matter.

============================  ==================================  ============
Backend                       Coupling                            Split
============================  ==================================  ============
:class:`ZetaSplitUFL`         monolithic in Newton                zeta's
:class:`CrossbridgeSegregated`  stabilized-segregated (R&Q)       Ca_i
============================  ==================================  ============

An external-operator crossbridge backend, coupling monolithically without
symbolic differentiation, is planned; it shares this interface.
"""

from .base import ActivationBackend, Transfer
from .segregated import CrossbridgeSegregated
from .zeta_split import Scheme, ZetaSplitUFL

__all__ = [
    "ActivationBackend",
    "Transfer",
    "ZetaSplitUFL",
    "CrossbridgeSegregated",
    "Scheme",
]
