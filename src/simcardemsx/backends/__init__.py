"""Activation backends.

Each backend generates active tension and hands it to the mechanics solve, but
they differ in where the EP/mechanics split is cut and in how force generation
is coupled to mechanics. See :mod:`simcardemsx.backends.base` for the
``CoupledBackend`` protocol and why the differences matter.

The controller drives a backend through :class:`CoupledBackend`;
:class:`GeneratedActivation` and :class:`CrossbridgeSegregated` are two. What crosses
between EP and activation is derived from the EP module and the backend's
``missing``/``provides`` by :func:`simcardemsx.transfer_plan.resolve` and moved by
:class:`simcardemsx.transfer_plan.TransferPlan`. :class:`ZetaSplitUFL` is driven
directly against ``pulse`` and declares its crossings with :class:`Transfer`.

======================================  ================================  =================
Backend                                 Coupling                          Split
======================================  ================================  =================
:class:`GeneratedActivation`            monolithic in Newton, or naive    any (``.ode``)
                                         segregated (``scheme``)
:class:`ZetaSplitUFL` (deprecated)      monolithic in Newton              zeta's
:class:`CrossbridgeSegregated`          stabilized-segregated (R&Q)       Ca_i
======================================  ================================  =================

An external-operator crossbridge backend, coupling monolithically without
symbolic differentiation, is planned; it would implement :class:`CoupledBackend`.
"""

from .base import CoupledBackend, Transfer
from .generated import GeneratedActivation
from .segregated import CrossbridgeSegregated
from .zeta_split import Scheme, ZetaSplitUFL

__all__ = [
    "CoupledBackend",
    "Transfer",
    "GeneratedActivation",
    "ZetaSplitUFL",
    "CrossbridgeSegregated",
    "Scheme",
]
