"""Activation backends.

Each backend generates active tension and hands it to the mechanics solve, but
they differ in where the EP/mechanics split is cut and in how force generation
is coupled to mechanics. See :mod:`simcardemsx.backends.base` for the
``ActivationBackend`` protocol and why the differences matter.

Only :class:`ZetaSplitUFL` and :class:`CrossbridgeSegregated` actually
implement that protocol. :class:`GeneratedActivation` does not: it has no
``wants_from_ep``/``gives_to_ep``/``ep_inputs`` -- what crosses between EP and
activation is instead derived from the two generated modules by
:func:`simcardemsx.transfer_plan.resolve` and moved by
:class:`simcardemsx.transfer_plan.TransferPlan`, which is what
:class:`~simcardemsx.controller.SimulationController` drives it through.

======================================  ================================  =================
Backend                                 Coupling                          Split
======================================  ================================  =================
:class:`GeneratedActivation`            monolithic in Newton, or naive    any (``.ode``)
                                         segregated (``scheme``)
:class:`ZetaSplitUFL` (deprecated)      monolithic in Newton              zeta's
:class:`CrossbridgeSegregated`          stabilized-segregated (R&Q)       Ca_i
======================================  ================================  =================

An external-operator crossbridge backend, coupling monolithically without
symbolic differentiation, is planned; it would implement the
``ActivationBackend`` protocol, like :class:`CrossbridgeSegregated`.
"""

from .base import ActivationBackend, Transfer
from .generated import GeneratedActivation
from .segregated import CrossbridgeSegregated
from .zeta_split import Scheme, ZetaSplitUFL

__all__ = [
    "ActivationBackend",
    "Transfer",
    "GeneratedActivation",
    "ZetaSplitUFL",
    "CrossbridgeSegregated",
    "Scheme",
]
