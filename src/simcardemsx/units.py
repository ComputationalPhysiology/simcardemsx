"""Unit conventions at the EP / crossbridge / mechanics boundaries.

Three libraries meet here and none of them agree:

==========  ==========================  =============  ==================
Quantity    EP (beat / ToR-ORd)         crossbridge    mechanics (pulse)
==========  ==========================  =============  ==================
time        ms                          s              s
calcium     mM (``cai``)                uM             --
tension     --                          kPa            kPa
length      --                          um (``SL``)    lambda, [-]
==========  ==========================  =============  ==================

None of these mismatches raises. A calcium transient handed over unconverted
is a thousand times too small, which the contraction model will faithfully
integrate into a plausible-looking flat line. So the conversions live here,
named and tested, rather than inline at the call sites.

The subtle one is :func:`active_stiffness_to_mechanics`. Everything else is a
scale factor on a value; that one is a chain rule, and getting it wrong leaves
the scheme stable but the Newton tangent wrong -- degraded convergence and an
O(1) transient error, with nothing to indicate a problem.
"""

from __future__ import annotations

import numpy.typing as npt

#: ToR-ORd's total troponin concentration [mM] (``trpnmax``). Used to turn a
#: crossbridge model's binding *rate* into the ``J_TRPN`` flux the EP model
#: needs back; see :func:`troponin_flux`.
TRPNMAX_MM = 0.07

MS_PER_S = 1000.0
UM_PER_MM = 1000.0


def ms_to_s(dt_ms: float) -> float:
    """Convert a simulation time step from ms to the seconds crossbridge wants."""
    return dt_ms / MS_PER_S


def s_to_ms(dt_s: float) -> float:
    return dt_s * MS_PER_S


def calcium_to_crossbridge(cai_mM: npt.NDArray) -> npt.NDArray:
    """Convert EP calcium from mM to the micromolar crossbridge expects.

    ToR-ORd's own troponin expression does this inline as ``cai*1000/cat50``,
    which is the same factor.
    """
    return cai_mM * UM_PER_MM


def calcium_from_crossbridge(cai_uM: npt.NDArray) -> npt.NDArray:
    return cai_uM / UM_PER_MM


def stretch_to_sarcomere_length(lmbda: npt.NDArray, SL_ref: float) -> npt.NDArray:
    """Convert mechanical fibre stretch to sarcomere length [um].

    ``SL_ref`` is the sarcomere length of the *reference configuration*, i.e.
    where ``lmbda == 1``. It is a modelling choice, not a property of either
    library: a mesh built from an unloaded geometry does not necessarily sit at
    the contraction model's own slack length ``SL0``.
    """
    return lmbda * SL_ref


def active_stiffness_to_mechanics(Ka: npt.NDArray, SL_ref: float, SL0: float) -> npt.NDArray:
    r"""Rescale active stiffness from the model's stretch variable to the solver's.

    crossbridge reports :math:`K_a = \partial \dot{T_a}/\partial\dot\Lambda` per
    unit of *its own* :math:`\Lambda = SL/SL_0`. The mechanics form multiplies
    it by an increment in the *solver's* stretch :math:`\lambda`, where
    :math:`SL = \lambda\,SL_{ref}`. Hence
    :math:`\Lambda = \lambda\,SL_{ref}/SL_0` and, by the chain rule,

    .. math::
        K_a^{solver} = K_a \cdot \frac{SL_{ref}}{SL_0}

    With the default ``SL_ref == SL0`` the factor is one, which is exactly why
    this is easy to omit and never notice.
    """
    return Ka * (SL_ref / SL0)


def troponin_flux(binding_rate_per_s: npt.NDArray, trpnmax_mM: float = TRPNMAX_MM) -> npt.NDArray:
    r"""Convert a crossbridge calcium binding rate into ToR-ORd's ``J_TRPN``.

    The EP model needs this back whenever troponin has been moved to the
    mechanics side, since its calcium balance reads

    .. math::
        \frac{d\,ca_i}{dt} = B_{ca_i}\left(\ldots - J_{TRPN}\right),
        \qquad J_{TRPN} = \frac{d\,CaTRPN}{dt}\,[TRPN]_{max}

    Returns mM/ms, matching the EP model's own time and concentration units.
    """
    return binding_rate_per_s * trpnmax_mM / MS_PER_S
