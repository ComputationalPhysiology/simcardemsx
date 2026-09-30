"""Pure analysis functions for the scheme comparison (numpy/scipy only)."""

from collections.abc import Sequence

import numpy as np
import scipy.optimize

#: Candidate floors on |Δλ|/dt (per ms) below which an increment is round-off noise
#: and not a change of direction.
FLOORS_PER_MS: tuple[float, ...] = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def weighted_rms(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w * x**2) / np.sum(w)))


def reversal_fraction(d_prev: np.ndarray, d_curr: np.ndarray, w: np.ndarray, floor: float) -> float:
    """Weighted fraction of points whose increment changes sign, both above ``floor``.

    ``floor`` is an increment (of λ over one step), not a rate.
    """
    d_prev = np.asarray(d_prev)
    d_curr = np.asarray(d_curr)
    flipped = (d_prev * d_curr < 0) & (np.abs(d_prev) > floor) & (np.abs(d_curr) > floor)
    return float(np.sum(w[flipped]) / np.sum(w))


def onset_index(
    fractions: Sequence[float],
    threshold: float = 0.5,
    consecutive: int = 3,
) -> int | None:
    """First index starting ``consecutive`` steps in a row at or above ``threshold``."""
    above = [f >= threshold for f in fractions]
    for i in range(len(above) - consecutive + 1):
        if all(above[i : i + consecutive]):
            return i
    return None


def self_convergence_order(
    dts: tuple[float, float, float],
    diff_coarse: float,
    diff_fine: float,
) -> float:
    """Order p with ``(d0^p − d1^p)/(d1^p − d2^p) = diff_coarse/diff_fine``.

    ``diff_coarse`` is the difference between the solutions at ``dts[0]`` and
    ``dts[1]``, ``diff_fine`` between those at ``dts[1]`` and ``dts[2]``.
    """
    d0, d1, d2 = dts
    target = diff_coarse / diff_fine

    def residual(p: float) -> float:
        return (d0**p - d1**p) / (d1**p - d2**p) - target

    return float(scipy.optimize.brentq(residual, 0.05, 5.0))


def reference_orders(dts: Sequence[float], errors: Sequence[float]) -> list[float]:
    """Observed order between each pair of successive ``dts`` from their errors."""
    return [
        float(np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1]))
        for i in range(len(dts) - 1)
    ]


def richardson(
    fine: np.ndarray,
    coarse: np.ndarray,
    ratio: float,
    order: float = 1.0,
) -> np.ndarray:
    """Extrapolate to dt → 0 from ``coarse`` (dt) and ``fine`` (dt/ratio)."""
    r = ratio**order
    return (r * np.asarray(fine) - np.asarray(coarse)) / (r - 1.0)


def naive_threshold_kPa(
    dt_ms: float,
    *,
    Kp_kPa: float,
    eta_Pa_s: float,
    rho_kg_m3: float,
    h_m: float,
) -> float:
    """Active stiffness above which the naive scheme is unstable: ``Kp + η/dt + ρh²/dt²``."""
    dt_s = dt_ms * 1e-3
    return Kp_kPa + (eta_Pa_s / dt_s + rho_kg_m3 * h_m**2 / dt_s**2) / 1e3
