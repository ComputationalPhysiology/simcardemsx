"""Tests for :func:`simcardemsx.transfer_plan.resolve`.

`resolve` is the pure half of the transfer plan: given the two modules
generated from one gotranx `.ode` file, it says what crosses between EP and
activation without touching meshes or function spaces. `test_shipped_splits_resolve`
checks it against the table in `.scratch/monolithic-activation/spec.md` (S2) for
all three splits shipped in `numerical_experiments/odefiles`;
`test_a_name_nothing_produces_raises` checks the guard against modules that
don't actually come from the same file.
"""

import types

import pytest

from simcardemsx.transfer_plan import Crossings, resolve


@pytest.mark.parametrize(
    "split, forward, backward, stretch",
    [
        ("caisplit", ("cai",), ("J_TRPN",), False),
        ("zetasplit", ("XS", "XW"), ("Zetas", "Zetaw"), True),
        ("catrpnsplit", ("CaTrpn",), (), True),
    ],
)
def test_shipped_splits_resolve(split_modules, split, forward, backward, stretch):
    ep, mech = split_modules[split]
    assert resolve(ep, mech) == Crossings(forward, backward, stretch)


def test_a_name_nothing_produces_raises():
    ep = types.SimpleNamespace(missing={"Zetas": 0}, provides={"cai": 0}, parameter={})
    mech = types.SimpleNamespace(missing={"cai": 0}, provides={}, parameter={})
    with pytest.raises(ValueError, match="Zetas"):
        resolve(ep, mech)
