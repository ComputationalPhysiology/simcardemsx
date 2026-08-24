"""Resolving a backend's declared transfers against the ODE file's split.

The checks here exist because the failure they prevent is silent. The transfer
buffers are positional; the backends are named. Pair a Ca_i-split ODE file with
a zeta-split backend and, without a check, calcium is written into a
crossbridge population and the simulation runs to completion producing
plausible, wrong numbers.

Deliberately constructible without a mesh or a mechanics problem: this is a
comparison between two mappings, and making it cheap is what allows it to be
tested across every combination that matters.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.backends import Transfer
from simcardemsx.transfers import TransferMismatch, resolve


@pytest.fixture
def V():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    return dolfinx.fem.functionspace(mesh, ("DG", 1))


class FakeBackend:
    """A backend reduced to the part `resolve` looks at."""

    def __init__(self, V, wants, gives):
        self._wants = tuple(wants)
        self._gives = tuple(gives)
        self.ep_inputs = {t.name: dolfinx.fem.Function(V, name=t.name) for t in self._wants}
        self.ep_outputs = {t.name: dolfinx.fem.Function(V, name=t.name) for t in self._gives}

    def wants_from_ep(self):
        return self._wants

    def gives_to_ep(self):
        return self._gives


class ZetaLike(FakeBackend):
    def __init__(self, V):
        super().__init__(
            V,
            (Transfer("XS"), Transfer("XW")),
            (Transfer("Zetas"), Transfer("Zetaw")),
        )


class CaiLike(FakeBackend):
    def __init__(self, V):
        super().__init__(V, (Transfer("cai", unit="mM"),), (Transfer("J_TRPN", unit="mM/ms"),))


def _resolve(backend, ep_missing, mech_missing, units=None, V=None):
    ep_space = V
    return resolve(
        backend,
        ep_missing=ep_missing,
        mech_missing=mech_missing,
        units=units or {},
        ep_sources={n: dolfinx.fem.Function(ep_space, name=n) for n in mech_missing},
        ep_targets={n: dolfinx.fem.Function(ep_space, name=n) for n in ep_missing},
    )


def test_matching_declarations_resolve(V):
    plan = _resolve(
        ZetaLike(V),
        ep_missing={"Zetas": 0, "Zetaw": 1},
        mech_missing={"XS": 0, "XW": 1},
        V=V,
    )
    assert [t.name for t in plan.from_ep] == ["XS", "XW"]
    assert [t.name for t in plan.to_ep] == ["Zetas", "Zetaw"]


def test_indices_follow_the_ode_file_not_the_declaration_order(V):
    """The backend declares XS then XW; the ODE file may order them the other
    way. The index must come from the file, or the transfer is transposed."""
    plan = _resolve(
        ZetaLike(V),
        ep_missing={"Zetas": 1, "Zetaw": 0},
        mech_missing={"XS": 1, "XW": 0},
        V=V,
    )
    by_name = {t.name: t.index for t in plan.from_ep}
    assert by_name == {"XS": 1, "XW": 0}
    back = {t.name: t.index for t in plan.to_ep}
    assert back == {"Zetas": 1, "Zetaw": 0}


def test_a_backend_paired_with_the_wrong_split_is_refused(V):
    """The motivating error: a zeta-split backend on a Ca_i-split ODE file.

    Without this the coupler would write calcium into XS -- positionally
    valid, physically nonsense, and silent.
    """
    with pytest.raises(TransferMismatch) as excinfo:
        _resolve(
            ZetaLike(V),
            ep_missing={"J_TRPN": 0},
            mech_missing={"cai": 0},
            V=V,
        )
    message = str(excinfo.value)
    assert "XS" in message and "cai" in message, "the error must name both sides"
    assert "ZetaLike" in message, "the error must name the backend"


def test_the_reverse_pairing_is_also_refused(V):
    with pytest.raises(TransferMismatch):
        _resolve(
            CaiLike(V),
            ep_missing={"Zetas": 0, "Zetaw": 1},
            mech_missing={"XS": 0, "XW": 1},
            V=V,
        )


def test_a_partially_overlapping_split_is_refused(V):
    """Harder than a total mismatch: one name lines up, so a check that only
    counted variables, or only looked at the first, would let it through."""
    with pytest.raises(TransferMismatch) as excinfo:
        _resolve(
            ZetaLike(V),
            ep_missing={"Zetas": 0, "Zetaw": 1},
            mech_missing={"XS": 0, "CaTrpn": 1},
            V=V,
        )
    assert "XW" in str(excinfo.value)


def test_a_declared_unit_disagreeing_with_the_ode_file_is_refused(V):
    with pytest.raises(TransferMismatch, match="micromolar|uM|'mM'"):
        _resolve(
            CaiLike(V),
            ep_missing={"J_TRPN": 0},
            mech_missing={"cai": 0},
            units={"cai": "uM"},
            V=V,
        )


def test_a_matching_unit_resolves(V):
    plan = _resolve(
        CaiLike(V),
        ep_missing={"J_TRPN": 0},
        mech_missing={"cai": 0},
        units={"cai": "mM", "J_TRPN": "mM/ms"},
        V=V,
    )
    assert plan.from_ep[0].unit == "mM"


def test_an_undeclared_unit_is_not_treated_as_a_disagreement(V):
    """None means the source did not say, not that it said "dimensionless".

    This is the common case -- no shipped ODE file declares a unit on a
    crossing variable -- so getting it wrong would reject every real split.
    """
    plan = _resolve(
        CaiLike(V),
        ep_missing={"J_TRPN": 0},
        mech_missing={"cai": 0},
        units={"cai": None, "J_TRPN": None},
        V=V,
    )
    assert plan.from_ep[0].unit == "mM"


def test_a_split_with_no_return_path_resolves(V):
    """The CaTrpn split sends nothing back, and gotranx then omits the entry
    entirely rather than recording an empty one."""
    backend = FakeBackend(V, (Transfer("CaTrpn"),), ())
    plan = _resolve(backend, ep_missing={}, mech_missing={"CaTrpn": 0}, V=V)
    assert plan.to_ep == ()


def test_the_plan_moves_values_into_the_backends_own_functions(V):
    """The forward transfer's target is the backend's Function, not a buffer
    the coupler owns and later copies from."""
    backend = ZetaLike(V)
    plan = _resolve(
        backend,
        ep_missing={"Zetas": 0, "Zetaw": 1},
        mech_missing={"XS": 0, "XW": 1},
        V=V,
    )
    sources = [dolfinx.fem.Function(V) for _ in range(2)]
    sources[0].x.array[:] = 3.0
    sources[1].x.array[:] = 5.0

    plan.push_to_backend(sources)

    assert np.allclose(backend.ep_inputs["XS"].x.array, 3.0)
    assert np.allclose(backend.ep_inputs["XW"].x.array, 5.0)


def test_the_plan_reads_the_return_path_from_the_backend(V):
    """The backward transfer's source is the backend's own output Function.

    It used to be a separate buffer that nothing ever wrote, which is why
    every zeta-split simulation delivered zeros to the EP subsystem.
    """
    backend = ZetaLike(V)
    plan = _resolve(
        backend,
        ep_missing={"Zetas": 0, "Zetaw": 1},
        mech_missing={"XS": 0, "XW": 1},
        V=V,
    )
    backend.ep_outputs["Zetas"].x.array[:] = 7.0
    backend.ep_outputs["Zetaw"].x.array[:] = 11.0

    targets = [dolfinx.fem.Function(V) for _ in range(2)]
    plan.pull_from_backend(targets)

    assert np.allclose(targets[0].x.array, 7.0)
    assert np.allclose(targets[1].x.array, 11.0)
