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

from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.backends import Transfer
from simcardemsx.transfers import TransferMismatch, check, resolve


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


def _check(backend, ep_missing, mech_missing, units=None):
    """Name and unit reconciliation only -- no mesh, no function spaces.

    This is the seam ticket 06 asked for: the comparison is between two
    mappings, and keeping it free of dolfinx is what makes it cheap enough to
    check every backend against every split.
    """
    return check(
        backend,
        ep_missing=ep_missing,
        mech_missing=mech_missing,
        units=units or {},
    )


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


def test_each_transfer_is_wired_to_the_function_named_for_it(V):
    """The plan pairs each backend Function with the EP Function for the same
    name, so neither end can drift onto the other's variable.

    Which EP Function belongs to a name is the ODE file's business; see
    `test_ode_model.py` for the row mapping itself.
    """
    backend = ZetaLike(V)
    ep_sources = {n: dolfinx.fem.Function(V, name=n) for n in ("XS", "XW")}
    ep_targets = {n: dolfinx.fem.Function(V, name=n) for n in ("Zetas", "Zetaw")}
    plan = resolve(
        backend,
        ep_missing={"Zetas": 0, "Zetaw": 1},
        mech_missing={"XS": 0, "XW": 1},
        units={},
        ep_sources=ep_sources,
        ep_targets=ep_targets,
    )

    for t in plan.from_ep:
        assert t.ep_function is ep_sources[t.name]
        assert t.function is backend.ep_inputs[t.name]
    for t in plan.to_ep:
        assert t.ep_function is ep_targets[t.name]
        assert t.function is backend.ep_outputs[t.name]


def test_a_backend_paired_with_the_wrong_split_is_refused(V):
    """The motivating error: a zeta-split backend on a Ca_i-split ODE file.

    Without this the coupler would write calcium into XS -- positionally
    valid, physically nonsense, and silent.
    """
    with pytest.raises(TransferMismatch) as excinfo:
        _check(
            ZetaLike(V),
            ep_missing={"J_TRPN": 0},
            mech_missing={"cai": 0},
        )
    message = str(excinfo.value)
    assert "XS" in message and "cai" in message, "the error must name both sides"
    assert "ZetaLike" in message, "the error must name the backend"


def test_the_reverse_pairing_is_also_refused(V):
    with pytest.raises(TransferMismatch):
        _check(
            CaiLike(V),
            ep_missing={"Zetas": 0, "Zetaw": 1},
            mech_missing={"XS": 0, "XW": 1},
        )


def test_a_partially_overlapping_split_is_refused(V):
    """Harder than a total mismatch: one name lines up, so a check that only
    counted variables, or only looked at the first, would let it through."""
    with pytest.raises(TransferMismatch) as excinfo:
        _check(
            ZetaLike(V),
            ep_missing={"Zetas": 0, "Zetaw": 1},
            mech_missing={"XS": 0, "CaTrpn": 1},
        )
    assert "XW" in str(excinfo.value)


def test_a_declared_unit_disagreeing_with_the_ode_file_is_refused(V):
    with pytest.raises(TransferMismatch, match="micromolar|uM|'mM'"):
        _check(
            CaiLike(V),
            ep_missing={"J_TRPN": 0},
            mech_missing={"cai": 0},
            units={"cai": "uM"},
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
    ep_sources = {n: dolfinx.fem.Function(V, name=n) for n in ("XS", "XW")}
    ep_targets = {n: dolfinx.fem.Function(V, name=n) for n in ("Zetas", "Zetaw")}
    plan = resolve(
        backend,
        ep_missing={"Zetas": 0, "Zetaw": 1},
        mech_missing={"XS": 0, "XW": 1},
        units={},
        ep_sources=ep_sources,
        ep_targets=ep_targets,
    )
    ep_sources["XS"].x.array[:] = 3.0
    ep_sources["XW"].x.array[:] = 5.0

    plan.push_to_backend()

    assert np.allclose(backend.ep_inputs["XS"].x.array, 3.0)
    assert np.allclose(backend.ep_inputs["XW"].x.array, 5.0)


def test_the_plan_reads_the_return_path_from_the_backend(V):
    """The backward transfer's source is the backend's own output Function.

    It used to be a separate buffer that nothing ever wrote, which is why
    every zeta-split simulation delivered zeros to the EP subsystem.
    """
    backend = ZetaLike(V)
    ep_sources = {n: dolfinx.fem.Function(V, name=n) for n in ("XS", "XW")}
    ep_targets = {n: dolfinx.fem.Function(V, name=n) for n in ("Zetas", "Zetaw")}
    plan = resolve(
        backend,
        ep_missing={"Zetas": 0, "Zetaw": 1},
        mech_missing={"XS": 0, "XW": 1},
        units={},
        ep_sources=ep_sources,
        ep_targets=ep_targets,
    )
    backend.ep_outputs["Zetas"].x.array[:] = 7.0
    backend.ep_outputs["Zetaw"].x.array[:] = 11.0

    plan.pull_from_backend()

    assert np.allclose(ep_targets["Zetas"].x.array, 7.0)
    assert np.allclose(ep_targets["Zetaw"].x.array, 11.0)


ODEFILES = Path(__file__).parent.parent / "numerical_experiments" / "odefiles"


def _real_maps(tmp_path, name):
    from simcardemsx.ode_model import load_ode_modules

    modules = load_ode_modules(ODEFILES / name, tmp_path / name.replace(".ode", ""))
    # gotranx omits `missing` entirely -- rather than emitting an empty one --
    # when a side needs nothing from the other, which is the CaTrpn split.
    return {
        "ep_missing": getattr(modules.ep, "missing", {}),
        "mech_missing": getattr(modules.mechanics, "missing", {}),
        "units": modules.ep.units,
    }


def _real_backend(kind, mesh):
    from simcardemsx.backends import CrossbridgeSegregated, ZetaSplitUFL

    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))
    if kind == "zeta":
        return ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh)
    return CrossbridgeSegregated(f0=f0, mesh=mesh)


@pytest.mark.parametrize(
    "odefile, kind",
    [
        ("ToRORd_dynCl_endo_zetasplit.ode", "zeta"),
        ("ToRORd_dynCl_endo_caisplit.ode", "cai"),
    ],
)
def test_the_shipped_splits_resolve_against_their_own_backend(tmp_path, odefile, kind):
    """Each shipped split must reconcile with the backend built for it.

    The other tests here use hand-written mappings, which can agree with a
    backend while disagreeing with what gotranx actually derives. This one uses
    the real generated maps.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    wants, gives = check(_real_backend(kind, mesh), **_real_maps(tmp_path, odefile))

    assert wants, "a backend that needs nothing from EP is not a split"
    assert {t.name for t in wants} | {t.name for t in gives}


@pytest.mark.parametrize("kind", ["zeta", "cai"])
def test_the_catrpn_split_matches_neither_shipped_backend(tmp_path, kind):
    """No backend implements the CaTrpn split, so both must be refused.

    Worth pinning: that split moves troponin-bound calcium and needs nothing
    back, so a check that only compared counts, or tolerated an empty return
    path, could wave it through.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    maps = _real_maps(tmp_path, "ToRORd_dynCl_endo_catrpnsplit.ode")

    assert maps["ep_missing"] == {}, "the CaTrpn split sends nothing back"
    assert set(maps["mech_missing"]) == {"CaTrpn"}

    with pytest.raises(TransferMismatch):
        check(_real_backend(kind, mesh), **maps)
