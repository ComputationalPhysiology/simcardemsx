"""The two coupled backends as ``Checkpointable``: restart functions, metadata, and a
restore that continues bit for bit.

Each backend is driven directly (``register`` / ``begin_step`` / ``post_solve``) with
prescribed inputs and a displacement that changes between steps, as in
``test_generated_activation.py`` and ``test_segregated_backend.py``.
"""

from __future__ import annotations

import json

from mpi4py import MPI

import crossbridge
import dolfinx
import numpy as np
import pytest
from conftest import _f0, calcium

from simcardemsx.backends import CrossbridgeSegregated, GeneratedActivation
from simcardemsx.checkpoint import Checkpointable

STRETCHES = [1.0, 1.02, 1.05, 1.03, 0.98, 0.97]
DT = 1.0


def _mesh() -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)


def _displacement(mesh) -> dolfinx.fem.Function:
    return dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 2, (3,))))


def _set_stretch(u: dolfinx.fem.Function, stretch: float) -> None:
    u.interpolate(
        lambda x: np.vstack([(stretch - 1.0) * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])]),
    )


def _step(backend, u, n: int) -> None:
    """Step ``n``: prescribed calcium, a displacement that changes, accepted."""
    _set_stretch(u, STRETCHES[n])
    for name in backend.inputs:
        backend.inputs[name].x.array[:] = calcium((n + 1) * DT)
    backend.begin_step(n * DT, DT)
    backend.post_solve()


def _snapshot(backend) -> dict[str, np.ndarray]:
    return {name: f.x.array.copy() for name, f in backend.restart_functions()}


def _restore(src, dst) -> None:
    """What the checkpointer does: values and (JSON round-tripped) metadata of ``src``
    into a fresh call of ``dst.restart_functions()``."""
    values = _snapshot(src)
    metadata = json.loads(json.dumps(src.restart_metadata()))
    functions = dst.restart_functions()
    assert [name for name, _ in functions] == list(values)
    for name, f in functions:
        f.x.array[:] = values[name]
    dst.load_restart(functions, metadata)


def _assert_same(a, b) -> None:
    sa, sb = _snapshot(a), _snapshot(b)
    assert sa.keys() == sb.keys()
    for name in sa:
        assert np.array_equal(sa[name], sb[name]), name
    assert np.array_equal(a.active_tension.x.array, b.active_tension.x.array)


def _generated(mech, scheme):
    mesh = _mesh()
    backend = GeneratedActivation(mech, mesh, _f0(mesh), quadrature_degree=2, scheme=scheme)
    u = _displacement(mesh)
    backend.register(u)
    return backend, u


@pytest.mark.parametrize("scheme", ["monolithic", "segregated", "stabilized"])
def test_generated_activation_restores_bit_for_bit(scheme, split_modules):
    _, mech = split_modules["caisplit"]
    a, ua = _generated(mech, scheme)
    for n in range(3):
        _step(a, ua, n)
    b, ub = _generated(mech, scheme)
    _set_stretch(ub, STRETCHES[2])
    _restore(a, b)
    _assert_same(a, b)
    for n in range(3, 5):
        _step(a, ua, n)
        _step(b, ub, n)
    _assert_same(a, b)
    assert np.any(a.tension_kPa.x.array != 0.0)


def test_generated_activation_names_and_metadata(split_modules):
    _, mech = split_modules["caisplit"]
    a, _ = _generated(mech, "stabilized")
    names = [name for name, _ in a.restart_functions()]
    assert names == [
        "activation_states_prev",
        "activation_lmbda_prev",
        "activation_lmbda_frozen",
        "activation_dLambda_frozen",
        "activation_tension_kPa",
        "activation_stiffness_kPa",
        *(f"activation_output_{name}" for name in sorted(a.outputs)),
    ]
    assert a.restart_metadata() == {
        "backend": "GeneratedActivation",
        "scheme": "stabilized",
        "state_names": sorted(mech.state, key=mech.state.__getitem__),
    }
    assert a.namespace == "activation"
    assert a.step_pending is False


def _in_another_order(backend) -> tuple[dict[str, np.ndarray], dict]:
    """``backend``'s restart values and metadata as a process that ordered the module's
    states otherwise would have written them: ``state_names`` rotated by one, and
    ``states_prev``'s components with them. A rotation, unlike a reversal, is not its
    own inverse, so a restore that applied the inverse permutation would not pass."""
    values = _snapshot(backend)
    metadata = json.loads(json.dumps(backend.restart_metadata()))
    names = metadata["state_names"]
    reordered = names[1:] + names[:1]
    components = values["activation_states_prev"].reshape(-1, len(names))
    values["activation_states_prev"] = components[:, [names.index(n) for n in reordered]].ravel()
    metadata["state_names"] = reordered
    return values, metadata


def _load(dst, values: dict[str, np.ndarray], metadata: dict) -> None:
    functions = dst.restart_functions()
    for name, f in functions:
        f.x.array[:] = values[name]
    dst.load_restart(functions, metadata)


@pytest.mark.parametrize("scheme", ["monolithic", "stabilized"])
def test_generated_activation_restores_states_saved_in_another_order(scheme, split_modules):
    """A checkpoint whose ``state_names`` are in another order, with ``states_prev``'s
    components in that order, is restored by name: bit for bit, and it continues so."""
    _, mech = split_modules["caisplit"]
    assert len(mech.state) > 2
    a, ua = _generated(mech, scheme)
    for n in range(3):
        _step(a, ua, n)
    values, metadata = _in_another_order(a)
    assert not np.array_equal(values["activation_states_prev"], a.states_prev.x.array)

    b, ub = _generated(mech, scheme)
    _set_stretch(ub, STRETCHES[2])
    _load(b, values, metadata)
    _assert_same(a, b)
    for n in range(3, 5):
        _step(a, ua, n)
        _step(b, ub, n)
    _assert_same(a, b)


def test_generated_activation_refuses_other_state_names(split_modules):
    """Saved state names that are not this module's, in any order, are refused."""
    _, mech = split_modules["caisplit"]
    a, _ = _generated(mech, "monolithic")
    values, metadata = _in_another_order(a)
    b, _ = _generated(mech, "monolithic")
    for names in (
        [*metadata["state_names"][:-1], "not_a_state"],
        metadata["state_names"][:-1],
        [],
    ):
        with pytest.raises(ValueError, match="activation states"):
            _load(b, values, {**metadata, "state_names": names})
    missing = {key: value for key, value in metadata.items() if key != "state_names"}
    with pytest.raises(ValueError, match="activation states"):
        _load(b, values, missing)


def _crossbridge(model, f0_mesh=None):
    mesh = _mesh()
    backend = CrossbridgeSegregated(
        f0=_f0(mesh),
        mesh=mesh,
        model=model,
        quadrature_degree=2,
        SL_ref=2.0 if model == "RDQ18" else None,
    )
    u = _displacement(mesh)
    backend.register(u)
    return backend, u


@pytest.mark.parametrize("model", sorted(crossbridge.MODEL_REGISTRY))
def test_crossbridge_restores_bit_for_bit(model):
    a, ua = _crossbridge(model)
    for n in range(3):
        _step(a, ua, n)
    b, ub = _crossbridge(model)
    _set_stretch(ub, STRETCHES[2])
    _restore(a, b)
    _assert_same(a, b)
    assert np.array_equal(a._lmbda_old, b._lmbda_old)
    for n in range(3, 5):
        _step(a, ua, n)
        _step(b, ub, n)
    _assert_same(a, b)
    assert np.array_equal(a._lmbda_old, b._lmbda_old)


def test_crossbridge_names_and_metadata():
    a, _ = _crossbridge("RDQ20MF")
    names = [name for name, _ in a.restart_functions()]
    assert names[:5] == [
        "activation_lmbda_prev",
        "activation_lmbda_old",
        "activation_tension_kPa",
        "activation_stiffness_kPa",
        "activation_output_J_TRPN",
    ]
    arrays = sorted(k for k, v in a.model.get_state().items() if isinstance(v, np.ndarray))
    assert names[5:] == [f"activation_model_{k}" for k in arrays]
    md = json.loads(json.dumps(a.restart_metadata()))
    assert md["backend"] == "CrossbridgeSegregated"
    assert md["model"] == "RDQ20MF"
    assert md["stabilized"] is True
    assert md["model_scalars"] == {
        k: v for k, v in a.model.get_state().items() if not isinstance(v, np.ndarray)
    }
    assert a.namespace == "activation"


def test_load_restart_refuses_other_metadata(split_modules):
    _, mech = split_modules["caisplit"]
    a, _ = _generated(mech, "monolithic")
    b, _ = _generated(mech, "segregated")
    with pytest.raises(ValueError, match="scheme"):
        b.load_restart(b.restart_functions(), a.restart_metadata())

    x, _ = _crossbridge("Land2017")
    y, _ = _crossbridge("Lewalle2024")
    with pytest.raises(ValueError, match="model"):
        y.load_restart(y.restart_functions(), x.restart_metadata())


def test_crossbridge_load_restart_discards_a_pending_step():
    a, ua = _crossbridge("Land2017")
    _step(a, ua, 0)
    metadata = a.restart_metadata()
    a.inputs["cai"].x.array[:] = calcium(2.0)
    a.begin_step(DT, DT)
    assert a.step_pending
    a.load_restart(a.restart_functions(), metadata)
    assert not a.step_pending
    with pytest.raises(RuntimeError):
        a.post_solve()


def test_backends_are_checkpointable(split_modules):
    _, mech = split_modules["caisplit"]
    g, _ = _generated(mech, "monolithic")
    c, _ = _crossbridge("Land2017")
    assert isinstance(g, Checkpointable)
    assert isinstance(c, Checkpointable)
