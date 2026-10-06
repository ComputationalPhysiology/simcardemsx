"""What the demos share (``numerical_experiments/demo_io.py``): the end of a run without
a ``Recorder``, and the P1 fields ``post.py`` writes from discontinuous sources.

The slab and BiV examples' own tests cover the rest through the examples.
"""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

EXAMPLES = Path(__file__).parent.parent / "numerical_experiments"
sys.path.insert(0, str(EXAMPLES))
try:
    demo_io = importlib.import_module("demo_io")
finally:
    sys.path.remove(str(EXAMPLES))


# ---------------------------------------------------------------------------
# finish, with write_run in place of a Recorder
# ---------------------------------------------------------------------------


def _checkpointer(order: list) -> SimpleNamespace:
    """What :func:`demo_io.finish` reads of a checkpointer: a controller after a step it
    accepted, with nothing pending, and the history."""
    controller = SimpleNamespace(t=4.0, t_failed=None, backend=SimpleNamespace(step_pending=False))
    return SimpleNamespace(
        controller=controller,
        history=[{"run": 1}],
        write=lambda: order.append("checkpoint"),
    )


def test_finish_without_a_recorder_writes_run_last(tmp_path):
    """The end checkpoint, the demo's files in order, then ``write_run`` with the extras
    a ``Recorder`` would get: the provenance, the checkpoint history and the restart flag."""
    order: list = []
    written: list[dict] = []

    def write_run(extra):
        order.append("run")
        written.append(dict(extra))

    demo_io.finish(
        None,
        _checkpointer(order),
        [("a", lambda: order.append("a")), ("b", lambda: order.append("b"))],
        failure=None,
        t_fail_ms=None,
        timings={},
        here=tmp_path,
        restart=True,
        write_run=write_run,
    )
    assert order == ["checkpoint", "a", "b", "run"]
    (extra,) = written
    assert set(extra) == {"provenance", "history", "restart"}
    assert extra["history"] == [{"run": 1}]
    assert extra["restart"] is True
    assert {"git_commit", "versions", "utc"} <= set(extra["provenance"])
    json.dumps(extra)  # what run.json is written from


def test_finish_without_a_recorder_still_writes_run_after_a_failed_file(tmp_path):
    """A demo file that fails does not stop the others or ``write_run``; its error is
    raised afterwards, since the loop did not fail."""
    order: list = []

    def broken():
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        demo_io.finish(
            None,
            _checkpointer(order),
            [("a", broken), ("b", lambda: order.append("b"))],
            failure=None,
            t_fail_ms=None,
            timings={},
            here=tmp_path,
            restart=False,
            write_run=lambda extra: order.append("run"),
        )
    assert order == ["checkpoint", "b", "run"]


@pytest.mark.parametrize("given", ["both", "neither"])
def test_finish_takes_a_recorder_or_write_run(tmp_path, given):
    recorder = SimpleNamespace(finish=lambda **kwargs: None) if given == "both" else None
    write_run = (lambda extra: None) if given == "both" else None
    with pytest.raises(TypeError, match="exactly one"):
        demo_io.finish(
            recorder,
            _checkpointer([]),
            [],
            failure=None,
            t_fail_ms=None,
            timings={},
            here=tmp_path,
            restart=False,
            write_run=write_run,
        )


# ---------------------------------------------------------------------------
# write_p1_fields: discontinuous sources are averaged, not interpolated
# ---------------------------------------------------------------------------


def _lumped_average(source: dolfinx.fem.Function) -> np.ndarray:
    """Each P1 dof's value: the volume-weighted mean of ``source``'s DG0 value over the
    cells around it, computed cell by cell, independently of ``make_averager``."""
    mesh = source.function_space.mesh
    shape = tuple(source.function_space.value_shape)
    bs = int(np.prod(shape)) if shape else 1
    P1 = dolfinx.fem.functionspace(mesh, ("P", 1))
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    volumes = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(ufl.TestFunction(DG0) * ufl.dx(domain=mesh)),
    ).array
    num_dofs = P1.dofmap.index_map.size_local
    total = np.zeros((num_dofs, bs))
    weight = np.zeros(num_dofs)
    values = source.x.array.reshape(-1, bs)
    for cell in range(mesh.topology.index_map(3).size_local):
        dofs = P1.dofmap.cell_dofs(cell)
        total[dofs] += volumes[cell] * values[DG0.dofmap.cell_dofs(cell)[0]]
        weight[dofs] += volumes[cell]
    return (total / weight[:, None]).reshape(-1)


@pytest.mark.parametrize("shape", [(), (3,)], ids=["scalar", "vector"])
def test_write_p1_fields_averages_a_dg_source(tmp_path, shape):
    """A DG0 field (as rodero's mask, moduli and fibres are) is shown on P1 as the
    volume-weighted average of the cells around each node, as the quadrature fields
    are, not interpolated from whichever cell is visited last."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)
    space = dolfinx.fem.functionspace(mesh, ("DG", 0, shape) if shape else ("DG", 0))
    source = dolfinx.fem.Function(space, name="field")
    values = np.arange(source.x.array.size, dtype=float) ** 2
    shown = {}

    def capture(t, p1):
        shown[t] = p1["shown"].x.array.copy()

    demo_io.write_p1_fields(
        tmp_path / "fields.bp",
        {"field": source},
        {"field": {0.0: values}},
        {"shown": "field"},
        capture,
    )
    source.x.array[:] = values
    np.testing.assert_allclose(shown[0.0], _lumped_average(source), rtol=1e-12)

    interpolated = dolfinx.fem.Function(
        dolfinx.fem.functionspace(mesh, ("P", 1, shape) if shape else ("P", 1)),
    )
    interpolated.interpolate(source)
    assert not np.allclose(shown[0.0], interpolated.x.array)
    assert (tmp_path / "fields.bp").is_dir()
