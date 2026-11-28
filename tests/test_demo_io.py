"""What the demos share (``examples/demo_io.py``): the end of a run without
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

EXAMPLES = Path(__file__).parent.parent / "examples"
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


# ---------------------------------------------------------------------------
# media: off-screen GIFs and their provenance
# ---------------------------------------------------------------------------


def _cube():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)


def test_animate_writes_a_small_gif(tmp_path):
    pytest.importorskip("pyvista")
    imageio = pytest.importorskip("imageio")
    mesh = _cube()
    space = dolfinx.fem.functionspace(mesh, ("P", 1))
    n = space.dofmap.index_map.size_local
    captions: list[int] = []
    out = demo_io.animate(
        tmp_path / "m.gif",
        [mesh],
        lambda i: [(None, np.full(n, float(i)))],
        3,
        titles=["t"],
        scalar="x",
        clim=(0.0, 2.0),
        caption=lambda i: captions.append(i) or f"t = {i}",
    )
    assert captions == [0, 1, 2]
    assert out is not None and out.stat().st_size < 2_000_000
    shape = imageio.v3.improps(out).shape
    assert shape[0] == 3 and shape[2] <= 640


def test_animate_refuses_wide_gifs(tmp_path):
    mesh = _cube()
    with pytest.raises(ValueError, match="640"):
        demo_io.animate(
            tmp_path / "m.gif",
            [mesh],
            lambda i: [],
            1,
            titles=["t"],
            scalar="x",
            clim=(0.0, 1.0),
            size=(800, 360),
        )


def test_media_skip_without_pyvista(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.warns(UserWarning, match="pyvista"):
        out = demo_io.animate(
            tmp_path / "m.gif",
            [_cube()],
            lambda i: [],
            1,
            titles=["t"],
            scalar="x",
            clim=(0.0, 1.0),
        )
    assert out is None and not (tmp_path / "m.gif").exists()


def test_write_media_provenance(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.resolved.toml").write_text('scheme = "stabilized"\n[mechanics]\ndt = 1.0\n')
    folder = tmp_path / "media"
    monkeypatch.chdir(tmp_path)  # outside the repository
    path = demo_io.write_media_provenance(folder, {"a": run}, ["python", "post.py", "--media"])
    data = json.loads(path.read_text())
    assert path == folder / "provenance.json"
    assert data["command"] == ["python", "post.py", "--media"]
    assert data["runs"]["a"]["scheme"] == "stabilized"
    assert data["runs"]["a"]["mechanics"]["dt"] == 1.0
    assert data["provenance"]["git_commit"]
    assert "versions" in data["provenance"]
    assert data["date"].startswith("20")


def test_write_media_provenance_falls_back_to_run_json(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text('{"scheme": "stabilized", "dt_mech_ms": 1.0}')
    monkeypatch.chdir(tmp_path)
    path = demo_io.write_media_provenance(tmp_path / "media", {"a": run}, ["x"])
    data = json.loads(path.read_text())
    assert data["runs"]["a"] == {"scheme": "stabilized", "dt_mech_ms": 1.0}


def test_write_media_provenance_records_each_runs_own_provenance(tmp_path, monkeypatch):
    """Beside a run's settings, the ``provenance`` and ``history`` of its ``run.json``: the
    commit that ran it, which may differ from the one that drew the media."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.resolved.toml").write_text('scheme = "stabilized"\n')
    ran = {"git_commit": "abc123", "git_dirty": False}
    (run / "run.json").write_text(
        json.dumps({"scheme": "stabilized", "provenance": ran, "history": [ran, ran]}),
    )
    monkeypatch.chdir(tmp_path)
    data = json.loads(demo_io.write_media_provenance(tmp_path / "m", {"a": run}, ["x"]).read_text())
    assert data["runs"]["a"]["scheme"] == "stabilized"
    assert data["runs"]["a"]["provenance"] == ran
    assert data["runs"]["a"]["history"] == [ran, ran]


def test_write_media_provenance_records_paths_relative_to_the_repository(tmp_path):
    """Paths inside the repository are recorded relative to its root, so the committed
    file does not name the machine it was made on; paths outside it are kept."""
    repo = Path(demo_io.__file__).resolve().parent.parent
    run = tmp_path / "run"
    run.mkdir()
    inside = repo / "examples" / "slab" / "output" / "splits" / "zeta"
    odefile = repo / "examples" / "odefiles" / "composed" / "torord_land_zeta.ode"
    (run / "config.resolved.toml").write_text(
        f'output_dir = "{inside}"\nelsewhere = "{tmp_path}"\n[odefile]\npath = "{odefile}"\n',
    )
    data = json.loads(
        demo_io.write_media_provenance(
            tmp_path / "m",
            {"a": run},
            ["compare_splits.py", "--root", str(inside.parent)],
        ).read_text(),
    )
    record = data["runs"]["a"]
    assert record["output_dir"] == "examples/slab/output/splits/zeta"
    assert record["odefile"]["path"] == "examples/odefiles/composed/torord_land_zeta.ode"
    assert record["elsewhere"] == str(tmp_path)
    assert data["command"] == ["compare_splits.py", "--root", "examples/slab/output/splits"]


def test_write_media_provenance_records_the_drawing_provenance_it_is_given(tmp_path):
    """A caller takes :func:`demo_io.drawing_provenance` before it writes any medium (which
    would itself make the tree dirty) and passes it in; it is recorded as given."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.resolved.toml").write_text('scheme = "stabilized"\n')
    drawn = {"git_commit": "abc123", "git_dirty": False}
    data = json.loads(
        demo_io.write_media_provenance(tmp_path / "m", {"a": run}, ["x"], drawn=drawn).read_text(),
    )
    assert data["provenance"] == drawn
    assert {"git_commit", "git_dirty", "versions"} <= set(demo_io.drawing_provenance())
