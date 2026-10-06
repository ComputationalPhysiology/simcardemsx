"""Loading the ``rodero_05`` case of the worked example (``numerical_experiments/rodero_05``).

``case.py`` reads physcardems' ``cases/rodero_05`` as it is, with no data copied into
this repository, and shrinks the imaged geometry by 0.9 about its centroid, as
physcardems does to approximate the unloaded heart.

- **Review Focus 5**: a case directory missing any of the files the example reads
  raises ``FileNotFoundError`` naming every one of them, before io4dolfinx is reached.
- The loaded case matches physcardems' numbers (cells, myocardium cells, LAT range),
  and has the two properties the example's mechanics rests on:

  - every cell's fibre, sheet and normal vectors are unit vectors, since the backend's
    stretch and Holzapfel-Ogden's invariants scale with their lengths (the lesson of
    ruling R25, where interpolated fibres were 0.32 to 0.95 long);
  - the LV and RV cavity surfaces are closed: pulse's cavity multiplier is the wall
    pressure only then. A volume computed by the divergence theorem is independent of
    the origin exactly when the surface is closed, so shifting the mesh must leave it
    unchanged. The same holds for the 0.9 scaling, which is about the centroid, not
    the origin.

``case.py`` is imported by path, from ``numerical_experiments/``: the examples are not
part of the installed package.
"""

import importlib
import sys
from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "numerical_experiments"
PHYSCARDEMS_CASE = ROOT / "third-party" / "physcardems" / "cases" / "rodero_05"

#: physcardems' reference scaling (``run_rodero_em_tref7.py``, ``REFERENCE_SCALE``).
REFERENCE_SCALE = 0.9

#: The number of cells tagged LV (1) and RV (2) myocardium, and of all cells; 7-10 are
#: the valve plugs. The tags are named in ``rodero_05_dolfinx_v2/info.json``; the counts
#: are read off the mesh's own cell tags (``cfun``), not from that file.
LV_MYOCARDIUM_CELLS = 13775
RV_MYOCARDIUM_CELLS = 9252
NUM_CELLS = 24811

#: A shift of the whole mesh, in metres: of the order of the heart's own size.
SHIFT = np.array([0.01, -0.02, 0.03])


@pytest.fixture(scope="module")
def case_module():
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("rodero_05.case")
    finally:
        sys.path.remove(str(EXAMPLES))


def test_load_case_reports_missing_files(case_module, tmp_path):
    (tmp_path / "rodero_05_dolfinx_v2").mkdir()
    required = case_module.required_files(tmp_path)
    # geometry.bp, ep_node_fields.bp and markers.json, and one steady-state JSON per
    # ToR-ORd cell type.
    assert len(required) == 6

    with pytest.raises(FileNotFoundError) as raised:
        case_module.load_case(tmp_path)

    message = str(raised.value)
    for path in required:
        assert str(path) in message


@pytest.mark.skipif(
    not PHYSCARDEMS_CASE.exists(),
    reason="needs physcardems' rodero_05 case in third-party/physcardems",
)
def test_load_case_matches_physcardems_geometry(case_module):
    import pulse

    import cardiac_geometries.geometry

    case = case_module.load_case(PHYSCARDEMS_CASE)
    geometry = case.geometry
    mesh = geometry.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    assert num_cells == NUM_CELLS

    # The imaged geometry, read independently of load_case.
    imaged = pulse.HeartGeometry.from_cardiac_geometries(
        cardiac_geometries.geometry.Geometry.from_folder(
            comm=MPI.COMM_WORLD,
            folder=PHYSCARDEMS_CASE / "rodero_05_dolfinx_v2",
        ),
    )
    volume = {c: geometry.volume(c) for c in ("LV", "RV")}
    for c in ("LV", "RV"):
        assert volume[c] == pytest.approx(REFERENCE_SCALE**3 * imaged.volume(c), rel=1e-10)

    x = mesh.geometry.x
    x[:] += SHIFT
    try:
        shifted = {c: geometry.volume(c) for c in ("LV", "RV")}
    finally:
        x[:] -= SHIFT
    for c in ("LV", "RV"):
        assert shifted[c] == pytest.approx(volume[c], rel=1e-10)

    for name in ("f0", "s0", "n0"):
        field = getattr(case, name)
        if field is None:
            continue
        # One vector per cell: DG0.
        assert field.function_space.element.basix_element.degree == 0
        vectors = field.x.array.reshape(-1, 3)
        assert vectors.shape[0] == num_cells
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-10)

    mask = case.myocardium_mask.x.array
    assert mask.sum() == LV_MYOCARDIUM_CELLS + RV_MYOCARDIUM_CELLS
    assert set(np.unique(mask)) == {0.0, 1.0}
    stiffness = case.stiffness_scale.x.array
    np.testing.assert_array_equal(stiffness, np.where(mask == 1.0, 1.0, 3.0))

    assert case.lat_ms.min() == pytest.approx(130.0)
    assert case.lat_ms.max() == pytest.approx(181.66, abs=0.01)
    assert set(np.unique(case.celltype)) == {0, 1, 2}
    assert sorted(case.initial_states) == [0, 1, 2]

    # Tref x 7 of physcardems' 120 kPa; kws x 3.86; the eLife cat50_ref.
    assert case.land["Tref"] == pytest.approx(840.0)
    assert case.land["kws"] == pytest.approx(0.012 * 3.86)
    assert case.land["cat50_ref"] == 0.534
    assert case.ep_scales == {"PCa_b": 2.0}


def test_electrodes_report_a_missing_file(case_module, tmp_path):
    """Only ``post.py`` reads the electrodes, so ``required_files`` does not list them;
    a case without them is refused when they are read, naming the file."""
    assert all(path.name != case_module.ELECTRODES for path in case_module.required_files(tmp_path))
    with pytest.raises(FileNotFoundError) as raised:
        case_module.electrodes(tmp_path)
    assert str(tmp_path / case_module.ELECTRODES) in str(raised.value)


@pytest.mark.skipif(
    not PHYSCARDEMS_CASE.exists(),
    reason="needs physcardems' rodero_05 case in third-party/physcardems",
)
def test_electrodes_are_read_in_metres(case_module):
    """physcardems' electrode file is in cm, one row per electrode in the order of its
    ``ecg.py``: LA, RA, LL, RL, V1..V6. ``electrodes`` gives them in metres, by name."""
    electrodes = case_module.electrodes(PHYSCARDEMS_CASE)
    assert list(electrodes) == ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]
    for position in electrodes.values():
        assert position.shape == (3,)

    rows = np.loadtxt(PHYSCARDEMS_CASE / case_module.ELECTRODES, delimiter=",")
    np.testing.assert_array_equal(electrodes["RA"], 0.01 * rows[1])
