"""The ``rodero_05`` case of physcardems' ``em_tref7`` run, loaded for simcardemsx.

physcardems' case directory (``cases/rodero_05``) is read as it is; nothing is copied
into simcardemsx. Every parameter value below is copied as a number from physcardems,
and not imported from it, with the file and lines it comes from:

- ``cases/rodero_05/run_rodero_em_tref7.py`` (the run script): the reference scaling,
  quadrature degree, tags, valve-plug stiffness and the unit conversions of the cycle;
- ``src/physcardems/parameters.py``: the Land parameters (``LAND_BASE``,
  ``LAND_OVERRIDES``, ``LAND_SCALES``) and ``EP_SCALES``;
- ``configs/elife/em_tref7.toml``: the ``[circulation.lv]`` and ``[circulation.rv]``
  tables, converted to SI with the run script's constants;
- ``cases/rodero_05/case.toml`` and ``src/physcardems/ecg.py``: the electrodes of the
  pseudo-ECG, which only ``post.py`` reads (:func:`electrodes`).

The geometry is shrunk by :data:`REFERENCE_SCALE` about its centroid before the
``pulse.HeartGeometry`` is built, physcardems' stand-in for unloading the imaged heart.

The initial states are physcardems' single-cell steady states, one JSON per ToR-ORd
cell type, in the one ``steady_state_pcl800_<hash>`` directory of the case. physcardems
names that directory by a hash of the parameters that change the paced EP states: its
``parameter_hash`` leaves out ``MECHANICS_ONLY_KEYS`` (``Tref``, ``Beta0``, ``Tot_A``),
which the EP half of the split never reads. So the same directory serves any
``tref_scale``: the one in the case was paced with ``Tref`` x 3 (its JSONs record 360
kPa), and physcardems' em_tref7 run, at ``Tref`` x 7, uses it too. :func:`load_case`
checks the parameters that do reach the EP states against the JSONs' own record instead
of recomputing the hash.

Serial only, as physcardems' run is: the centroid is the mean of this process's nodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import pulse
from pulse.cycle import CycleParams, PrescribedInflow, Windkessel

import cardiac_geometries.geometry

#: The converted geometry: cell tags, DG0 fibres, and the EP node fields.
GEOMETRY_DIR = "rodero_05_dolfinx_v2"

#: The steady states' directory, ``steady_state_pcl800_<parameter hash>``.
STEADY_STATE_GLOB = "steady_state_pcl800_*"

#: ToR-ORd's cell types, the values of the ``celltype`` node field, and the names of
#: their steady-state files (run script, lines 223-229).
CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}

#: Quadrature degree of the mechanics form (run script, line 52, ``QUAD_DEGREE``).
QUADRATURE_DEGREE = 4

#: Facet tags the example uses, as the run script asserts them (line 53, ``TAGS``).
TAGS = {"LV": 30, "RV": 20, "EPI": 40, "BASE": 10}

#: Cell tags of the LV and RV myocardium (line 55, ``MYOCARDIUM_TAGS``). The other tags,
#: 7-10, are the valve plugs.
MYOCARDIUM_TAGS = (1, 2)

#: Passive stiffness of the valve plugs relative to the myocardium (line 54).
VALVE_STIFFNESS_SCALE = 3.0

#: Isotropic scaling of the imaged geometry about its centroid (line 64,
#: ``REFERENCE_SCALE``).
REFERENCE_SCALE = 0.9

#: Pacing cycle length, in ms (line 47, ``PCL_MS``): the steady states' and the cycle's.
PCL_MS = 800.0

#: The Land parameters of the split's ``.ode`` names, before calibration: physcardems'
#: ``parameters.py`` ``LAND_BASE`` (Margara et al.).
LAND_BASE = {
    "Trpn50": 0.35,
    "ntrpn": 2.0,
    "ktrpn": 0.1,
    "rs": 0.25,
    "rw": 0.5,
    "Tot_A": 25.0,
    "ku": 0.021,
    "ntm": 2.036,
    "Beta0": 2.3,
    "Beta1": -2.4,
    "gammas": 0.0085,
    "gammaw": 0.615,
    "phi": 2.23,
    "cat50_ref": 0.805,  # uM
    "Tref": 120.0,  # kPa
    "kws": 0.012,  # 1/ms
    "kuw": 0.182,  # 1/ms
}

#: The eLife calibration on top of :data:`LAND_BASE`: ``parameters.py``
#: ``LAND_OVERRIDES``, and ``em_tref7.toml`` ``[land] overrides``.
LAND_OVERRIDES = {"cat50_ref": 0.534}

#: ``em_tref7.toml`` ``[land] scales`` (and the run script, line 74), less ``Tref``,
#: which :func:`land_parameters` takes as ``tref_scale``.
LAND_SCALES = {"kws": 3.86}

#: em_tref7's ``Tref`` scale (``[land] scales``; the run script, line 74).
TREF_SCALE = 7.0

#: physcardems' ``parameters.py`` ``MECHANICS_ONLY_KEYS``: the Land parameters that do
#: not change the paced EP states, and so are left out of the steady states' hash.
MECHANICS_ONLY_KEYS = ("Tref", "Beta0", "Tot_A")

#: Scales of other cell-model parameters: ``parameters.py`` ``EP_SCALES``, and
#: ``em_tref7.toml`` ``[cell] ep_scales`` (the eLife two-fold GCaL).
EP_SCALES = {"PCa_b": 2.0}

#: The electrode positions of the pseudo-ECG, one row per electrode, in the case
#: directory, and the factor taking them to the mesh's metres: the file is in cm
#: (physcardems ``cases/rodero_05/case.toml``, lines 16-17, ``[ecg]``).
ELECTRODES = "rodero_05_fine_nodefield_electrode_xyz.csv"
ELECTRODE_UNIT_SCALE = 1e-2

#: The electrode of each row of :data:`ELECTRODES`, in order (physcardems
#: ``src/physcardems/ecg.py``, line 16, ``ELECTRODE_ORDER``).
ELECTRODE_ORDER = ("LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6")

# The run script's unit conversions for the cycle (lines 83-85).
ML_PER_MS = 1e-6 / 1e-3  # m^3/s per mL/ms
MMHG_S_PER_ML = 133.322 / 1e-6  # Pa s/m^3 per mmHg s/mL
ML_PER_MMHG = 1e-6 / 133.322  # m^3/Pa per mL/mmHg


@dataclass
class Case:
    """Everything the example takes from the case directory.

    Attributes
    ----------
    geometry:
        The mechanics geometry, scaled by :data:`REFERENCE_SCALE`, in metres, its
        measures at :data:`QUADRATURE_DEGREE`.
    f0, s0, n0:
        Fibre, sheet and sheet-normal directions, DG0.
    myocardium_mask:
        DG0: 1 in the myocardium (cell tags :data:`MYOCARDIUM_TAGS`), 0 in the valve
        plugs.
    stiffness_scale:
        DG0: 1 in the myocardium, :data:`VALVE_STIFFNESS_SCALE` in the valve plugs.
    lat_ms:
        Local activation time per P1 dof, in ms: ToR-ORd's ``i_Stim_Start``.
    celltype:
        ToR-ORd cell type per P1 dof: 0 endo, 1 epi, 2 mid.
    initial_states:
        Single-cell steady state per cell type, every state of the whole ``.ode``
        (both halves of the split), by name.
    land:
        The Land parameters by ``.ode`` name (:func:`land_parameters`).
    ep_scales:
        Factors on other cell-model parameters, by name (:data:`EP_SCALES`).
    lv, rv:
        Each ventricle's five-phase cycle, in SI.
    """

    geometry: pulse.HeartGeometry
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function | None
    myocardium_mask: dolfinx.fem.Function
    stiffness_scale: dolfinx.fem.Function
    lat_ms: np.ndarray
    celltype: np.ndarray
    initial_states: dict[int, dict[str, float]]
    land: dict[str, float]
    ep_scales: dict[str, float]
    lv: CycleParams
    rv: CycleParams


def land_parameters(tref_scale: float = TREF_SCALE) -> dict[str, float]:
    """physcardems' ``land_values``: :data:`LAND_BASE`, with :data:`LAND_OVERRIDES`,
    then scaled by :data:`LAND_SCALES` and ``Tref`` by ``tref_scale``."""
    land = LAND_BASE | LAND_OVERRIDES
    for name, scale in (LAND_SCALES | {"Tref": tref_scale}).items():
        land[name] = land[name] * scale
    return land


def cycle_parameters() -> dict[str, CycleParams]:
    """em_tref7's ``[circulation.lv]`` and ``[circulation.rv]`` in SI.

    Converted with the run script's constants (lines 83-85); the same numbers as its
    ``lv_params`` and ``rv_params`` (lines 87-104). The beat is :data:`PCL_MS`.
    """
    period = PCL_MS / 1e3
    lv = CycleParams(
        t_zero=0.05,
        preload_pressure=500.0,
        t_end_diastole=0.12,
        p_end_diastole=1000.0,
        p_fill=500.0,
        period=period,
        windkessel=Windkessel(
            p_init=9000.0,
            resistance=1.1 * MMHG_S_PER_ML,
            compliance=1.5 * ML_PER_MMHG,
            characteristic_impedance=0.03 * MMHG_S_PER_ML,
        ),
        filling=PrescribedInflow(rate=0.046 * ML_PER_MS),
    )
    rv = CycleParams(
        t_zero=0.05,
        preload_pressure=170.0,
        t_end_diastole=0.12,
        p_end_diastole=330.0,
        p_fill=170.0,
        period=period,
        windkessel=Windkessel(
            p_init=3000.0,
            resistance=0.1 * MMHG_S_PER_ML,
            compliance=4.0 * ML_PER_MMHG,
            characteristic_impedance=0.01 * MMHG_S_PER_ML,
        ),
        filling=PrescribedInflow(rate=0.046 * ML_PER_MS),
    )
    return {"LV": lv, "RV": rv}


def _steady_state_dir(case_dir: Path) -> Path:
    """The case's one steady-state directory, or the glob pattern if it has none.

    Raises
    ------
    ValueError
        If more than one directory matches: which one was paced with this run's
        parameters is not for the loader to guess.
    """
    found = sorted(p for p in case_dir.glob(STEADY_STATE_GLOB) if p.is_dir())
    if len(found) > 1:
        raise ValueError(
            f"{case_dir} has {len(found)} steady-state directories "
            f"({', '.join(p.name for p in found)}); exactly one {STEADY_STATE_GLOB} is "
            "expected.",
        )
    return found[0] if found else case_dir / STEADY_STATE_GLOB


def _steady_state_file(state_dir: Path, celltype: int) -> Path:
    return state_dir / f"celltype{celltype}_{CELLTYPES[celltype]}.json"


def required_files(case_dir: Path) -> list[Path]:
    """Every file :func:`load_case` reads from ``case_dir``.

    The geometry's ``geometry.bp``, ``ep_node_fields.bp`` and ``markers.json``, and the
    steady state of each cell type in the one ``steady_state_pcl800_*`` directory. With
    no such directory, the steady states' paths hold the glob pattern.

    Raises
    ------
    ValueError
        If there is more than one ``steady_state_pcl800_*`` directory.
    """
    case_dir = Path(case_dir)
    geometry_dir = case_dir / GEOMETRY_DIR
    state_dir = _steady_state_dir(case_dir)
    return [
        geometry_dir / "geometry.bp",
        geometry_dir / "ep_node_fields.bp",
        geometry_dir / "markers.json",
        *(_steady_state_file(state_dir, c) for c in CELLTYPES),
    ]


def check_required_files(case_dir: Path) -> None:
    """Raise ``FileNotFoundError``, naming every missing one, unless every file of
    :func:`required_files` exists. Raises ``ValueError`` as :func:`required_files` does."""
    case_dir = Path(case_dir)
    missing = [path for path in required_files(case_dir) if not path.exists()]
    if missing:
        listed = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            f"The rodero_05 case in {case_dir} is missing {len(missing)} file(s):\n{listed}",
        )


def files_sha256(case_dir: Path) -> str:
    """The sha256 of every file :func:`load_case` reads from ``case_dir``, by its path
    relative to ``case_dir`` and its contents. A ``.bp`` path is a directory: each file
    in it counts. The files are about 10 MB, so hashing their contents is cheap, and,
    unlike their sizes and modification times, does not change when the case is copied.

    Raises ``FileNotFoundError`` as :func:`check_required_files` does.
    """
    case_dir = Path(case_dir)
    check_required_files(case_dir)
    digest = hashlib.sha256()
    for path in required_files(case_dir):
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            digest.update(str(file.relative_to(case_dir)).encode() + b"\0")
            digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def electrodes(case_dir: Path) -> dict[str, np.ndarray]:
    """The pseudo-ECG's electrodes, by name (:data:`ELECTRODE_ORDER`), in metres: the
    rows of :data:`ELECTRODES` times :data:`ELECTRODE_UNIT_SCALE` (physcardems'
    ``ecg.load_electrodes``).

    They are in the imaged heart's frame, as physcardems uses them: the
    :data:`REFERENCE_SCALE` shrink of the mesh does not move them.

    Raises
    ------
    FileNotFoundError
        If the case has no :data:`ELECTRODES`, naming the file.
    ValueError
        If it does not hold one 3-vector per electrode.
    """
    path = Path(case_dir) / ELECTRODES
    if not path.is_file():
        raise FileNotFoundError(f"The rodero_05 case has no electrode file: {path} does not exist")
    xyz = np.loadtxt(path, delimiter=",", ndmin=2) * ELECTRODE_UNIT_SCALE
    if xyz.shape != (len(ELECTRODE_ORDER), 3):
        raise ValueError(
            f"{path} holds {xyz.shape} values, not one 3-vector per electrode {ELECTRODE_ORDER}",
        )
    return dict(zip(ELECTRODE_ORDER, xyz))


def _cell_fields(
    geo: cardiac_geometries.geometry.Geometry,
) -> tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
    """The DG0 myocardium mask and passive-stiffness scale, from the cell tags (run
    script, lines 191-197)."""
    assert geo.cfun is not None  # checked by the caller
    DG0 = dolfinx.fem.functionspace(geo.mesh, ("DG", 0))
    mask = dolfinx.fem.Function(DG0, name="myocardium")
    stiffness_scale = dolfinx.fem.Function(DG0, name="stiffness_scale")
    dofs = DG0.dofmap.list[geo.cfun.indices, 0]
    is_myocardium = np.isin(geo.cfun.values, MYOCARDIUM_TAGS)
    mask.x.array[dofs] = is_myocardium.astype(float)
    stiffness_scale.x.array[dofs] = np.where(is_myocardium, 1.0, VALVE_STIFFNESS_SCALE)
    return mask, stiffness_scale


def _node_field(mesh: dolfinx.mesh.Mesh, path: Path, name: str) -> np.ndarray:
    """One of the EP node fields, per P1 dof (run script, lines 201-207)."""
    function = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name=name)
    io4dolfinx.read_function(filename=path, u=function, name=name)
    return function.x.array.copy()


def _initial_states(
    state_dir: Path,
    land: dict[str, float],
    ep_scales: dict[str, float],
) -> dict[int, dict[str, float]]:
    """The steady state of each cell type, by state name (run script, lines 223-229).

    Raises
    ------
    ValueError
        If a steady state was paced at another cycle length, or with other values of
        the parameters that change the paced EP states: every Land parameter but
        :data:`MECHANICS_ONLY_KEYS`, and :data:`EP_SCALES`. This is what physcardems'
        parameter hash compares (run script, lines 227-228).
    """
    states = {}
    for celltype in CELLTYPES:
        path = _steady_state_file(state_dir, celltype)
        data = json.loads(path.read_text())
        paced = {
            "pcl_ms": data["pcl_ms"],
            "land": {k: v for k, v in data["land"].items() if k not in MECHANICS_ONLY_KEYS},
            "ep_overrides": data["ep_overrides"],
            "ep_scales": data["ep_scales"],
        }
        expected = {
            "pcl_ms": PCL_MS,
            "land": {k: v for k, v in land.items() if k not in MECHANICS_ONLY_KEYS},
            "ep_overrides": {},
            "ep_scales": ep_scales,
        }
        if paced != expected:
            raise ValueError(
                f"The steady state in {path} was paced with {paced}, not with this run's "
                f"{expected}; re-pace it with physcardems' pace_steady_state.py.",
            )
        states[celltype] = {name: float(value) for name, value in data["states"].items()}
    return states


def load_case(
    case_dir: Path,
    *,
    tref_scale: float = TREF_SCALE,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> Case:
    """Load physcardems' ``rodero_05`` case from ``case_dir``, for ``Tref`` x ``tref_scale``.

    Raises
    ------
    FileNotFoundError
        If any file of :func:`required_files` is missing, naming every missing one.
    ValueError
        If the geometry's facet tags are not :data:`TAGS`, it has no cell tags or no
        fibre and sheet directions, or a steady state was paced with other parameters
        (see :func:`_initial_states`).
    """
    case_dir = Path(case_dir)
    check_required_files(case_dir)
    geometry_dir = case_dir / GEOMETRY_DIR
    state_dir = _steady_state_dir(case_dir)

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geometry_dir)
    for name, tag in TAGS.items():
        if int(geo.markers[name][0]) != tag:
            raise ValueError(f"{name}: expected facet tag {tag}, the case has {geo.markers[name]}")
    if geo.cfun is None:
        raise ValueError(f"{geometry_dir} has no cell tags, which mark the valve plugs")
    if geo.f0 is None or geo.s0 is None:
        raise ValueError(f"{geometry_dir} has no fibre and sheet directions")

    # The reference scaling (run script, lines 182-184), before the HeartGeometry: its
    # measures do not depend on the coordinates, but nothing may see the imaged ones.
    x = geo.mesh.geometry.x
    centroid = x.mean(axis=0)
    x[:] = centroid + REFERENCE_SCALE * (x - centroid)
    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": QUADRATURE_DEGREE},
    )

    myocardium_mask, stiffness_scale = _cell_fields(geo)
    node_fields = geometry_dir / "ep_node_fields.bp"
    land = land_parameters(tref_scale)
    ep_scales = dict(EP_SCALES)
    cycles = cycle_parameters()
    return Case(
        geometry=geometry,
        f0=geo.f0,
        s0=geo.s0,
        n0=geo.n0,
        myocardium_mask=myocardium_mask,
        stiffness_scale=stiffness_scale,
        lat_ms=_node_field(geo.mesh, node_fields, "lat_ms"),
        # Stored as floats; the run script rounds them (line 207).
        celltype=np.rint(_node_field(geo.mesh, node_fields, "celltype")).astype(int),
        initial_states=_initial_states(state_dir, land, ep_scales),
        land=land,
        ep_scales=ep_scales,
        lv=cycles["LV"],
        rv=cycles["RV"],
    )
