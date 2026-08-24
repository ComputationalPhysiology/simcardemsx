# # Declaring what crosses: splits, units, and how they fail
#
# A coupled electro-mechanics simulation cuts one cellular ODE system in two. Where
# you cut it decides which variables have to travel between the electrophysiology
# subsystem and the activation backend, and in which direction.
#
# The awkward part is that the transfer buffers are **positional** -- row 0, row 1 --
# while backends refer to variables by **name**. Something has to map between them,
# and if that mapping is wrong nothing crashes. Calcium gets written into a
# crossbridge population, the solve converges, and you get numbers.
#
# This demo shows what the coupler checks before it will run, and what each failure
# looks like. It needs no mesh and runs in seconds.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

from simcardemsx.backends import CrossbridgeSegregated, Transfer, ZetaSplitUFL
from simcardemsx.ode_model import load_ode_modules
from simcardemsx.transfers import (
    AssumedUnitWarning,
    TransferMismatch,
    UnitPolicy,
    check,
)

ODEFILES = Path("../numerical_experiments/odefiles")

# ## Two descriptions of the same split
#
# `gotranx` derives what must cross from where the `.ode` file was cut, and writes
# it into each generated module as a name-to-index mapping. Each side's `missing`
# names what *that* side needs from the other.


def maps_for(name):
    modules = load_ode_modules(ODEFILES / name, Path("_generated") / name[:-4])
    return {
        # gotranx omits `missing` entirely -- rather than emitting an empty one --
        # when a side needs nothing back, which is the CaTrpn split.
        "ep_missing": getattr(modules.ep, "missing", {}),
        "mech_missing": getattr(modules.mechanics, "missing", {}),
        "units": modules.ep.units,
    }


for name in sorted(p.name for p in ODEFILES.glob("*.ode")):
    m = maps_for(name)
    print(f"{name:38s} EP needs {list(m['ep_missing']) or '-- nothing --'}")
    print(f"{'':38s} mechanics needs {list(m['mech_missing'])}")

# The backend independently declares the same thing, because it is the one that
# owns the Functions those values land in:

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

zeta = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh)
cai = CrossbridgeSegregated(f0=f0, mesh=mesh)

for backend in (zeta, cai):
    wants = [t.name for t in backend.wants_from_ep()]
    gives = [t.name for t in backend.gives_to_ep()]
    print(f"{type(backend).__name__:22s} wants {wants} from EP, gives back {gives}")

# ## Reconciling the two
#
# `check` compares them. Matching declarations resolve quietly.

check(zeta, **maps_for("ToRORd_dynCl_endo_zetasplit.ode"))
check(cai, **maps_for("ToRORd_dynCl_endo_caisplit.ode"))
print("both backends agree with their own split")

# ## Failure 1: the wrong split
#
# This is the error the check mainly exists to catch. Without it, `cai` is written
# into `XS` -- positionally valid, physically nonsense, and completely silent.

try:
    check(zeta, **maps_for("ToRORd_dynCl_endo_caisplit.ode"))
except TransferMismatch as e:
    print(e)

# No policy switches this off. Units are a preference; pairing a backend with the
# wrong split is a correctness bug.

# ## Failure 2: the wrong unit
#
# Three libraries meet here with three conventions -- the EP model works in
# millimolar, crossbridge in micromolar, mechanics in kPa. A mistake is a factor of
# a thousand, and produces a calcium transient that looks entirely reasonable.
#
# Units are compared with `pint`, on dimension and scale rather than spelling, so
# `mmol/L` and `mM` agree while `µm` and `µM` do not.

wrong = maps_for("ToRORd_dynCl_endo_caisplit.ode")
wrong["units"] = {**wrong["units"], "cai": "uM"}  # the file says mM

try:
    check(cai, **wrong)
except TransferMismatch as e:
    print(e)

# ## When the source says nothing
#
# Not every variable *can* declare a unit. `J_TRPN` is an intermediate -- a derived
# expression -- and the ODE syntax has nowhere to attach one. The coupler assumes
# what the backend expects, which is the only assumption under which the coupling is
# correct, and says so rather than proceeding silently.

import warnings  # noqa: E402

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    check(cai, **maps_for("ToRORd_dynCl_endo_caisplit.ode"))

for w in caught:
    if issubclass(w.category, AssumedUnitWarning):
        print(str(w.message).split(". Annotate")[0])

# ## Opting out
#
# A user who does not annotate units, and does not want to hear about it, can turn
# the whole thing off -- without importing anything to say so.

check(cai, **wrong, policy=UnitPolicy.off)
print("units='off': no checking, no warnings")

# `UnitPolicy.warn` sits in between: it reports the disagreement and carries on,
# which is what you want while fixing an ODE file's annotations.

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    check(cai, **wrong, policy=UnitPolicy.warn)
print(f"units='warn': ran anyway, {len(caught)} warning(s)")

# ## What a Transfer carries
#
# A bare name is not enough to drive a transfer: it says nothing about direction or
# units. Direction comes from which method returned it; the unit rides along.

for t in cai.wants_from_ep() + cai.gives_to_ep():
    print(f"  {t.name:8s} {t.unit}")

assert Transfer("cai", unit="mM").unit == "mM"
