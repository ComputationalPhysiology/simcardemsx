# 10: Port the worked example and the architecture docs

**What to build:** The one end-to-end example in the repository runs against the current interface, and the architecture documentation describes the coupling as it now is.

**Blocked by:** 05, 07

**Status:** resolved

- [x] The zeta-split experiment runs on the current interface
- [x] It reads its transfer variables from the backend rather than reaching into transfer buffers positionally
- [x] Its results change is stated plainly in the commit message: it has been running without distortion feedback
- [x] The architecture documentation matches the code, including that generated modules now record units
