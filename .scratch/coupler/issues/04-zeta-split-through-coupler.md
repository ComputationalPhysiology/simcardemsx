# 04: The zeta split runs through a backend-driven coupler

**What to build:** A coupled simulation where the orchestration layer asks the activation backend what needs to cross, moves exactly that, and delivers the backend's computed distortion states back to the EP subsystem. This is the tracer bullet: the first complete path through every layer.

Fixes the mechano-electric feedback bug by construction — the backward transfer's source becomes the backend's own output Functions rather than a buffer nothing ever wrote.

**Blocked by:** 01, 02

**Status:** resolved

- [x] The controller takes its activation backend explicitly and refuses a backend that is not the one in the mechanics form
- [x] Transfers are resolved by name against the split the ODE file describes, in one place
- [x] The forward transfer lands directly in the Functions the backend owns
- [x] The backward transfer reads directly from the Functions the backend exposes
- [x] The zeta-split backend no longer accepts injected Functions; it owns its own
- [x] Gate test 4 passes: the EP side receives the distortion states the backend computed
- [x] Verified against a synthetic ODE file built in the test
- [x] Buffers made dead by this change are left in place for now, to keep this one slice
