# 05: The Ca_i split runs through the same coupler

**What to build:** A crossbridge contraction model driven by calcium produced by a real EP solve — the thing that has not been possible until now — using the same orchestration code as the zeta split.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] The Ca_i-split backend runs in a coupled simulation, receiving calcium from EP
- [ ] The troponin buffering flux crosses back to the EP subsystem
- [ ] Gate test 1 passes: both backends run through one coupler, each on its own split
- [ ] Gate test 3 passes: the buffering flux measurably changes the calcium transient
- [ ] One test exercises the real ToR-ORd Ca_i split and is marked slow
- [ ] The backend-direct test suite still passes unchanged — backends remain usable without a coupler
