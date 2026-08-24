# 01: Write the five gate tests, red

**What to build:** The five tests that define "the coupling works", written before any implementation, each failing for a stated and verified reason.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Test 1: both activation backends drive one coupled simulation, each on its own split
- [ ] Test 2: with stretch clamped to one, the coupled path matches a standalone contraction-model run
- [ ] Test 3: the troponin buffering flux reaching EP measurably changes the calcium transient
- [ ] Test 4: the distortion states arriving at the EP side are the ones the backend computed, not zeros
- [ ] Test 5: peak active tension falls monotonically with shortening velocity
- [ ] Tests 3 and 4 fail because the return path delivers zeros, confirmed by reading the failure, not by assumption
- [ ] Tests 1, 2 and 5 fail because no coupler consumes the backend interface
- [ ] Each failure message recorded, so a later green result is known to mean something
