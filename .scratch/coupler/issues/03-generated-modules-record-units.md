# 03: Generated ODE modules record their units

**What to build:** Each generated ODE module carries a mapping from variable name to the unit the ODE source declares for it, so a later consumer can check a transfer's units without re-parsing the source.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Both generated modules expose a unit mapping derived from the parsed ODE
- [ ] The mapping covers the variables that cross between EP and mechanics for all three shipped splits
- [ ] Nothing consumes it yet and nothing breaks — this is purely additive
- [ ] A variable the source gives no unit for is representable without inventing one
