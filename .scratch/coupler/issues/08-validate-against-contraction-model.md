# 08: Validate the coupled path against the contraction model

**What to build:** Evidence that the coupled path reproduces the contraction model it wraps — separating a coupling bug from a model bug, and pinning the sign of the sarcomere length-change rate.

**Blocked by:** 05

**Status:** resolved

- [x] Gate test 2 passes: with stretch clamped to one, the coupled path matches a standalone run driven by the same calcium, to tight tolerance
- [x] Gate test 5 passes: peak active tension falls monotonically with shortening velocity
- [x] The tolerance in the clamp test is justified, not tuned until green
- [x] The velocity test would fail if the sign of the length-change rate were flipped
