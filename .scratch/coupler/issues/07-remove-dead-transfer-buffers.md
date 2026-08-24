# 07: Remove the dead transfer buffers

**What to build:** The contract half of ticket 04. The mechanics-side transfer buffers and the previous-values structure are no longer on any path; remove them and the per-step work that maintained them.

**Blocked by:** 04, 05

**Status:** resolved

- [x] The mechanics-side Functions and value arrays made dead by direct interpolation are gone
- [x] The previous-values structure, written every step and read by nothing, is gone
- [x] The non-matching-mesh interpolation machinery is untouched and its tests still pass
- [x] No buffer remains that is read but never written, or written but never read
- [x] The full suite passes
