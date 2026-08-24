# 07: Remove the dead transfer buffers

**What to build:** The contract half of ticket 04. The mechanics-side transfer buffers and the previous-values structure are no longer on any path; remove them and the per-step work that maintained them.

**Blocked by:** 04, 05

**Status:** ready-for-agent

- [ ] The mechanics-side Functions and value arrays made dead by direct interpolation are gone
- [ ] The previous-values structure, written every step and read by nothing, is gone
- [ ] The non-matching-mesh interpolation machinery is untouched and its tests still pass
- [ ] No buffer remains that is read but never written, or written but never read
- [ ] The full suite passes
