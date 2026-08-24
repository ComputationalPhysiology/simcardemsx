# 06: A mismatched split or unit is refused

**What to build:** Pairing an activation backend with an ODE file that describes a different split fails loudly at construction, instead of transferring calcium into a crossbridge population and producing plausible wrong numbers. The same for a transfer whose declared unit disagrees with the ODE source.

**Blocked by:** 03, 04

**Status:** resolved

- [x] A backend paired with an ODE file describing a different split raises, naming both sides of the disagreement
- [x] A declared unit disagreeing with the generated unit mapping raises
- [x] Matching declarations resolve without complaint for all three shipped splits
- [x] These checks are testable without constructing a mesh or a mechanics problem
- [x] The error says what to change, not merely that something is wrong
