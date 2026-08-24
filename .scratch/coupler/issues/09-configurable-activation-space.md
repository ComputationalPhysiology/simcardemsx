# 09: The activation function space is configurable

**What to build:** A researcher can choose the function space the activation state lives on, trading interpolation error against cost, without editing the package.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Both activation backends accept a function space specification
- [ ] The default is what the package used before this change
- [ ] Results are unchanged at the default, demonstrably
- [ ] The two backends agree on how the space is specified
