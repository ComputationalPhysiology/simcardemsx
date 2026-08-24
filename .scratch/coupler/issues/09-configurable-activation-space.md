# 09: The activation function space is configurable

**What to build:** A researcher can choose the function space the activation state lives on, trading interpolation error against cost, without editing the package.

**Blocked by:** 05

**Status:** resolved

- [x] Both activation backends accept a function space specification
- [x] The default is what the package used before this change
- [x] Results are unchanged at the default, demonstrably
- [x] The two backends agree on how the space is specified
