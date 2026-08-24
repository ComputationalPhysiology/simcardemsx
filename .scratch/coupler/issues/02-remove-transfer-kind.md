# 02: Remove the state-or-monitor field from the transfer record

**What to build:** Activation backends stop declaring a field that nothing reads. Prefactor: makes the transfer record mean only what it can deliver.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The field is gone from the transfer record and from both backends' declarations
- [x] The docstring rationale claiming states and monitors need different index lookups is gone with it
- [x] Assertions pinning the field are removed
- [x] The full suite still passes
