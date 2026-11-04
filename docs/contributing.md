# Contributing

Project rules that are not C++ style. C++ rules are in
[`cpp_code_style.md`](cpp_code_style.md).

## Documentation
- Every decision, rule and fact is written in one place only, and everything else links to it. [review]

## Optimisation
- The optimiser resolves every `PERF` marker and sweeps all code written in the task, with before-and-after measurements. [loop]
- Flagged code outside the task gets an issue, which its `PERF` marker references. [loop]
- No unreferenced `PERF` marker reaches a PR. [script]

## Build
- CMake 3.28; options via `cmake_dependent_option`; no `file(GLOB_RECURSE)`. [build, script]

## Dependencies
- Prefer a mature open-source library to writing our own, when it meets these rules. [review]
- OSI-approved permissive licences only; never copyleft (GPL, LGPL, AGPL, MPL). [script]
- It must support every target platform. [CI]
- One source folder owns it behind an interface; its reason is in [`design.md`](design.md). Test and build tools are exempt. [script, review]
