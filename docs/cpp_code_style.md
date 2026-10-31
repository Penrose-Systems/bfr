# C++ code style

Rules tools can't check. `.clang-format` and `.clang-tidy` are authoritative and not restated.
Tags name the enforcement: [tool] clang-format or clang-tidy, [build] CMake or the compiler, [CI] CI builds, [tests], [script], [review]

## Language
- C++23
- Use `std::span`, `std::expected`, concepts, designated initialisers; ranges only to replace a loop. No modules, no coroutines. [review]
- Strings through `std::string`, `std::string_view` and their functions; no C string functions such as `strlen`, `strcmp` or `atoi`. [review]
- Aim to use std:: functions wherever possible [review]
- 
## Values (important)
- No hard-coded numbers in code; a named `const` or `constexpr` is allowed. Markup values and field defaults count as named. [tool, review]
- Prefer every tunable value configurable from an external file, so it can be changed without recompiling. [review]

## Files
- snake_case `.h` and `.cpp`; one primary abstraction per file. [script, review]
- Warn above 500 lines, fail above 600; over 500 needs a split proposal or a reason in the PR. [script]
- Public headers in `modules/<module>/public/`, included by file name; sources and private headers in `modules/<module>/private/`. Public header names are unique across modules. [review]

## Ownership
- No raw `new` or `delete`; `std::unique_ptr` for single ownership. [tool]
- Handles for shared references; `std::shared_ptr` only with a reason at the declaration. [review]
- `friend` only where `VTYPE` generates it for the Reflection table; any other `friend` needs Justin's approval and a reason at the declaration. [script, review]
- Pass views or `const&`; accessors return borrowed references. [tool, review]
- Ownership points down only: an object never holds a pointer or reference to its owner or creator. What it needs from above is given to it at creation, by value. [review]

## Duplication
- One owner for each piece of logic, constant and failure code; extend the owner, never copy it. A routine parameterised by its caller is still a copy. [review, script]
- Documentation too: every decision, rule and fact is written in one place only, and everything else links to it. [review]
- A second implementation of existing functionality needs Justin's approval and a stated reason. [review]

## Hot path
Anything per frame, view, pass, draw or instance, plus any job the frame waits on.
- No heap allocation after warmup; scratch comes from the frame arena. [audit]
- No `std::function`, string keys or allocating formatting. [audit, review]
- Virtual calls per pass are fine, per draw or instance are not. [review]
- Elsewhere, allocate from the lifetime class's home; the general heap only in documented cold code. [review]

## Data-oriented
The measured fastest correct layout beats textbook idiom.
- Resolve keys to indices at cold boundaries; hot code never repeats a map lookup. [review]
- Removable entries use slot-plus-generation handles, never bare indices. [review, tests]
- Contiguous arrays, not node-based containers, in hot code. [review]
- A non-obvious layout names its measurement in a comment. [review]

## Optimisation
- Standard C++ may come first in a task, marked `// PERF: …`. [loop]
- The optimiser resolves every marker and sweeps all code written in the task, with before-and-after measurements. [loop]
- Flagged code outside the task gets an issue and `// PERF(#issue): …`. [loop]
- No unreferenced `PERF` marker reaches a PR. [script]

## Interfaces
- Abstract classes: no prefix or suffix, virtual destructor, protected constructor, no data, static factory returning `std::unique_ptr`. The one exception: the RHI's `Rhi` holds the `RhiConfig` every backend runs with. [review]

## Errors and text
- User-facing text is `LocalizedText` from a message ID; diagnostics stay English with stable codes. [build]

## Comments
- The code is the comment: names and structure explain it. [review]
- Prefer one overview comment for a block of code to comments plastered through it; add more only where they truly make the code understandable. [review]
- Why, not what; public declarations document ownership, lifetime, preconditions and failure. [review]
- An annotation description is published data, not a comment: it says what, for users. Never repeat it in a comment. [review]
- `TODO` and `FIXME` reference an issue. [script]

## Tests
- Catch2; each case has one module tag and one kind tag (`[unit]` or `[gpu]`), partitioning the suite. [script]
- Test behaviour, never implementation: pure logic as data in and out, GPU work on real drivers. [review]
- Test doubles only for non-GPU dependencies, and always Trompeloeil mocks; no hand-written fakes or spies. [review]

## Build
- CMake 3.28; `VESPER_` options via `cmake_dependent_option`; no `file(GLOB_RECURSE)`. [build, script]

## Dependencies
- Prefer a mature open-source library to writing our own, when it meets these rules. [review]
- OSI-approved permissive licences only; never copyleft (GPL, LGPL, AGPL, MPL). [script]
- It must support every target platform and keep our hot-path rules. [CI, audit]
- One module owns it behind a Vesper interface; its reason is in that module's doc. Test and build tools are exempt. [script, review]

## Patterns
- A recurring problem uses its listed pattern. A new pattern needs Justin's approval and is added here first. [review]
- Failure: `std::expected` with an error struct holding a code from one engine-wide list and its associated data, such as a description.
- Configuration: each module owns its typed configuration.
- Published types: field markup macros, read by the annotation tool (`modules/reflection.md`).
