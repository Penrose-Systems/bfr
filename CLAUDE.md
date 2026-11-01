@docs/cpp_code_style.md
@docs/roadmap.md

# BFR

BFR unifies graphics APIs behind one API. That is the whole job.

- Remember that this is simply an abstraction so that our API can unify any number of backend APIs.
- Never allow a backend to bleed into the public API.
- Make the smallest change whilst being mindful of code architecture, structure, and maintainability for humans
- Always notify owner in a precise and concise manner, and wait for direction, when a decision is required that you're unable to determine.
- Report what the backend says. Do not invent, hallucinate, substitute, or make a fall back.
- Public API (api/bfr.yaml, bfr.h, bfr.hpp, include/bfr) changes: visually notify the exact signatures first.
- Every reason given to the owner cites a file:line read this session; a claim with no source is checked or cut.
- Before the owner sees a proposal, a fresh agent with no context checks its claims against the code and the reference engines; fix what it finds first.
- Never defer what the current roadmap step needs, unless a missing dependency can be cited.
- Say what changed in a precise and concise manner, then stop.
