# BFR

BFR unifies graphics APIs behind one API. That is the whole job.

- Make the smallest change that does what was asked. Nothing else.
- Do not add tests, helpers, options, modules, minimums, promises or docs
  that were not asked for. Prefer deleting over adding.
- Report what the backend says. BFR never invents, substitutes or falls back.
- Public API (api/bfr.yaml, bfr.h, bfr.hpp, include/bfr) changes: show the
  owner the exact signatures first.
- Say what changed in a few plain lines, then stop.
