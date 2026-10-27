#!/usr/bin/env python3
"""Fail if any .h, .hpp or .cpp file git tracks or would track differs from .clang-format.

    python3 tools/check_format.py            # list the offending files
    clang-format -i <file>                   # fix one
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CLANG_FORMAT = os.environ.get("CLANG_FORMAT") or shutil.which("clang-format-19") or "clang-format"


def sources(root: pathlib.Path = REPO_ROOT) -> list[pathlib.Path]:
    """The .h, .hpp and .cpp files git tracks or would track: new files count before
    their first commit, ignored ones never, and deleted ones are skipped."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.h", "*.hpp", "*.cpp"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [root / name for name in listed.split("\0") if name and (root / name).exists()]


def violations(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """The files clang-format would change."""
    failed = []
    for path in files:
        result = subprocess.run([CLANG_FORMAT, "--dry-run", "-Werror", str(path)], capture_output=True)
        if result.returncode != 0:
            failed.append(path)
    return failed


def main() -> int:
    failed = violations(sources())
    for path in failed:
        print(f"check_format: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
    if failed:
        print("Run: clang-format -i <file>", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
