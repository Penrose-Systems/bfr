#!/usr/bin/env python3

import pathlib
import subprocess
import tempfile
import unittest

from check_format import sources, violations


class SourceTests(unittest.TestCase):
    def test_new_sources_are_listed_and_ignored_or_deleted_ones_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            def git(*args: str) -> None:
                subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

            git("init", "-q")
            (root / ".gitignore").write_text("build/\n", encoding="utf-8")
            for name in ("tracked.cpp", "deleted.cpp", "new.h", "build/generated.cpp", "notes.txt"):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("", encoding="utf-8")
            git("add", "tracked.cpp", "deleted.cpp")
            (root / "deleted.cpp").unlink()

            self.assertEqual(sorted(sources(root)), [root / "new.h", root / "tracked.cpp"])


class ViolationTests(unittest.TestCase):
    def test_only_the_misformatted_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            good = pathlib.Path(temporary) / "good.cpp"
            bad = pathlib.Path(temporary) / "bad.cpp"
            good.write_text("int main() { return 0; }\n", encoding="utf-8")
            bad.write_text("int main( ){return 0 ;}\n", encoding="utf-8")

            self.assertEqual(violations([good, bad]), [bad])

    def test_no_files_means_no_violations(self) -> None:
        self.assertEqual(violations([]), [])


if __name__ == "__main__":
    unittest.main()
