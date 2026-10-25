#!/usr/bin/env python3

import pathlib
import tempfile
import unittest

from gen_engine_diagrams import ENGINE_COLUMNS, architecture_mermaid, load_model, module_list, render_all

REPO = pathlib.Path(__file__).resolve().parent.parent

MINIMAL = """
[[client]]
id = "agent"
name = "LLM agents"

[[client]]
id = "editor"
name = "Editor"

[[module]]
id = "endpoint"
name = "Live endpoint"

[[module]]
id = "loop"
name = "Main loop"

[[module]]
id = "scene"
name = "Scene"
"""


def model_from(text: str):
    with tempfile.TemporaryDirectory() as temporary:
        path = pathlib.Path(temporary) / "engine.toml"
        path.write_text(text, encoding="utf-8")
        return load_model(path)


class LoadTests(unittest.TestCase):
    def test_modules_keep_their_written_order(self) -> None:
        model = model_from(MINIMAL)

        self.assertEqual([module["id"] for module in model["module"]], ["endpoint", "loop", "scene"])

    def test_clients_keep_their_written_order_outside_the_modules(self) -> None:
        model = model_from(MINIMAL)

        self.assertEqual([client["id"] for client in model["client"]], ["agent", "editor"])
        self.assertNotIn("agent", [module["id"] for module in model["module"]])

    def test_a_file_without_clients_has_none(self) -> None:
        text = MINIMAL.replace('[[client]]\nid = "agent"\nname = "LLM agents"\n', "").replace(
            '[[client]]\nid = "editor"\nname = "Editor"\n', ""
        )

        self.assertEqual(model_from(text)["client"], [])

    def test_a_file_without_modules_is_rejected(self) -> None:
        text = MINIMAL[: MINIMAL.index("[[module]]")]

        with self.assertRaisesRegex(ValueError, "no modules"):
            model_from(text)

    def test_a_module_id_used_twice_is_rejected(self) -> None:
        text = MINIMAL.replace('id = "loop"', 'id = "endpoint"')

        with self.assertRaisesRegex(ValueError, "endpoint.*more than once"):
            model_from(text)

    def test_a_client_id_used_by_a_module_is_rejected(self) -> None:
        text = MINIMAL.replace('id = "agent"', 'id = "scene"')

        with self.assertRaisesRegex(ValueError, "scene.*more than once"):
            model_from(text)

    def test_a_client_without_a_name_is_rejected(self) -> None:
        text = MINIMAL.replace('name = "LLM agents"\n', "")

        with self.assertRaisesRegex(ValueError, "every client needs an id and a name"):
            model_from(text)

    def test_a_module_without_a_name_is_rejected(self) -> None:
        text = MINIMAL.replace('name = "Main loop"\n', "")

        with self.assertRaisesRegex(ValueError, "every module needs an id and a name"):
            model_from(text)

    def test_a_client_detail_is_rejected(self) -> None:
        text = MINIMAL.replace('name = "Editor"\n', 'name = "Editor"\ndetail = "later"\n')

        with self.assertRaisesRegex(ValueError, "client 'editor'.*'detail'"):
            model_from(text)

    def test_a_module_detail_is_rejected(self) -> None:
        text = MINIMAL.replace('name = "Main loop"\n', 'name = "Main loop"\ndetail = "fixed"\n')

        with self.assertRaisesRegex(ValueError, "module 'loop'.*'detail'"):
            model_from(text)

    def test_a_module_uses_list_is_rejected(self) -> None:
        text = MINIMAL.replace('name = "Main loop"\n', 'name = "Main loop"\nuses = ["scene"]\n')

        with self.assertRaisesRegex(ValueError, "module 'loop'.*'uses'"):
            model_from(text)

    def test_a_band_is_rejected(self) -> None:
        text = MINIMAL + '\n[[band]]\nid = "world"\nname = "World"\n'

        with self.assertRaisesRegex(ValueError, "'band'"):
            model_from(text)

    def test_a_flow_is_rejected(self) -> None:
        text = MINIMAL + '\n[[flow]]\nfrom = "agent"\nto = "endpoint"\n'

        with self.assertRaisesRegex(ValueError, "'flow'"):
            model_from(text)


class ArchitectureTests(unittest.TestCase):
    def test_the_engine_is_one_grid_of_modules_in_written_order(self) -> None:
        text = architecture_mermaid(model_from(MINIMAL))

        self.assertTrue(text.startswith("block-beta\n"))
        self.assertIn(f"  block:engine:1\n    columns {ENGINE_COLUMNS}\n", text)
        self.assertIn('    engine_title["Engine"] endpoint["Live endpoint"] loop["Main loop"] scene["Scene"]\n', text)

    def test_clients_are_one_row_above_the_engine(self) -> None:
        text = architecture_mermaid(model_from(MINIMAL))

        self.assertIn("  block:clients:1\n    columns 3\n", text)
        self.assertIn(
            '    clients_title["Clients<br/>outside the engine, through the live endpoint only"]'
            ' agent["LLM agents"] editor["Editor"]\n',
            text,
        )
        self.assertLess(text.index("block:clients"), text.index("block:engine"))

    def test_without_clients_there_is_no_clients_row(self) -> None:
        model = model_from(MINIMAL)
        model["client"] = []

        self.assertNotIn("clients", architecture_mermaid(model))

    def test_modules_are_engine_and_clients_are_client(self) -> None:
        text = architecture_mermaid(model_from(MINIMAL))

        self.assertIn("  class endpoint engine\n", text)
        self.assertIn("  class loop engine\n", text)
        self.assertIn("  class scene engine\n", text)
        self.assertIn("  class agent client\n", text)
        self.assertIn("  class editor client\n", text)

    def test_the_output_says_it_is_generated(self) -> None:
        text = architecture_mermaid(model_from(MINIMAL))

        self.assertIn("Generated by tools/gen_engine_diagrams.py", text)


ARCHITECTURE = "# Map\n\n<!-- modules -->\nold\n<!-- /modules -->\n\nRules.\n"
LISTED = "- Live endpoint\n- Main loop\n- [Scene](modules/scene.md)\n"


def docs_with(architecture: str | None = ARCHITECTURE):
    temporary = tempfile.TemporaryDirectory()
    docs = pathlib.Path(temporary.name)
    (docs / "engine_architecture.toml").write_text(MINIMAL, encoding="utf-8")
    (docs / "modules").mkdir()
    (docs / "modules" / "scene.md").write_text("# Scene\n", encoding="utf-8")
    if architecture is not None:
        (docs / "architecture.md").write_text(architecture, encoding="utf-8")
    return temporary, docs


class MapTests(unittest.TestCase):
    def test_the_module_list_links_each_module_that_has_a_doc(self) -> None:
        temporary, docs = docs_with()
        with temporary:
            self.assertEqual(module_list(model_from(MINIMAL), docs), LISTED)

    def test_render_all_replaces_only_the_block_between_the_markers(self) -> None:
        temporary, docs = docs_with()
        with temporary:
            render_all(docs / "engine_architecture.toml", check=False)

            text = (docs / "architecture.md").read_text(encoding="utf-8")
        self.assertEqual(text, f"# Map\n\n<!-- modules -->\n{LISTED}<!-- /modules -->\n\nRules.\n")

    def test_check_reports_a_stale_module_list(self) -> None:
        temporary, docs = docs_with()
        with temporary:
            render_all(docs / "engine_architecture.toml", check=False)
            (docs / "architecture.md").write_text(ARCHITECTURE, encoding="utf-8")

            self.assertEqual(render_all(docs / "engine_architecture.toml", check=True), ["architecture.md"])

    def test_check_passes_when_the_module_list_is_current(self) -> None:
        temporary, docs = docs_with()
        with temporary:
            render_all(docs / "engine_architecture.toml", check=False)

            self.assertEqual(render_all(docs / "engine_architecture.toml", check=True), [])

    def test_a_map_without_markers_is_rejected(self) -> None:
        temporary, docs = docs_with("# Map\n")
        with temporary:
            with self.assertRaisesRegex(ValueError, "architecture.md.*<!-- modules -->"):
                render_all(docs / "engine_architecture.toml", check=True)

    def test_a_missing_map_is_rejected(self) -> None:
        temporary, docs = docs_with(None)
        with temporary:
            with self.assertRaisesRegex(ValueError, "architecture.md"):
                render_all(docs / "engine_architecture.toml", check=True)


class RepositoryTests(unittest.TestCase):
    def test_committed_diagrams_match_the_module_list(self) -> None:
        stale = render_all(REPO / "docs" / "engine_architecture.toml", check=True)

        self.assertEqual(stale, [], f"regenerate with tools/gen_engine_diagrams.py: {stale}")


if __name__ == "__main__":
    unittest.main()
