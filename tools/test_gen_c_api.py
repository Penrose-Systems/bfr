#!/usr/bin/env python3

import copy
import unittest

from gen_c_api import HEADER, SOURCE, Api, SpecError, load, render

MINIMAL = {
    "version": {"major": 1, "minor": 2, "patch": 3},
    "enums": {
        "result": {
            "doc": "What a call did.",
            "values": {"success": {"value": 0, "doc": "Done."}, "error_lost": {"value": -1, "doc": "Lost."}},
        },
    },
    "objects": {"device": {"doc": "A device."}},
    "handles": {"buffer": {"doc": "A buffer."}},
    "callbacks": {
        "log": {"doc": "Logs.", "params": [{"name": "text", "type": "string"}]},
    },
    "structs": {
        "buffer_desc": {"doc": "A buffer to make.", "extensible": True, "members": [{"name": "size", "type": "u64"}]},
        "buffer_name": {
            "doc": "A name for it.",
            "extends": ["buffer_desc"],
            "struct_type": 7,
            "members": [{"name": "name", "type": "string"}],
        },
        "list": {
            "doc": "Counted things.",
            "members": [{"name": "count", "type": "u32"}, {"name": "items", "type": "u32", "array": "count", "pointer": "in"}],
        },
    },
    "functions": {
        "create_buffer": {
            "doc": "Makes a buffer.",
            "returns": "result",
            "params": [
                {"name": "device", "type": "device"},
                {"name": "desc", "type": "buffer_desc", "pointer": "in"},
                {"name": "buffer", "type": "buffer", "pointer": "out"},
            ],
        },
        "destroy_device": {
            "doc": "Destroys a device.",
            "params": [{"name": "device", "type": "device", "optional": True}],
        },
    },
}


def broken(change) -> dict:
    spec = copy.deepcopy(MINIMAL)
    change(spec)
    return spec


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.header = render(Api(copy.deepcopy(MINIMAL)))

    def test_enums_keep_their_values_and_are_forced_to_32_bits(self) -> None:
        self.assertIn("BFR_RESULT_SUCCESS = 0,", self.header)
        self.assertIn("BFR_RESULT_ERROR_LOST = -1,", self.header)
        self.assertIn("BFR_RESULT_FORCE32 = 0x7FFFFFFF", self.header)

    def test_objects_are_opaque_pointers_and_handles_are_64_bit_structs(self) -> None:
        self.assertIn("typedef struct bfr_device bfr_device;", self.header)
        self.assertIn("typedef struct bfr_buffer {\n    uint64_t bits;\n} bfr_buffer;", self.header)
        self.assertIn("    bfr_device* device,\n    const bfr_buffer_desc* desc,\n    bfr_buffer* buffer);", self.header)

    def test_extensible_structs_lead_with_next_and_extensions_with_their_chain(self) -> None:
        self.assertIn("typedef struct bfr_buffer_desc {\n    const bfr_chain* next;", self.header)
        self.assertIn("typedef struct bfr_buffer_name {\n    bfr_chain chain;", self.header)
        self.assertIn("BFR_STRUCT_TYPE_BUFFER_NAME = 7,", self.header)

    def test_flags_are_spelled_out_in_the_docs(self) -> None:
        self.assertIn("// An array of count elements.\n    const uint32_t* items;", self.header)
        self.assertIn("// device: May be null.\nBFR_API void bfr_destroy_device(bfr_device* device);", self.header)

    def test_callbacks_are_function_pointers(self) -> None:
        self.assertIn("typedef void (*bfr_log)(bfr_string text);", self.header)

    def test_long_prototypes_take_one_param_per_line(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["functions"]["create_buffer"]["params"].append({"name": "a_rather_long_parameter_name", "type": "u64"})
        header = render(Api(spec))
        self.assertIn("BFR_API bfr_result bfr_create_buffer(\n    bfr_device* device,\n", header)
        self.assertTrue(all(len(line) <= 100 for line in header.splitlines()))

    def test_the_header_compiles_as_c_and_cpp_and_is_exempt_from_lint_and_format(self) -> None:
        self.assertIn('#ifdef __cplusplus\nextern "C" {\n#endif', self.header)
        self.assertIn("// clang-format off\n// NOLINTBEGIN", self.header)
        self.assertTrue(self.header.endswith("// NOLINTEND\n// clang-format on\n"))


class CheckTests(unittest.TestCase):
    def assert_rejected(self, change, message: str) -> None:
        with self.assertRaisesRegex(SpecError, message):
            Api(broken(change))

    def test_an_enum_value_without_an_explicit_value_is_rejected(self) -> None:
        self.assert_rejected(lambda s: s["enums"]["result"]["values"].update(other=None), "explicit value")

    def test_a_repeated_enum_value_is_rejected(self) -> None:
        self.assert_rejected(
            lambda s: s["enums"]["result"]["values"].update(other={"value": 0, "doc": "Again."}), "repeats"
        )

    def test_an_unknown_type_is_rejected(self) -> None:
        self.assert_rejected(
            lambda s: s["structs"]["buffer_desc"]["members"].append({"name": "x", "type": "nope"}), "unknown type"
        )

    def test_an_unknown_key_is_rejected(self) -> None:
        # What an unquoted comma in a {...} doc turns into.
        self.assert_rejected(lambda s: s["enums"]["result"]["values"]["success"].update({"more text": None}), "keys")

    def test_an_item_without_a_doc_is_rejected(self) -> None:
        self.assert_rejected(lambda s: s["handles"]["buffer"].pop("doc"), "needs a doc")

    def test_extending_a_struct_that_is_not_extensible_is_rejected(self) -> None:
        self.assert_rejected(lambda s: s["structs"]["buffer_name"].update(extends=["list"]), "not an extensible")

    def test_extension_structs_need_unique_struct_types(self) -> None:
        self.assert_rejected(lambda s: s["structs"]["buffer_name"].pop("struct_type"), "no struct_type")

    def test_an_array_needs_its_count_beside_it(self) -> None:
        self.assert_rejected(lambda s: s["structs"]["list"]["members"][1].update(array="missing"), "count")


class RepositoryTests(unittest.TestCase):
    def test_committed_header_matches_the_description(self) -> None:
        self.assertEqual(HEADER.read_text(encoding="utf-8"), render(load(SOURCE)), "run: python3 tools/gen_c_api.py")


if __name__ == "__main__":
    unittest.main()
