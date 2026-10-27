#!/usr/bin/env python3

import copy
import unittest

from bfr_api import Api, SpecError

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


if __name__ == "__main__":
    unittest.main()
