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
    "objects": {"device": {"doc": "A device.", "destroy": "destroy_device"}},
    "handles": {"buffer": {"doc": "A buffer.", "parent": "device", "destroy": "destroy_buffer"}},
    "callbacks": {
        "log": {
            "doc": "Logs.",
            "params": [{"name": "text", "type": "string"}, {"name": "user_data", "type": "opaque"}],
        },
    },
    "structs": {
        "buffer_desc": {"doc": "A buffer to make.", "extensible": True, "members": [{"name": "size", "type": "u64"}]},
        "logger": {
            "doc": "Where to log.",
            "members": [
                {"name": "log", "type": "log"},
                {"name": "log_user_data", "type": "opaque", "user_data": "log"},
            ],
        },
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
        "destroy_buffer": {
            "doc": "Destroys a buffer.",
            "params": [{"name": "device", "type": "device"}, {"name": "buffer", "type": "buffer"}],
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

    def test_an_object_without_a_destroy_is_rejected(self) -> None:
        self.assert_rejected(lambda s: s["objects"]["device"].pop("destroy"), "needs destroy")

    def test_a_destroy_that_takes_other_params_is_rejected(self) -> None:
        self.assert_rejected(lambda s: s["handles"]["buffer"].update(destroy="destroy_device"), "must take device, buffer")

    def test_a_destroy_that_returns_something_is_rejected(self) -> None:
        self.assert_rejected(lambda s: s["functions"]["destroy_device"].update(returns="result"), "return nothing")

    def test_a_handle_needs_an_object_for_its_parent(self) -> None:
        self.assert_rejected(lambda s: s["handles"]["buffer"].update(parent="buffer_desc"), "needs a parent")

    def test_user_data_must_sit_beside_its_callback(self) -> None:
        self.assert_rejected(lambda s: s["structs"]["logger"]["members"].pop(0), "beside a log member")

    def test_user_data_must_be_opaque(self) -> None:
        self.assert_rejected(lambda s: s["structs"]["logger"]["members"][1].update(type="u64"), "must be opaque")

    def test_a_fixed_length_must_be_positive(self) -> None:
        for length in (0, -1, True, "16"):
            self.assert_rejected(
                lambda s, length=length: s["structs"]["buffer_desc"]["members"].append(
                    {"name": "id", "type": "u8", "fixed": length}
                ),
                "positive length",
            )

    def test_only_a_struct_member_may_be_fixed(self) -> None:
        self.assert_rejected(
            lambda s: s["functions"]["create_buffer"]["params"].append({"name": "id", "type": "u8", "fixed": 4}),
            "only a struct member",
        )

    def test_a_struct_member_must_be_declared_first(self) -> None:
        self.assert_rejected(
            lambda s: s["structs"]["buffer_desc"]["members"].append({"name": "items", "type": "list"}),
            "declare list before buffer_desc",
        )


if __name__ == "__main__":
    unittest.main()
