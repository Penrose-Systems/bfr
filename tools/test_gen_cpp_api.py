#!/usr/bin/env python3

import copy
import os
import subprocess
import sys
import unittest

from bfr_api import REPO_ROOT, SOURCE, Api, SpecError, load
from gen_cpp_api import HEADER, render
from test_bfr_api import MINIMAL


def with_more(spec: dict) -> dict:
    """MINIMAL with the error source, a static create whose desc names a
    callback, and a list."""
    spec = copy.deepcopy(spec)
    spec["functions"]["get_last_error"] = {"doc": "The last error.", "returns": "string", "params": []}
    spec["structs"]["format"] = {"doc": "A format.", "members": [{"name": "bits", "type": "u32"}]}
    spec["functions"]["create_device"] = {
        "doc": "Makes a device.",
        "returns": "result",
        "params": [
            {"name": "desc", "type": "logger", "pointer": "in"},
            {"name": "device", "type": "device", "pointer": "out"},
        ],
    }
    spec["functions"]["device_get_formats"] = {
        "doc": "Lists the formats. With formats null, writes the count.",
        "returns": "result",
        "params": [
            {"name": "device", "type": "device"},
            {"name": "count", "type": "u32", "pointer": "inout"},
            {"name": "formats", "type": "format", "array": "count", "pointer": "out", "optional": True},
        ],
    }
    return spec


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.header = render(Api(with_more(MINIMAL)))

    def test_enums_are_enum_classes_valued_by_the_c_names(self) -> None:
        self.assertIn("enum class Result : std::int32_t {", self.header)
        self.assertIn("    ErrorLost = BFR_RESULT_ERROR_LOST,", self.header)

    def test_objects_and_handles_destroy_what_they_own(self) -> None:
        self.assertIn("    ~Device() { bfr_destroy_device(device_); }", self.header)
        self.assertIn("    ~Buffer() { bfr_destroy_buffer(device_, buffer_); }", self.header)
        self.assertIn("    // Borrowed: the Device that made this outlives it.\n    bfr_device* device_ = nullptr;",
                      self.header)

    def test_a_method_returns_its_out_param_and_takes_extensions_after_its_desc(self) -> None:
        self.assertIn(
            "    template <class... Extensions>\n"
            "        requires(detail::kExtends<Extensions, BufferDesc> && ...)\n"
            "    [[nodiscard]] Expected<Buffer> createBuffer(const BufferDesc& desc, "
            "const Extensions&... extensions) const;",
            self.header,
        )
        self.assertIn("template <> inline constexpr bool kExtends<BufferName, BufferDesc> = true;", self.header)
        self.assertIn("    return Buffer{device_, buffer};", self.header)

    def test_methods_on_objects_are_const_and_on_handles_are_not(self) -> None:
        self.assertIn("Expected<std::vector<Format>> formats() const;", self.header)
        spec = with_more(MINIMAL)
        spec["functions"]["device_fill_buffer"] = {
            "doc": "Fills a buffer.",
            "returns": "result",
            "params": [{"name": "device", "type": "device"}, {"name": "buffer", "type": "buffer"}],
        }
        header = render(Api(spec))
        self.assertIn("    [[nodiscard]] Expected<void> fill();", header)
        self.assertIn("inline Expected<void> Buffer::fill() {", header)

    def test_a_static_create_keeps_its_callback_alive_with_what_it_makes(self) -> None:
        self.assertIn("using Log = std::function<void(std::string_view text)>;", self.header)
        self.assertIn("    auto log = desc.log ? std::make_shared<Log>(desc.log) : nullptr;", self.header)
        self.assertIn("    return Device{device, std::move(log)};", self.header)
        self.assertIn("callQuietly([&] { (*static_cast<const Log*>(user_data))(text(text)); });", self.header)

    def test_a_listing_returns_a_vector_and_asks_again_while_it_grows(self) -> None:
        self.assertIn("    [[nodiscard]] Expected<std::vector<Format>> formats() const;", self.header)
        self.assertIn("    /// Lists the formats.\n", self.header)
        self.assertIn("        if (result != BFR_RESULT_INCOMPLETE) {", self.header)

    def test_an_out_struct_converts_the_structs_and_arrays_it_holds(self) -> None:
        spec = with_more(MINIMAL)
        spec["structs"]["format"]["members"] += [
            {"name": "uuid", "type": "u8", "fixed": 16},
            {"name": "period", "type": "f32"},
        ]
        spec["structs"]["caps"] = {
            "doc": "Caps.",
            "members": [{"name": "format", "type": "format"}, {"name": "formats", "type": "format", "fixed": 2}],
        }
        spec["functions"]["device_get_caps"] = {
            "doc": "Gets caps.",
            "returns": "result",
            "params": [{"name": "device", "type": "device"}, {"name": "caps", "type": "caps", "pointer": "out"}],
        }
        header = render(Api(spec))
        self.assertIn("    std::array<std::uint8_t, 16> uuid{};", header)
        self.assertIn("    float period{};", header)
        self.assertIn(".uuid = std::to_array(value.uuid)", header)
        self.assertIn("        .format = fromC(value.format),", header)
        self.assertIn("arrayFrom<Format>(value.formats, [](const bfr_format& element) { return fromC(element); })",
                      header)
        self.assertIn("    [[nodiscard]] Expected<Caps> caps() const;", header)

    def test_a_plain_out_param_is_returned_as_its_value(self) -> None:
        spec = with_more(MINIMAL)
        spec["functions"]["device_get_native"] = {
            "doc": "Gets the native object.",
            "returns": "result",
            "params": [{"name": "device", "type": "device"}, {"name": "native", "type": "opaque", "pointer": "out"}],
        }
        header = render(Api(spec))
        self.assertIn("    [[nodiscard]] Expected<void*> native() const;", header)
        self.assertIn("    void* native{};", header)
        self.assertIn("    return native;", header)

    def test_the_header_is_exempt_from_lint_and_format(self) -> None:
        self.assertIn("// clang-format off\n// NOLINTBEGIN", self.header)
        self.assertTrue(self.header.endswith("// NOLINTEND\n// clang-format on\n"))


class PlacementTests(unittest.TestCase):
    def assert_rejected(self, change, message: str) -> None:
        spec = with_more(MINIMAL)
        change(spec)
        with self.assertRaisesRegex(SpecError, message):
            render(Api(spec))

    def test_a_function_with_no_class_is_rejected(self) -> None:
        self.assert_rejected(
            lambda s: s["functions"].update(frobnicate={"doc": "Frobs.", "params": [{"name": "x", "type": "u32"}]}),
            "cannot tell which class",
        )

    def test_two_out_params_are_rejected(self) -> None:
        self.assert_rejected(
            lambda s: s["functions"]["create_buffer"]["params"].append(
                {"name": "other", "type": "buffer", "pointer": "out"}
            ),
            "at most one out param",
        )

    def test_a_handle_made_away_from_its_parent_is_rejected(self) -> None:
        def change(spec: dict) -> None:
            spec["objects"]["pool"] = {"doc": "A pool.", "destroy": "destroy_pool"}
            spec["functions"]["destroy_pool"] = {"doc": "Destroys.", "params": [{"name": "pool", "type": "pool"}]}
            spec["functions"]["pool_make_buffer"] = {
                "doc": "Makes.",
                "returns": "result",
                "params": [{"name": "pool", "type": "pool"}, {"name": "buffer", "type": "buffer", "pointer": "out"}],
            }

        self.assert_rejected(change, "made by a method of its parent")

    def test_a_struct_passed_in_and_out_is_rejected(self) -> None:
        self.assert_rejected(
            lambda s: s["functions"].update(
                device_set_format={
                    "doc": "Sets a format.",
                    "returns": "result",
                    "params": [{"name": "device", "type": "device"}, {"name": "format", "type": "format", "pointer": "in"}],
                }
            ),
            "both in and out",
        )


class DeterminismTests(unittest.TestCase):
    def test_the_header_is_the_same_whatever_the_hash_seed(self) -> None:
        script = "import sys; sys.path.insert(0, 'tools'); from gen_cpp_api import render; from bfr_api import " \
                 "load, SOURCE; print(render(load(SOURCE)))"
        outputs = {
            subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env={**os.environ, "PYTHONHASHSEED": seed},
                           capture_output=True, text=True, check=True).stdout
            for seed in ("0", "1", "2", "3")
        }
        self.assertEqual(len(outputs), 1)


class RepositoryTests(unittest.TestCase):
    def test_committed_header_matches_the_description(self) -> None:
        self.assertEqual(HEADER.read_text(encoding="utf-8"), render(load(SOURCE)), "run: python3 tools/gen_cpp_api.py")


if __name__ == "__main__":
    unittest.main()
