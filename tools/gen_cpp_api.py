#!/usr/bin/env python3
"""Generate the C++ wrapper header from the API description.

    python3 tools/gen_cpp_api.py           # write bindings/cpp/include/bfr.hpp
    python3 tools/gen_cpp_api.py --check   # fail if bfr.hpp is out of date

bfr.hpp wraps bfr.h. Enums become enum classes and structs take C++ types.
Each object and handle becomes a class that owns what it names and destroys
it, and each function a method on the object or handle it acts on, returning
std::expected. bfr_detail.hpp holds the fixed helpers it uses.

Where each function goes follows from api/bfr.yaml:
- A function whose params start with a handle's parent and then the handle
  is a method on the handle; one whose first param is an object, a method on
  that object; one that creates an object from no other, a static method on
  the object it creates.
- A function returning result with one out param returns that param. One
  with an optional out array whose count is inout lists: the method asks
  until the list stops growing and returns a std::vector.
- An object a method creates keeps alive what its creator keeps, such as
  the state of a callback the creator's desc named.
- Methods on objects are const: several threads may call them at once.
  Methods on handles are not, since calls on one handle must not overlap,
  and const would say they may.

What the generator does not handle yet fails with a SpecError, never quietly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

from bfr_api import BUILTINS as C_BUILTINS
from bfr_api import REPO_ROOT, SOURCE, WIDTH, Api, SpecError, c_name, comment, load, macro

HEADER = REPO_ROOT / "bindings" / "cpp" / "include" / "bfr.hpp"

BUILTINS = {
    "bool": "bool",
    "u8": "std::uint8_t",
    "u32": "std::uint32_t",
    "u64": "std::uint64_t",
    "i32": "std::int32_t",
    "f32": "float",
    "size": "std::size_t",
    "opaque": "void*",
    "string": "std::string_view",
}


def pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def camel(name: str) -> str:
    words = pascal(name)
    return words[0].lower() + words[1:]


def doc(text: str, indent: str = "") -> list[str]:
    return comment(text, indent, "///")


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    end = text.find(". ")
    return text if end == -1 else text[: end + 1]


@dataclass
class Method:
    """One function, as a method of the class named owner."""

    function: str
    owner: str
    name: str
    static: bool = False
    # The params that name this, in the C call.
    self_params: list[str] = field(default_factory=list)
    # The param the method returns, or the array a list method returns.
    returned: dict[str, Any] | None = None
    count: dict[str, Any] | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)


class Plan:
    """Where each part of the description goes in C++."""

    def __init__(self, api: Api) -> None:
        self.api = api
        if "result" not in api.enums or "get_last_error" not in api.functions:
            raise SpecError("the C++ wrapper needs the result enum and get_last_error")
        self.destroys = {entry["destroy"] for entry in [*api.objects.values(), *api.handles.values()]}
        self.methods: dict[str, list[Method]] = {name: [] for name in [*api.objects, *api.handles]}
        for name, function in api.functions.items():
            if name in self.destroys or name == "get_last_error":
                continue
            method = self._place(name, function)
            if any(other.name == method.name for other in self.methods[method.owner]):
                raise SpecError(f"{name}: {pascal(method.owner)} already has a method {method.name}")
            self.methods[method.owner].append(method)

        self.out_structs = self._with_members(self._structs_passed("out"))
        self.in_structs = self._structs_passed("in")
        if self.out_structs & self.in_structs:
            raise SpecError(f"{sorted(self.out_structs & self.in_structs)}: the C++ generator does not yet handle a "
                            "struct passed both in and out")
        # In the description's order, as everything rendered is: a set's order
        # changes from run to run.
        self.extensions = {
            desc: [name for name, struct in api.structs.items() if desc in struct.get("extends", [])]
            for desc in api.structs
            if desc in self.in_structs
        }
        self.extensions = {desc: names for desc, names in self.extensions.items() if names}
        self.in_structs |= {name for names in self.extensions.values() for name in names}
        for name in sorted(self.in_structs | self.out_structs):
            self._check_members(name)

    def _place(self, name: str, function: dict[str, Any]) -> Method:
        api = self.api
        params = function["params"]
        types = [param["type"] for param in params]
        if function.get("returns") not in (None, "result"):
            raise SpecError(f"{name}: the C++ generator does not yet handle a function returning {function['returns']}")
        handle = types[1] if len(types) > 1 and params[1].get("pointer") is None else None
        if handle in api.handles and api.handles[handle]["parent"] == types[0]:
            method = Method(name, handle, self._method_name(name, types[0], handle), self_params=params[:2])
        elif types and types[0] in api.objects:
            method = Method(name, types[0], self._method_name(name, types[0], None), self_params=params[:1])
        else:
            created = [p["type"] for p in params if p.get("pointer") == "out" and p["type"] in api.objects]
            if len(created) != 1:
                raise SpecError(f"{name}: the C++ generator cannot tell which class it belongs to")
            stem = name.removesuffix(f"_{created[0]}")
            method = Method(name, created[0], camel(stem), static=True)
        rest = params[len(method.self_params) :]

        lists = [p for p in rest if p.get("array") and p.get("pointer") == "out"]
        outs = [p for p in rest if p.get("pointer") == "out" and not p.get("array")]
        if lists and outs or len(lists) + len(outs) > 1:
            raise SpecError(f"{name}: the C++ generator handles at most one out param")
        if lists:
            array = lists[0]
            count = next(p for p in rest if p["name"] == array["array"])
            if not array.get("optional") or count.get("pointer") != "inout":
                raise SpecError(f"{name}: a listed array must be optional and its count inout")
            method.returned, method.count = array, count
        elif outs:
            method.returned = outs[0]
        if method.returned is not None and function.get("returns") != "result":
            raise SpecError(f"{name}: an out param needs the function to return result")
        returned = method.returned["type"] if method.returned else None
        if returned in api.handles and api.handles[returned]["parent"] != method.owner:
            raise SpecError(f"{name}: a {returned} is made by a method of its parent")
        method.inputs = [p for p in rest if p is not method.returned and p is not method.count]
        for param in method.inputs:
            if param.get("pointer") not in (None, "in") or param.get("array"):
                raise SpecError(f"{name}.{param['name']}: the C++ generator does not yet handle this param")
            if param.get("pointer") == "in" and param["type"] not in api.structs:
                raise SpecError(f"{name}.{param['name']}: only structs are passed by pointer")
        return method

    def _method_name(self, name: str, owner: str, handle: str | None) -> str:
        stem = name.removeprefix(f"{owner}_")
        if handle is not None:
            stem = stem.removesuffix(f"_{handle}")
        return camel(stem.removeprefix("get_"))

    def _structs_passed(self, direction: str) -> set[str]:
        found = set()
        for name, function in self.api.functions.items():
            if name in self.destroys:
                continue
            for param in function["params"]:
                if param["type"] in self.api.structs and param.get("pointer") == direction:
                    found.add(param["type"])
        return found

    def _with_members(self, structs: set[str]) -> set[str]:
        """structs and every struct their members hold, however deep."""
        found = set(structs)
        pending = list(structs)
        while pending:
            for member in self.api.structs[pending.pop()]["members"]:
                if member["type"] in self.api.structs and member["type"] not in found:
                    found.add(member["type"])
                    pending.append(member["type"])
        return found

    def _check_members(self, name: str) -> None:
        api = self.api
        members = api.structs[name]["members"]
        for member in members:
            where = f"{name}.{member['name']}"
            kind = member["type"]
            if member.get("pointer") or member.get("array"):
                raise SpecError(f"{where}: the C++ generator does not yet handle pointer or array members")
            if kind in api.callbacks:
                if name in self.out_structs:
                    raise SpecError(f"{where}: a callback is passed in, not out")
                if not any(other.get("user_data") == member["name"] or other.get("user_data") == kind
                           for other in members):
                    raise SpecError(f"{where}: a callback needs its user_data beside it")
                opaque = [p for p in api.callbacks[kind]["params"] if p["type"] == "opaque"]
                if len(opaque) != 1:
                    raise SpecError(f"{kind}: a callback takes exactly one opaque param, its user data")
            elif kind in api.structs:
                if name not in self.out_structs:
                    raise SpecError(f"{where}: the C++ generator handles a struct member only in a struct passed out")
            elif member.get("fixed") and kind in ("bool", "string", "opaque"):
                raise SpecError(f"{where}: the C++ generator does not yet handle a fixed array of {kind}")
            elif kind not in BUILTINS and kind not in api.enums:
                raise SpecError(f"{where}: the C++ generator does not yet handle a {kind} member")

    def callbacks_of(self, struct: str) -> list[dict[str, Any]]:
        return [m for m in self.api.structs[struct]["members"] if m["type"] in self.api.callbacks]

    def is_extensible(self, struct: str) -> bool:
        return struct in self.extensions


def cpp_type(api: Api, kind: str, out: bool = False) -> str:
    if kind == "string" and out:
        return "std::string"
    if kind in BUILTINS:
        return BUILTINS[kind]
    return pascal(kind)


def member_type(api: Api, member: dict[str, Any], out: bool) -> str:
    """A struct member's C++ type: a std::array for a fixed one."""
    kind = cpp_type(api, member["type"], out)
    return f"std::array<{kind}, {member['fixed']}>" if member.get("fixed") else kind


def member_from_c(api: Api, member: dict[str, Any], value: str) -> str:
    """A struct member's C value, value, as its C++ type."""
    kind = member["type"]
    if not member.get("fixed"):
        return from_c(api, kind, value)
    if kind in BUILTINS:
        return f"std::to_array({value})"
    element = from_c(api, kind, "element")
    return f"arrayFrom<{cpp_type(api, kind)}>({value}, [](const {c_type(kind)}& element) {{ return {element}; }})"


def to_c(api: Api, kind: str, value: str) -> str:
    """value, of the C++ type for kind, as its C type."""
    if kind == "bool":
        return f"{value} ? BFR_TRUE : BFR_FALSE"
    if kind == "string":
        return f"detail::toC({value})"
    if kind in api.enums:
        return f"static_cast<{c_name(kind)}>({value})"
    return value


def from_c(api: Api, kind: str, value: str) -> str:
    """value, of kind's C type, as its C++ type."""
    if kind == "bool":
        return f"{value} != BFR_FALSE"
    if kind == "string":
        return f"std::string{{detail::text({value})}}"
    if kind in api.enums:
        return f"static_cast<{pascal(kind)}>({value})"
    if kind in api.structs:
        return f"fromC({value})"
    return value


def callback_arg(api: Api, kind: str, value: str) -> str:
    """value, of kind's C type, as a callback's C++ argument: strings are views."""
    return f"text({value})" if kind == "string" else from_c(api, kind, value)


def c_type(kind: str) -> str:
    return C_BUILTINS.get(kind) or c_name(kind)


def initializer(fields: list[str]) -> list[str]:
    """A return of fields as a designated initializer: on one line, or one field a line if that
    is wider than WIDTH."""
    line = f"    return {{{', '.join(fields)}}};"
    if len(line) <= WIDTH:
        return [line]
    return ["    return {"] + [f"        {field}," for field in fields] + ["    };"]


def failure_check(call: str) -> list[str]:
    return [
        f"    if (const bfr_result result = {call}; result < 0) {{",
        "        return std::unexpected(detail::failure(result));",
        "    }",
    ]


class Renderer:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.api = plan.api
        self.out: list[str] = []

    def emit(self, *lines: str) -> None:
        self.out.extend(lines)

    def render(self) -> str:
        api = self.api
        self.emit(
            "// Generated by tools/gen_cpp_api.py from api/bfr.yaml. Do not edit.",
            "//",
            "// The BFR C++ API, over bfr.h. The generator lays this file out, not clang-format, and",
            "// the C++ lint rules do not apply to it.",
            "// clang-format off",
            "// NOLINTBEGIN",
            "#pragma once",
            "",
            '#include "bfr.h"',
            '#include "bfr_detail.hpp"',
            "",
            "#include <array>",
            "#include <cstddef>",
            "#include <cstdint>",
            "#include <expected>",
            "#include <functional>",
            "#include <memory>",
            "#include <string>",
            "#include <string_view>",
            "#include <utility>",
            "#include <vector>",
            "",
            "namespace bfr {",
        )
        for name, enum in api.enums.items():
            self.emit("")
            self.emit(*doc(enum["doc"]))
            self.emit(f"enum class {pascal(name)} : std::int32_t {{")
            for value_name, value in enum["values"].items():
                if value.get("doc"):
                    self.emit(*doc(value["doc"], "    "))
                self.emit(f"    {pascal(value_name)} = {macro(name, value_name)},")
            self.emit("};")
        self.emit(
            "",
            "/// A failed call: what it returned and the message bfr_get_last_error gave for it.",
            "struct Error {",
            "    Result code{};",
            "    std::string message;",
            "};",
            "",
            "/// What a call returns: its value, or why it failed.",
            "template <class T>",
            "using Expected = std::expected<T, Error>;",
        )
        for name, callback in api.callbacks.items():
            self.emit("")
            self.emit(*doc(callback["doc"] + " May be called on several threads at once."))
            params = [p for p in callback["params"] if p["type"] != "opaque"]
            for param in params:
                if param.get("doc"):
                    self.emit(*doc(f"{param['name']}: {param['doc']}"))
            signature = ", ".join(f"{cpp_type(api, p['type'])} {camel(p['name'])}" for p in params)
            self.emit(f"using {pascal(name)} = std::function<void({signature})>;")
        for name in [*api.objects, *api.handles]:
            self.emit("")
            self.emit(f"class {pascal(name)};")
        for name, struct in api.structs.items():
            if name in self.plan.in_structs or name in self.plan.out_structs:
                self.struct(name, struct)
        self.detail()
        for name in api.objects:
            self.object_class(name)
        for name in api.handles:
            self.handle_class(name)
        for owner, methods in self.plan.methods.items():
            for method in methods:
                self.method_definition(owner, method)
        self.emit("", "} // namespace bfr", "", "// NOLINTEND", "// clang-format on")
        return "\n".join(self.out) + "\n"

    def struct(self, name: str, struct: dict[str, Any]) -> None:
        api = self.api
        out = name in self.plan.out_structs
        text = struct["doc"]
        if self.plan.is_extensible(name):
            extensions = ", ".join(pascal(e) for e in self.plan.extensions[name])
            text += f" Pass any of {extensions} after it."
        if struct.get("extends"):
            text += " Extends " + ", ".join(pascal(base) for base in struct["extends"]) + "."
        self.emit("")
        self.emit(*doc(text))
        self.emit(f"struct {pascal(name)} {{")
        for member in struct["members"]:
            if member.get("user_data"):
                continue
            member_doc = member.get("doc", "")
            if member.get("optional") and member["type"] in api.callbacks:
                member_doc = (member_doc + " May be empty.").strip()
            if member_doc:
                self.emit(*doc(member_doc, "    "))
            self.emit(f"    {member_type(api, member, out)} {camel(member['name'])}{{}};")
        self.emit("};")

    def detail(self) -> None:
        api = self.api
        plan = self.plan
        self.emit("", "namespace detail {", "")
        self.emit(
            "/// The error for result, with the calling thread's message for it.",
            "inline Error failure(bfr_result result) {",
            "    return {.code = static_cast<Result>(result), .message = lastError()};",
            "}",
        )
        for name, callback in api.callbacks.items():
            params = callback["params"]
            c_params = ", ".join(f"{c_type(p['type'])} {p['name']}" for p in params)
            user_data = next(p for p in params if p["type"] == "opaque")["name"]
            args = ", ".join(callback_arg(api, p["type"], p["name"]) for p in params if p["type"] != "opaque")
            self.emit(
                "",
                f"/// Calls the {pascal(name)} user_data points at; nothing it throws reaches C.",
                f"inline void {camel(name)}Trampoline({c_params}) noexcept {{",
                f"    callQuietly([&] {{ (*static_cast<const {pascal(name)}*>({user_data}))({args}); }});",
                "}",
            )
        for name in sorted(plan.in_structs, key=list(api.structs).index):
            self.to_c_function(name, api.structs[name])
        for name in sorted(plan.out_structs, key=list(api.structs).index):
            struct = api.structs[name]
            members = [f".{camel(m['name'])} = {member_from_c(api, m, 'value.' + m['name'])}"
                       for m in struct["members"]]
            self.emit("", f"inline {pascal(name)} fromC(const {c_name(name)}& value) {{",
                      *initializer(members), "}")
        for desc, extensions in plan.extensions.items():
            self.emit("")
            for extension in extensions:
                self.emit(f"template <> inline constexpr bool kExtends<{pascal(extension)}, {pascal(desc)}> = true;")
        self.emit("", "} // namespace detail")

    def to_c_function(self, name: str, struct: dict[str, Any]) -> None:
        api = self.api
        callbacks = self.plan.callbacks_of(name)
        params = [f"const {pascal(name)}&{' value' if struct['members'] else ''}"]
        fields = []
        if struct.get("extensible"):
            params.append("const bfr_chain* next")
            fields.append(".next = next")
        if struct.get("extends"):
            fields.append(f".chain = {{.next = nullptr, .type = {macro('struct_type', name)}}}")
        params += [f"{pascal(c['type'])}* {camel(c['name'])}" for c in callbacks]
        for member in struct["members"]:
            if member["type"] in api.callbacks:
                fields.append(f".{member['name']} = {camel(member['name'])} != nullptr ? "
                              f"&{camel(member['type'])}Trampoline : nullptr")
            elif member.get("user_data"):
                partner = next(c for c in callbacks if c["type"] == member["user_data"]
                               or c["name"] == member["user_data"])
                fields.append(f".{member['name']} = {camel(partner['name'])}")
            else:
                fields.append(f".{member['name']} = {to_c(api, member['type'], 'value.' + camel(member['name']))}")
        self.emit("", f"inline {c_name(name)} toC({', '.join(params)}) {{", *initializer(fields), "}")

    def owner_class(self, name: str, members: list[str], destroy: str, moved: list[tuple[str, str]],
                    accessor: tuple[str, str, str], private_ctor: tuple[str, str]) -> None:
        api = self.api
        entry = api.objects.get(name) or api.handles[name]
        cls = pascal(name)
        creators = sorted({owner for owner, methods in self.plan.methods.items() for m in methods
                           if m.returned is not None and m.count is None and m.returned["type"] == name
                           and owner != name})
        self.emit("")
        self.emit(*doc(entry["doc"]))
        self.emit(f"class {cls} {{", "public:")
        self.emit(f"    /// Owns nothing, as a moved-from {cls} does.", f"    {cls}() = default;")
        self.emit(*doc(api.functions[entry["destroy"]]["doc"], "    "))
        self.emit(f"    ~{cls}() {{ {destroy}; }}")
        self.emit(f"    {cls}(const {cls}&) = delete;", f"    {cls}& operator=(const {cls}&) = delete;")
        inits = ", ".join(f"{member}({how})" for member, how in moved)
        self.emit(f"    {cls}({cls}&& other) noexcept : {inits} {{}}")
        self.emit(f"    {cls}& operator=({cls}&& other) noexcept {{", "        if (this != &other) {",
                  f"            {destroy};")
        for member, how in moved:
            self.emit(f"            {member} = {how};")
        self.emit("        }", "        return *this;", "    }")
        kind, getter, test = accessor
        self.emit("", "    /// The C value, for calls bfr.hpp does not wrap; this still owns it.",
                  f"    [[nodiscard]] {kind} get() const {{ return {getter}; }}",
                  f"    [[nodiscard]] explicit operator bool() const {{ return {test}; }}")
        for method in self.plan.methods[name]:
            self.emit("")
            self.method_declaration(method)
        ctor_params, ctor_inits = private_ctor
        self.emit("", "private:", f"    {cls}({ctor_params}) : {ctor_inits} {{}}")
        for creator in creators:
            self.emit(f"    friend class {pascal(creator)};")
        self.emit("")
        self.emit(*members)
        self.emit("};")

    def object_class(self, name: str) -> None:
        member = f"{camel(name)}_"
        destroy = f"{c_name(self.api.objects[name]['destroy'])}({member})"
        self.owner_class(
            name,
            [f"    {c_name(name)}* {member} = nullptr;",
             "    // What the C object may still call into, such as a callback's state; it goes after the",
             "    // object does.",
             "    detail::KeepAlive keepAlive_;"],
            destroy,
            [(member, f"std::exchange(other.{member}, nullptr)"), ("keepAlive_", "std::move(other.keepAlive_)")],
            (f"{c_name(name)}*", member, f"{member} != nullptr"),
            (f"{c_name(name)}* {camel(name)}, detail::KeepAlive keepAlive",
             f"{member}({camel(name)}), keepAlive_(std::move(keepAlive))"),
        )

    def handle_class(self, name: str) -> None:
        handle = self.api.handles[name]
        parent = handle["parent"]
        parent_member, member = f"{camel(parent)}_", f"{camel(name)}_"
        destroy = f"{c_name(handle['destroy'])}({parent_member}, {member})"
        self.owner_class(
            name,
            [f"    // Borrowed: the {pascal(parent)} that made this outlives it.",
             f"    {c_name(parent)}* {parent_member} = nullptr;",
             f"    {c_name(name)} {member}{{}};"],
            destroy,
            [(parent_member, f"std::exchange(other.{parent_member}, nullptr)"),
             (member, f"std::exchange(other.{member}, {{}})")],
            (c_name(name), member, f"{member}.bits != 0"),
            (f"{c_name(parent)}* {camel(parent)}, {c_name(name)} {camel(name)}",
             f"{parent_member}({camel(parent)}), {member}({camel(name)})"),
        )

    def signature(self, method: Method) -> tuple[list[str], str, str]:
        """The template lines, return type and params of method."""
        api = self.api
        template: list[str] = []
        params = []
        for param in method.inputs:
            kind = param["type"]
            if param.get("pointer") == "in":
                params.append(f"const {pascal(kind)}& {camel(param['name'])}")
                if self.plan.is_extensible(kind):
                    template = ["template <class... Extensions>",
                                f"    requires(detail::kExtends<Extensions, {pascal(kind)}> && ...)"]
                    params.append("const Extensions&... extensions")
            else:
                params.append(f"{cpp_type(api, kind)} {camel(param['name'])}")
        if method.returned is None:
            returns = "Expected<void>"
        elif method.count is not None:
            returns = f"Expected<std::vector<{pascal(method.returned['type'])}>>"
        else:
            returns = f"Expected<{cpp_type(api, method.returned['type'], out=True)}>"
        return template, returns, ", ".join(params)

    def const(self, method: Method) -> str:
        """ const on a method of an object, which several threads may call at
        once; not on a handle's, whose calls the caller keeps from overlapping,
        nor on a static."""
        return " const" if not method.static and method.owner in self.api.objects else ""

    def method_declaration(self, method: Method) -> None:
        function = self.api.functions[method.function]
        text = first_sentence(function["doc"]) if method.count is not None else function["doc"]
        self.emit(*doc(text, "    "))
        template, returns, params = self.signature(method)
        self.emit(*(f"    {line}" for line in template))
        qualifiers = "static " if method.static else ""
        const = self.const(method)
        self.emit(f"    [[nodiscard]] {qualifiers}{returns} {method.name}({params}){const};")

    def method_definition(self, owner: str, method: Method) -> None:
        api = self.api
        function = api.functions[method.function]
        template, returns, params = self.signature(method)
        const = self.const(method)
        self.emit("")
        self.emit(*template)
        inline = "" if template else "inline "
        self.emit(f"{inline}{returns} {pascal(owner)}::{method.name}({params}){const} {{")

        args = [f"{camel(p['type'])}_" for p in method.self_params]
        # The state of a callback the desc names, which the object made keeps.
        callback_state = None
        for param in method.inputs:
            kind, name = param["type"], camel(param["name"])
            if param.get("pointer") != "in":
                args.append(to_c(api, kind, name))
                continue
            extra = []
            if self.plan.is_extensible(kind):
                self.emit("    const detail::Chain<decltype(detail::toC(extensions))...> chain{"
                          "detail::toC(extensions)...};")
                extra.append("chain.head()")
            elif api.structs[kind].get("extensible"):
                extra.append("nullptr")
            for callback in self.plan.callbacks_of(kind):
                state = camel(callback["name"])
                self.emit(f"    auto {state} = {name}.{state} ? std::make_shared<{pascal(callback['type'])}>("
                          f"{name}.{state}) : nullptr;")
                extra.append(f"{state}.get()")
                if not method.static or callback_state is not None:
                    raise SpecError(f"{method.function}: the C++ generator handles one callback, in the desc "
                                    "of a static create")
                callback_state = state
            c_value = f"c{pascal(param['name'])}"
            self.emit(f"    const {c_name(kind)} {c_value} = detail::toC({', '.join([name, *extra])});")
            args.append(f"&{c_value}")
        if callback_state is not None:
            keep_alive = f"std::move({callback_state})"
        else:
            keep_alive = "nullptr" if method.static else "keepAlive_"

        call = c_name(method.function)
        returned = method.returned
        if returned is not None and method.count is not None:
            self.list_body(method, call, args)
        else:
            if returned is not None:
                kind, name = returned["type"], camel(returned["name"])
                initial = " = nullptr" if kind in api.objects else "{}"
                self.emit(f"    {c_type(kind)}{'*' if kind in api.objects else ''} {name}{initial};")
                args_in_order = self.ordered_args(method, args, {returned["name"]: f"&{name}"})
            else:
                args_in_order = self.ordered_args(method, args, {})
            self.emit(*failure_check(f"{call}({', '.join(args_in_order)})"))
            if returned is None:
                self.emit("    return {};")
            elif kind in api.objects:
                self.emit(f"    return {pascal(kind)}{{{name}, {keep_alive}}};")
            elif kind in api.handles:
                self.emit(f"    return {pascal(kind)}{{{camel(api.handles[kind]['parent'])}_, {name}}};")
            elif kind in api.structs:
                self.emit(f"    return detail::fromC({name});")
            else:
                self.emit(f"    return {from_c(api, kind, name)};")
        self.emit("}")

    def ordered_args(self, method: Method, args: list[str], extra: dict[str, str]) -> list[str]:
        """args, which follow the self params and inputs in order, with extra
        placed where its params stand in the C call."""
        function = self.api.functions[method.function]
        queue = list(args)
        ordered = []
        for param in function["params"]:
            if param["name"] in extra:
                ordered.append(extra[param["name"]])
            else:
                ordered.append(queue.pop(0))
        return ordered

    def list_body(self, method: Method, call: str, args: list[str]) -> None:
        api = self.api
        array, count = method.returned, method.count
        kind = array["type"]
        listed = f"{camel(array['name'])}C"
        count_name = camel(count["name"])
        asking = self.ordered_args(method, args, {count["name"]: f"&{count_name}", array["name"]: "nullptr"})
        filling = self.ordered_args(method, args, {count["name"]: f"&{count_name}", array["name"]: f"{listed}.data()"})
        self.emit(
            f"    std::vector<{c_name(kind)}> {listed};",
            f"    std::uint32_t {count_name} = 0;",
            "    // Asked again while the list grows between the two calls.",
            "    for (;;) {",
            f"        bfr_result result = {call}({', '.join(asking)});",
            "        if (result < 0) {",
            "            return std::unexpected(detail::failure(result));",
            "        }",
            f"        {listed}.resize({count_name});",
            f"        result = {call}({', '.join(filling)});",
            "        if (result < 0) {",
            "            return std::unexpected(detail::failure(result));",
            "        }",
            "        if (result != BFR_RESULT_INCOMPLETE) {",
            "            break;",
            "        }",
            "    }",
            f"    std::vector<{pascal(kind)}> {camel(array['name'])};",
            f"    {camel(array['name'])}.reserve({count_name});",
            f"    for (std::uint32_t i = 0; i < {count_name}; ++i) {{",
            f"        {camel(array['name'])}.push_back(detail::fromC({listed}[i]));",
            "    }",
            f"    return {camel(array['name'])};",
        )


def render(api: Api) -> str:
    return Renderer(Plan(api)).render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the header is out of date")
    args = parser.parse_args(argv)
    try:
        header = render(load(SOURCE))
    except SpecError as error:
        print(f"gen_cpp_api: {SOURCE.relative_to(REPO_ROOT)}: {error}", file=sys.stderr)
        return 1
    if args.check:
        if not HEADER.exists() or HEADER.read_text(encoding="utf-8") != header:
            print(f"gen_cpp_api: {HEADER.relative_to(REPO_ROOT)} is out of date", file=sys.stderr)
            print("Run: python3 tools/gen_cpp_api.py", file=sys.stderr)
            return 1
        return 0
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text(header, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
