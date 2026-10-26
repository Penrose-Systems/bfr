#!/usr/bin/env python3
"""Generate the C API header from the API description.

    python3 tools/gen_c_api.py           # write include/bfr/public/bfr.h
    python3 tools/gen_c_api.py --check   # fail if bfr.h is out of date
    python3 tools/gen_c_api.py --functions   # list the C functions, one a line

api/bfr.yaml explains the description's format. The output depends only on
the description, so it is the same on every machine.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "api" / "bfr.yaml"
HEADER = REPO_ROOT / "include" / "bfr" / "public" / "bfr.h"

PREFIX = "bfr_"
MACRO_PREFIX = "BFR_"
FORCE32 = "0x7FFFFFFF"
WIDTH = 100

BUILTINS = {
    "bool": "bfr_bool",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "i32": "int32_t",
    "size": "size_t",
    "opaque": "void*",
    "string": "bfr_string",
}
POINTERS = {"in", "out", "inout"}
# The keys each kind of entry may have, so a typo or a split value fails.
KEYS = {
    "enum": {"doc", "values"},
    "value": {"value", "doc"},
    "object": {"doc"},
    "handle": {"doc"},
    "callback": {"doc", "params"},
    "struct": {"doc", "extensible", "extends", "struct_type", "members"},
    "function": {"doc", "returns", "params"},
    "field": {"name", "type", "doc", "pointer", "array", "optional", "lifetime"},
}


class SpecError(Exception):
    """The description breaks one of its rules."""


def check_keys(where: str, kind: str, entry: dict[str, Any]) -> None:
    unknown = sorted(set(entry) - KEYS[kind])
    if unknown:
        raise SpecError(f"{where}: unknown keys {unknown}")


class Api:
    """The description, checked: every name it uses is declared and every
    value it fixes is unique."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.version = spec["version"]
        self.enums: dict[str, Any] = spec.get("enums") or {}
        self.objects: dict[str, Any] = spec.get("objects") or {}
        self.handles: dict[str, Any] = spec.get("handles") or {}
        self.callbacks: dict[str, Any] = spec.get("callbacks") or {}
        self.structs: dict[str, Any] = spec.get("structs") or {}
        self.functions: dict[str, Any] = spec.get("functions") or {}
        self._check()

    def _check(self) -> None:
        for kind, entries in (
            ("enum", self.enums),
            ("object", self.objects),
            ("handle", self.handles),
            ("callback", self.callbacks),
            ("struct", self.structs),
            ("function", self.functions),
        ):
            for name, entry in entries.items():
                check_keys(name, kind, entry)
                if not entry.get("doc"):
                    raise SpecError(f"{name} needs a doc")

        names: dict[str, str] = {}
        for kind in ("enums", "objects", "handles", "callbacks", "structs", "functions"):
            for name in getattr(self, kind):
                if name in names or name in BUILTINS:
                    raise SpecError(f"{name} is declared twice")
                names[name] = kind

        for name, enum in self.enums.items():
            seen: dict[int, str] = {}
            for value_name, value in enum["values"].items():
                if not isinstance(value, dict) or "value" not in value:
                    raise SpecError(f"{name}.{value_name} needs an explicit value")
                check_keys(f"{name}.{value_name}", "value", value)
                if value["value"] in seen:
                    raise SpecError(f"{name}.{value_name} repeats the value of {seen[value['value']]}")
                seen[value["value"]] = value_name

        struct_types: dict[int, str] = {}
        for name, struct in self.structs.items():
            for base in struct.get("extends", []):
                if not self.structs.get(base, {}).get("extensible"):
                    raise SpecError(f"{name} extends {base}, which is not an extensible struct")
            if struct.get("extends"):
                struct_type = struct.get("struct_type")
                if struct_type is None:
                    raise SpecError(f"{name} extends a struct but has no struct_type")
                if struct_type in struct_types:
                    raise SpecError(f"{name} repeats the struct_type of {struct_types[struct_type]}")
                struct_types[struct_type] = name
            self._check_fields(name, struct["members"])

        for name, callback in self.callbacks.items():
            self._check_fields(name, callback["params"])
        for name, function in self.functions.items():
            self._check_fields(name, function["params"])
            returns = function.get("returns")
            if returns is not None:
                self._check_type(name, returns)

    def _check_fields(self, owner: str, fields: list[dict[str, Any]]) -> None:
        field_names = {field["name"] for field in fields}
        for field in fields:
            where = f"{owner}.{field['name']}"
            check_keys(where, "field", field)
            self._check_type(where, field["type"])
            pointer = field.get("pointer")
            if pointer is not None and pointer not in POINTERS:
                raise SpecError(f"{where}: pointer must be one of {sorted(POINTERS)}")
            count = field.get("array")
            if count is not None and count not in field_names:
                raise SpecError(f"{where}: its count {count} is not in {owner}")

    def _check_type(self, where: str, name: str) -> None:
        declared = {**self.enums, **self.objects, **self.handles, **self.callbacks, **self.structs}
        if name not in BUILTINS and name not in declared:
            raise SpecError(f"{where}: unknown type {name}")

    def struct_types(self) -> list[tuple[str, int]]:
        """The extension structs and their fixed type values, by value."""
        found = [(name, s["struct_type"]) for name, s in self.structs.items() if s.get("extends")]
        return sorted(found, key=lambda entry: entry[1])


def c_name(name: str) -> str:
    return PREFIX + name


def macro(*parts: str) -> str:
    return MACRO_PREFIX + "_".join(part.upper() for part in parts)


def comment(text: str, indent: str = "") -> list[str]:
    """text as // lines, wrapped to WIDTH."""
    text = " ".join(text.split())
    return [f"{indent}// {line}" for line in textwrap.wrap(text, WIDTH - len(indent) - 3)]


def field_doc(field: dict[str, Any]) -> str:
    """A field's doc, with what its flags say spelled out."""
    parts = [field.get("doc", "").strip()]
    if field.get("array"):
        parts.append(f"An array of {field['array']} elements.")
    if field.get("optional"):
        parts.append("May be null.")
    if field.get("lifetime"):
        parts.append(f"Valid while the {field['lifetime']} lives.")
    return " ".join(part if part.endswith(".") else part + "." for part in parts if part)


def c_type(api: Api, field: dict[str, Any]) -> str:
    """The C type of a member or param."""
    name = field["type"]
    base = BUILTINS.get(name) or c_name(name)
    if name in api.objects:
        base += "*"
    pointer = field.get("pointer")
    if pointer is None:
        return base
    if pointer == "in":
        return f"const {base}*"
    return f"{base}*"


def prototype(head: str, params: list[str], tail: str) -> list[str]:
    """head(params)tail on one line, or one param per line if that is wider than WIDTH."""
    line = f"{head}({', '.join(params)}){tail}"
    if len(line) <= WIDTH:
        return [line]
    return [f"{head}("] + [f"    {p}," for p in params[:-1]] + [f"    {params[-1]}){tail}"]


def render(api: Api) -> str:
    out: list[str] = []
    emit = out.append
    version = api.version

    emit("// Generated by tools/gen_c_api.py from api/bfr.yaml. Do not edit.")
    emit("//")
    emit("// The BFR C API. The generator lays this file out, not clang-format, and")
    emit("// the C++ lint rules do not apply to it.")
    emit("// clang-format off")
    emit("// NOLINTBEGIN")
    emit("#pragma once")
    emit("")
    emit("#include <stddef.h>")
    emit("#include <stdint.h>")
    emit("")
    emit(f"#define {macro('version', 'major')} {version['major']}")
    emit(f"#define {macro('version', 'minor')} {version['minor']}")
    emit(f"#define {macro('version', 'patch')} {version['patch']}")
    emit("")
    emit("// BFR_STATIC: linking the static library. BFR_BUILDING: building the shared one.")
    emit("#if defined(BFR_STATIC)")
    emit("#define BFR_API")
    emit("#elif defined(_WIN32) && defined(BFR_BUILDING)")
    emit("#define BFR_API __declspec(dllexport)")
    emit("#elif defined(_WIN32)")
    emit("#define BFR_API __declspec(dllimport)")
    emit("#else")
    emit('#define BFR_API __attribute__((visibility("default")))')
    emit("#endif")
    emit("")
    emit("#ifdef __cplusplus")
    emit('extern "C" {')
    emit("#endif")
    emit("")
    emit("// 32 bits on every compiler and in every binding.")
    emit("typedef uint32_t bfr_bool;")
    emit("#define BFR_FALSE 0u")
    emit("#define BFR_TRUE 1u")
    emit("")
    emit("// Text that is not null-terminated.")
    emit("typedef struct bfr_string {")
    emit("    const char* data;")
    emit("    size_t length;")
    emit("} bfr_string;")

    for name, enum in api.enums.items():
        emit("")
        out.extend(comment(enum["doc"]))
        emit(f"typedef enum {c_name(name)} {{")
        for value_name, value in enum["values"].items():
            if value.get("doc"):
                out.extend(comment(value["doc"], "    "))
            emit(f"    {macro(name, value_name)} = {value['value']},")
        emit(f"    {macro(name, 'force32')} = {FORCE32}")
        emit(f"}} {c_name(name)};")

    emit("")
    emit("// Which extension struct a bfr_chain begins.")
    emit("typedef enum bfr_struct_type {")
    for name, value in api.struct_types():
        emit(f"    {macro('struct_type', name)} = {value},")
    emit(f"    {macro('struct_type', 'force32')} = {FORCE32}")
    emit("} bfr_struct_type;")
    emit("")
    emit("// The head of every extension struct. An extensible struct's next points at one,")
    emit("// and each one's next at the following one.")
    emit("typedef struct bfr_chain {")
    emit("    const struct bfr_chain* next;")
    emit("    bfr_struct_type type;")
    emit("} bfr_chain;")

    for name, obj in api.objects.items():
        emit("")
        out.extend(comment(obj["doc"]))
        emit(f"typedef struct {c_name(name)} {c_name(name)};")

    for name, handle in api.handles.items():
        emit("")
        out.extend(comment(handle["doc"] + " A generational handle; 0 is never valid."))
        emit(f"typedef struct {c_name(name)} {{")
        emit("    uint64_t bits;")
        emit(f"}} {c_name(name)};")

    for name, callback in api.callbacks.items():
        emit("")
        out.extend(comment(callback["doc"]))
        for param in callback["params"]:
            doc = field_doc(param)
            if doc:
                out.extend(comment(f"{param['name']}: {doc}"))
        params = [f"{c_type(api, p)} {p['name']}" for p in callback["params"]]
        out.extend(prototype(f"typedef void (*{c_name(name)})", params, ";"))

    for name, struct in api.structs.items():
        emit("")
        doc = struct["doc"]
        if struct.get("extends"):
            extended = ", ".join(c_name(base) for base in struct["extends"])
            doc += f" Extends {extended}; chain.type is {macro('struct_type', name)}."
        out.extend(comment(doc))
        emit(f"typedef struct {c_name(name)} {{")
        if struct.get("extensible"):
            emit("    const bfr_chain* next;")
        if struct.get("extends"):
            emit("    bfr_chain chain;")
        for member in struct["members"]:
            member_doc = field_doc(member)
            if member_doc:
                out.extend(comment(member_doc, "    "))
            emit(f"    {c_type(api, member)} {member['name']};")
        emit(f"}} {c_name(name)};")

    for name, function in api.functions.items():
        emit("")
        out.extend(comment(function["doc"]))
        for param in function["params"]:
            doc = field_doc(param)
            if doc:
                out.extend(comment(f"{param['name']}: {doc}"))
        returns = function.get("returns")
        result = c_type(api, {"type": returns}) if returns else "void"
        params = [f"{c_type(api, p)} {p['name']}" for p in function["params"]] or ["void"]
        out.extend(prototype(f"BFR_API {result} {c_name(name)}", params, ";"))

    emit("")
    emit("#ifdef __cplusplus")
    emit("}")
    emit("#endif")
    emit("")
    emit("// NOLINTEND")
    emit("// clang-format on")
    return "\n".join(out) + "\n"


def load(path: pathlib.Path) -> Api:
    with path.open(encoding="utf-8") as handle:
        return Api(yaml.safe_load(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the header is out of date")
    parser.add_argument("--functions", action="store_true", help="list the C functions, one a line")
    args = parser.parse_args(argv)
    try:
        api = load(SOURCE)
        header = render(api)
    except SpecError as error:
        print(f"gen_c_api: {SOURCE.relative_to(REPO_ROOT)}: {error}", file=sys.stderr)
        return 1
    if args.functions:
        print("\n".join(c_name(name) for name in api.functions))
        return 0
    if args.check:
        if not HEADER.exists() or HEADER.read_text(encoding="utf-8") != header:
            print(f"gen_c_api: {HEADER.relative_to(REPO_ROOT)} is out of date", file=sys.stderr)
            print("Run: python3 tools/gen_c_api.py", file=sys.stderr)
            return 1
        return 0
    HEADER.write_text(header, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
