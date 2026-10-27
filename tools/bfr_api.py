"""The API description, api/bfr.yaml, loaded and checked, for every generator.

api/bfr.yaml explains the description's format. Each generator renders the
same checked description, so the C header and the wrappers cannot disagree on
what it says.
"""

from __future__ import annotations

import pathlib
import textwrap
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "api" / "bfr.yaml"

PREFIX = "bfr_"
MACRO_PREFIX = "BFR_"
FORCE32 = "0x7FFFFFFF"
WIDTH = 100

BUILTINS = {
    "bool": "bfr_bool",
    "u8": "uint8_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "i32": "int32_t",
    "f32": "float",
    "size": "size_t",
    "opaque": "void*",
    "string": "bfr_string",
}
POINTERS = {"in", "out", "inout"}
# The keys each kind of entry may have, so a typo or a split value fails.
KEYS = {
    "enum": {"doc", "values"},
    "value": {"value", "doc"},
    "object": {"doc", "destroy"},
    "handle": {"doc", "parent", "destroy"},
    "callback": {"doc", "params"},
    "struct": {"doc", "extensible", "extends", "struct_type", "members"},
    "function": {"doc", "returns", "params"},
    "field": {"name", "type", "doc", "pointer", "array", "fixed", "optional", "lifetime", "user_data"},
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
            for member in struct["members"]:
                if member["type"] in self.structs and list(self.structs).index(member["type"]) > list(
                    self.structs
                ).index(name):
                    raise SpecError(f"{name}.{member['name']}: declare {member['type']} before {name}")

        for name, handle in self.handles.items():
            if handle.get("parent") not in self.objects:
                raise SpecError(f"{name} needs a parent, an object")
        for name, obj in self.objects.items():
            self._check_destroy(name, obj, [name])
        for name, handle in self.handles.items():
            self._check_destroy(name, handle, [handle["parent"], name])

        for name, callback in self.callbacks.items():
            self._check_fields(name, callback["params"])
            self._check_not_fixed(name, callback["params"])
        for name, function in self.functions.items():
            self._check_fields(name, function["params"])
            self._check_not_fixed(name, function["params"])
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
            fixed = field.get("fixed")
            if fixed is not None and (
                not isinstance(fixed, int) or isinstance(fixed, bool) or fixed < 1 or pointer or count
            ):
                raise SpecError(f"{where}: fixed must be a positive length, without pointer or array")
            callback = field.get("user_data")
            if callback is not None and (
                field["type"] != "opaque" or callback not in {other["type"] for other in fields}
            ):
                raise SpecError(f"{where}: user data for {callback} must be opaque, beside a {callback} member")

    def _check_not_fixed(self, owner: str, fields: list[dict[str, Any]]) -> None:
        for field in fields:
            if field.get("fixed") is not None:
                raise SpecError(f"{owner}.{field['name']}: only a struct member may be fixed")

    def _check_destroy(self, name: str, entry: dict[str, Any], params: list[str]) -> None:
        """entry names the function that destroys it, which takes params' types
        in order and returns nothing."""
        destroy = self.functions.get(entry.get("destroy", ""))
        if destroy is None:
            raise SpecError(f"{name} needs destroy, the function that destroys it")
        if destroy.get("returns") is not None or [param["type"] for param in destroy["params"]] != params:
            raise SpecError(f"{name}: {entry['destroy']} must take {', '.join(params)} and return nothing")

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


def comment(text: str, indent: str = "", marker: str = "//") -> list[str]:
    """text as marker lines, // by default, wrapped to WIDTH."""
    text = " ".join(text.split())
    return [f"{indent}{marker} {line}" for line in textwrap.wrap(text, WIDTH - len(indent) - len(marker) - 1)]


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


def load(path: pathlib.Path) -> Api:
    with path.open(encoding="utf-8") as handle:
        return Api(yaml.safe_load(handle))
