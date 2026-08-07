"""Every read of the `wordlevel` source goes through here, and it all goes through Griffe.

Griffe parses the AST rather than importing, so the generator never executes package code. That
also means it sees "trailing" docstrings on attribute assignments (`Foo: TypeAlias = ...` followed
by a string literal), which Python discards at runtime and `__doc__` therefore cannot report.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

import griffe


MemberKind = Literal["methods", "classes", "attributes"]

_MODULE_CACHE: dict[str, griffe.Module | None] = {}


def load_module(path: str) -> griffe.Module | None:
    """Load a module by dotted path, or return None if it is not a module. Cached per run."""
    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]

    try:
        module = griffe.load(path)
    # A failed probe means this prefix is not a module, so the caller tries a shorter one. Griffe
    # signals that with a raw KeyError or ImportError
    except (ImportError, KeyError):
        module = None

    resolved = module if isinstance(module, griffe.Module) else None
    _MODULE_CACHE[path] = resolved
    return resolved


def resolve_object(target: str) -> griffe.Object | griffe.Alias | None:
    """Resolve a dotted path to a Griffe object, or None.

    The split between module path and attribute path is unknown, so the longest module prefix is
    tried first and shortened until one loads.
    """
    parts = target.split(".")

    for split in range(len(parts), 0, -1):
        module = load_module(".".join(parts[:split]))
        if module is None:
            continue

        obj: griffe.Object | griffe.Alias = module
        try:
            for attribute in parts[split:]:
                obj = obj.members[attribute]
        except KeyError:
            continue

        return obj

    return None


def public_members(source: str, kind: MemberKind, prefix: str = "") -> list[str]:
    """Names of a class's or module's own public members of the given kind, in source order.

    Aliases are excluded, so a module reports only what it defines rather than what it imports,
    and a class reports only what it declares rather than what it inherits.
    """
    wanted: griffe.Kind
    if kind == "methods":
        wanted = griffe.Kind.FUNCTION
    elif kind == "classes":
        wanted = griffe.Kind.CLASS
    else:
        wanted = griffe.Kind.ATTRIBUTE

    return [
        name
        for name, member in _own_members(source)
        if member.kind is wanted and name.startswith(prefix)
    ]


def all_public_members(source: str) -> list[str]:
    """Names of a class's or module's own public members, of every kind, in source order."""
    return [name for name, _ in _own_members(source)]


def _own_members(source: str) -> list[tuple[str, griffe.Object | griffe.Alias]]:
    obj = resolve_object(source)
    if obj is None:
        raise LookupError(f"Could not resolve {source!r}")

    return [
        (name, member)
        for name, member in obj.members.items()
        if not name.startswith("_") and not member.is_alias
    ]


def docstring(target: str) -> str:
    """The raw docstring for a dotted path, or an empty string."""
    obj = resolve_object(target)
    if obj is None or obj.docstring is None:
        return ""

    return obj.docstring.value


def column_header(targets: Sequence[str]) -> str:
    """The left-hand column header for a summary table, inferred from what its rows document.

    Raises:
        ValueError: If the targets are of mixed kinds, which no table should be
    """
    if not targets:
        return ""

    headers = {_header_for(target) for target in targets}
    if len(headers) > 1:
        raise ValueError(f"Summary table mixes object kinds {sorted(headers)}: {list(targets)}")

    return headers.pop()


def _header_for(target: str) -> str:
    obj = resolve_object(target)
    if obj is None:
        return ""
    if obj.kind is griffe.Kind.CLASS:
        return "Class"
    if obj.kind is griffe.Kind.FUNCTION:
        return "Method"

    return "Type"


def summary(target: str) -> str:
    """The first sentence of a docstring, flattened to fit in one Markdown table cell."""
    doc = docstring(target).strip()
    if not doc:
        return ""

    return _escape_cell(_first_sentence(doc))


def _first_sentence(doc: str) -> str:
    # Try ".\n" before ". " - Google-style docstrings end the summary line with a period followed
    # by a newline (and a blank line). Trying ". " first would match an in-body period such as
    # "...one per corpus. Each..." inside the Args block, pulling the whole Args section into the
    # description.
    for separator in (".\n", ". "):
        index = doc.find(separator)
        if index != -1:
            return doc[: index + 1]

    return doc.splitlines()[0]


def _escape_cell(sentence: str) -> str:
    # Collapse internal whitespace (including newlines from wrapped first-sentence lines) so the
    # result fits in a single Markdown table cell.
    collapsed = " ".join(sentence.split())
    # Strip mkdocstrings cross-references like `[text][target]` down to just `text`. These would
    # otherwise be mangled by the bracket-escaping below, which exists to prevent residual
    # bracketed text (e.g., `list[int]`) from being parsed as reference-style links.
    dereferenced = re.sub(r"\[([^\[\]]+)\]\[[^\[\]]*\]", r"\1", collapsed)

    return dereferenced.replace("|", r"\|").replace("[", r"\[").replace("]", r"\]")
