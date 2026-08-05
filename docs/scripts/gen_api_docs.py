"""Auto-generate API documentation pages and the API nav block in zensical.toml.

Discovery is rule-based:
- Each `Scope` declares a class or module to introspect, what kind of members
  count for coverage, and any names that are intentionally not documented.
- Each `Rule` says which subset of a scope's members goes to which
  (section, subsection) of the docs tree.

Ordering follows source order within each scope; section/subsection
ordering follows first appearance in `RULES`.

A coverage check fails the script if any public member of any scope is
neither matched by a rule nor explicitly excluded — that's the safety net
that prevents new public methods from silently missing from the docs.

Usage:
    uv run python docs/scripts/gen_api_docs.py
    uv run python docs/scripts/gen_api_docs.py --check    # CI mode
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Protocol

import griffe
import tomlkit


REPO_ROOT = Path(__file__).resolve().parents[2]
ZENSICAL_CONFIG = REPO_ROOT / "zensical.toml"
DEFAULT_OUTPUT_CONFIG = REPO_ROOT / "zensical-build.toml"
DOCS_DIR = REPO_ROOT / "docs"
API_RELATIVE_DIR = "documentation/api"
API_ABSOLUTE_DIR = DOCS_DIR / API_RELATIVE_DIR


# =============================================================================
# Matchers
# =============================================================================


class Matcher(Protocol):
    def matches(self, name: str) -> bool: ...


@dataclass(frozen=True)
class All:
    def matches(self, name: str) -> bool:
        return True


@dataclass(frozen=True)
class Prefix:
    prefix: str

    def matches(self, name: str) -> bool:
        return name.startswith(self.prefix)


@dataclass(frozen=True)
class Exact:
    name: str

    def matches(self, name: str) -> bool:
        return name == self.name


# =============================================================================
# Scopes and Rules — the source of truth for what gets documented
# =============================================================================


@dataclass
class Scope:
    """A class or module to introspect.

    `kind` decides what counts as a "documentable member":
        "methods"  - public methods (incl. classmethods/staticmethods)
                     of a class, defined on that class (no inheritance)
        "classes"  - public classes defined in a module
        "any"      - all explicitly enumerated names (use `only`)

    `excluded` and `excluded_prefix` skip members from coverage —
    use these for things that are public Python attributes but
    intentionally not part of the user-facing docs.

    For "any" kind, `only` enumerates the names to consider (because
    distinguishing module-defined names from imported ones is unreliable
    for type aliases and similar).
    """

    path: str
    kind: str
    excluded: frozenset[str] = field(default_factory=frozenset)
    excluded_prefix: str | None = None
    only: frozenset[str] | None = None


@dataclass
class Rule:
    scope: str  # must match some Scope.path
    match: Matcher
    section: str
    subsection: str | None  # None means leaves attached directly under section
    group: str | None = None  # third nesting level inside a subsection; when set,
    # the group name is stripped as a prefix from member names to produce leaf
    # titles and filenames (e.g. group="labMT" turns labMTLanguage → "Language").
    title_template: str = "{name}"
    title_for: dict[str, str] = field(default_factory=dict)  # name -> display title
    filename_for: dict[str, str] = field(default_factory=dict)  # name -> file
    nav_label_for: dict[str, str] = field(default_factory=dict)  # name -> sidebar label


SCOPES: tuple[Scope, ...] = (
    Scope(
        path="wordlevel.api.catalog.catalog.Catalog",
        kind="methods",
    ),
    Scope(path="wordlevel.api.catalog.plot.Plotter", kind="methods"),
    Scope(path="wordlevel.api.comparison.comparison.Comparison", kind="methods"),
    Scope(path="wordlevel.api.comparison.score.Scorer", kind="methods"),
    Scope(path="wordlevel.api.comparison.exclude.ComparisonExcluder", kind="methods"),
    Scope(
        path="wordlevel.api.comparison.comparison",
        kind="any",
        only=frozenset({"comp", "ComparisonResult"}),
    ),
    Scope(path="wordlevel.api.lexicons.lexicon.LexiconLoader", kind="methods"),
    Scope(path="wordlevel.configs", kind="classes"),
    Scope(
        path="wordlevel.api.lexicons.types.labmt",
        kind="any",
        only=frozenset({"labMTLanguage"}),
    ),
    Scope(
        path="wordlevel.api.lexicons.types.socialsent",
        kind="any",
        only=frozenset({"SocialSentYear"}),
    ),
    Scope(
        path="wordlevel.api.catalog.constructors",
        kind="any",
        only=frozenset({"ListOfFreqs", "DictOfFreqs"}),
    ),
    Scope(
        path="wordlevel.core.comparison.normalization",
        kind="any",
        only=frozenset({"NORMALIZE_BY"}),
    ),
)


RULES: tuple[Rule, ...] = (
    Rule(
        scope="wordlevel.api.catalog.catalog.Catalog",
        match=Prefix("from_"),
        section="Catalog",
        subsection="Construct",
        title_template="Catalog.{name}",
    ),
    Rule(
        scope="wordlevel.api.catalog.catalog.Catalog",
        match=Prefix("with_"),
        section="Catalog",
        subsection="Manipulate",
        title_template="Catalog.{name}",
    ),
    Rule(
        scope="wordlevel.api.catalog.plot.Plotter",
        match=All(),
        section="Catalog",
        subsection="Plot",
        title_template="Catalog.plot.{name}",
    ),
    Rule(
        scope="wordlevel.api.comparison.comparison",
        match=Exact("comp"),
        section="Comparison",
        subsection=None,
        title_template="wordlevel.{name}",
        nav_label_for={"comp": "Construct"},
    ),
    Rule(
        scope="wordlevel.api.comparison.comparison.Comparison",
        match=All(),
        section="Comparison",
        subsection="Manipulate",
        title_template="Comparison.{name}",
    ),
    Rule(
        scope="wordlevel.api.comparison.score.Scorer",
        match=All(),
        section="Comparison",
        subsection="Score",
        title_template="Comparison.score.{name}",
    ),
    Rule(
        scope="wordlevel.api.comparison.exclude.ComparisonExcluder",
        match=All(),
        section="Comparison",
        subsection="Exclude",
        title_template="Comparison.exclude.{name}",
    ),
    Rule(
        scope="wordlevel.api.comparison.comparison",
        match=Exact("ComparisonResult"),
        section="Comparison",
        subsection=None,
        nav_label_for={"ComparisonResult": "Result"},
    ),
    Rule(
        scope="wordlevel.api.lexicons.lexicon.LexiconLoader",
        match=All(),
        section="Lexicons",
        subsection=None,
        title_template="Lexicon.{name}",
        filename_for={"labMT": "labmt.md", "SocialSent": "socialsent.md"},
    ),
    Rule(
        scope="wordlevel.configs",
        match=All(),
        section="Configuration",
        subsection="Plots",
    ),
    Rule(
        scope="wordlevel.api.lexicons.types.labmt",
        match=All(),
        section="Configuration",
        subsection="Lexicons",
        group="labMT",
    ),
    Rule(
        scope="wordlevel.api.lexicons.types.socialsent",
        match=All(),
        section="Configuration",
        subsection="Lexicons",
        group="SocialSent",
    ),
    Rule(
        scope="wordlevel.api.catalog.constructors",
        match=All(),
        section="Types",
        subsection=None,
    ),
    Rule(
        scope="wordlevel.core.comparison.normalization",
        match=All(),
        section="Types",
        subsection=None,
    ),
)


# =============================================================================
# Discovery
# =============================================================================


@dataclass(frozen=True)
class Leaf:
    title: str
    target: str  # mkdocstrings dotted path
    filename: str
    nav_label: str | None = None  # if set, overrides the sidebar label


def _resolve(dotted: str) -> object:
    parts = dotted.split(".")
    last_err: Exception | None = None
    for i in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError as exc:
            last_err = exc
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Could not resolve {dotted!r}") from last_err


def _public_member_names(scope: Scope) -> list[str]:
    obj = _resolve(scope.path)

    if scope.kind == "methods":
        if not inspect.isclass(obj):
            raise TypeError(f"Scope {scope.path!r}: kind='methods' but not a class")
        names = [
            name
            for name, val in vars(obj).items()
            if not name.startswith("_")
            and (inspect.isfunction(val) or isinstance(val, (classmethod, staticmethod)))
        ]
    elif scope.kind == "classes":
        if not inspect.ismodule(obj):
            raise TypeError(f"Scope {scope.path!r}: kind='classes' but not a module")
        names = [
            name
            for name, val in vars(obj).items()
            if not name.startswith("_")
            and inspect.isclass(val)
            and getattr(val, "__module__", None) == scope.path
        ]
    elif scope.kind == "any":
        if not inspect.ismodule(obj):
            raise TypeError(f"Scope {scope.path!r}: kind='any' but not a module")
        if scope.only is None:
            raise ValueError(f"Scope {scope.path!r}: kind='any' requires `only` to enumerate names")
        module_attrs = set(vars(obj))
        missing = scope.only - module_attrs
        if missing:
            raise ValueError(
                f"Scope {scope.path!r}: `only` references unknown names: {sorted(missing)}"
            )
        names = [name for name in scope.only if not name.startswith("_")]
    else:
        raise ValueError(f"Unknown scope kind: {scope.kind!r}")

    if scope.excluded_prefix:
        names = [n for n in names if not n.startswith(scope.excluded_prefix)]
    excluded = set(scope.excluded)
    return [n for n in names if n not in excluded]


@dataclass
class Bucket:
    """One rule's worth of leaves, attached to a section / subsection / group.

    `subsection is None` means the leaves attach directly under the section.
    `group is None` means the leaves attach directly under the subsection.
    Buckets are ordered by rule declaration; multiple rules targeting the same
    (section, subsection, group) triple are merged.
    """

    section: str
    subsection: str | None
    group: str | None
    leaves: list[Leaf]


def discover() -> tuple[list[Bucket], list[str]]:
    """Apply rules to scopes; return (buckets, errors).

    Buckets are returned in rule declaration order. Leaves within each bucket
    are sorted alphabetically (case-insensitive). `errors` is a list of
    human-readable coverage failures.
    """
    by_scope: dict[str, Scope] = {s.path: s for s in SCOPES}
    for rule in RULES:
        if rule.scope not in by_scope:
            raise ValueError(f"Rule references unknown scope {rule.scope!r}; add it to SCOPES.")

    buckets: list[Bucket] = []
    covered: dict[str, set[str]] = {s.path: set() for s in SCOPES}

    for rule in RULES:
        scope = by_scope[rule.scope]
        rule_leaves: list[Leaf] = []
        for name in _public_member_names(scope):
            if not rule.match.matches(name):
                continue
            covered[scope.path].add(name)
            # When a rule has a `group`, the convention is that member names are
            # `{group}{Suffix}` — strip the group prefix to get a clean leaf title
            # and filename (e.g. labMTLanguage → "Language" / language.md).
            if rule.group is not None:
                stripped = name.removeprefix(rule.group) or name
                default_title = stripped
                default_filename = _snake_case(stripped) + ".md"
            else:
                default_title = rule.title_template.format(name=name)
                default_filename = _snake_case(name) + ".md"
            title = rule.title_for.get(name, default_title)
            filename = rule.filename_for.get(name, default_filename)
            target = f"{scope.path}.{name}"
            nav_label = rule.nav_label_for.get(name)
            rule_leaves.append(
                Leaf(title=title, target=target, filename=filename, nav_label=nav_label)
            )
        if not rule_leaves:
            continue

        # Merge into an existing bucket only for named subsections — direct-leaf
        # rules (subsection=None) each get their own bucket so that interleaved
        # ordering with subsections in the same section is preserved. The merge
        # key includes `group` so each lexicon (etc.) keeps its own bucket.
        if rule.subsection is not None:
            existing = next(
                (
                    b
                    for b in buckets
                    if b.section == rule.section
                    and b.subsection == rule.subsection
                    and b.group == rule.group
                ),
                None,
            )
            if existing is not None:
                existing.leaves.extend(rule_leaves)
                continue
        buckets.append(
            Bucket(
                section=rule.section,
                subsection=rule.subsection,
                group=rule.group,
                leaves=rule_leaves,
            )
        )

    for b in buckets:
        b.leaves.sort(key=lambda leaf: leaf.title.lower())

    errors: list[str] = []
    for scope in SCOPES:
        for name in _public_member_names(scope):
            if name not in covered[scope.path]:
                kind_word = {"methods": "method", "classes": "class", "any": "name"}[scope.kind]
                errors.append(
                    f"  {scope.path}.{name}: public {kind_word} but no rule "
                    f"matched. Either rename to _{name}, add `{name}` to the "
                    f"Scope's `excluded`, or add a Rule that matches it."
                )

    return buckets, errors


_SNAKE_FIRST = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_SECOND = re.compile(r"([a-z0-9])([A-Z])")


def _snake_case(name: str) -> str:
    return _SNAKE_SECOND.sub(r"\1_\2", _SNAKE_FIRST.sub(r"\1_\2", name)).lower()


# =============================================================================
# Output: markdown files
# =============================================================================


LEAF_TEMPLATE = dedent(
    """\
    ---
    title: {title}
    ---

    ::: {target}
        options:
            heading: {title}
            heading_level: 2
            show_root_heading: true
            show_root_full_path: false
            parameter_headings: true
            separate_signature: true
            show_signature_annotations: true
            docstring_section_style: table
            members: false
    """
)


CLASS_OVERVIEW_TEMPLATE = dedent(
    """\
    ---
    title: {title}
    ---

    ::: {target}
        options:
            heading: {title}
            heading_level: 1
            show_root_heading: true
            show_root_full_path: false
            members_order: alphabetical
            separate_signature: true
            show_signature_annotations: true
            docstring_section_style: table
            merge_init_into_class: true
            summary: true
            show_category_heading: true
    """
)


# Sections whose index.md is a class-level mkdocstrings overview rather than
# a table of subsections. Click-through from the nav lands on the class page
# (Material's `navigation.indexes` feature wires this up).
SECTION_DOCS: dict[str, str] = {
    "Catalog": "wordlevel.api.catalog.catalog.Catalog",
    "Comparison": "wordlevel.api.comparison.comparison.Comparison",
}

# Sections that should not have an index.md at all — the sidebar header just
# expands/collapses its children instead of linking to an overview page.
SECTIONS_WITHOUT_INDEX: frozenset[str] = frozenset({"Configuration"})

# Subsections without an index.md, keyed by (section, subsection). Groups
# (the third nesting level) never have an index.
SUBSECTIONS_WITHOUT_INDEX: frozenset[tuple[str, str]] = frozenset({("Configuration", "Lexicons")})


_griffe_module_cache: dict[str, griffe.Module] = {}


def _griffe_load(module_path: str) -> griffe.Module | None:
    """Load a module via griffe, cached per script run."""
    if module_path in _griffe_module_cache:
        return _griffe_module_cache[module_path]
    try:
        mod = griffe.load(module_path)
    # A failed probe means this prefix is not a module, so the caller tries a shorter one. griffe
    # signals that with a raw KeyError or ImportError
    except (ImportError, KeyError):
        return None
    if not isinstance(mod, griffe.Module):
        return None
    _griffe_module_cache[module_path] = mod
    return mod


def _griffe_docstring(target: str) -> str:
    """Read the docstring for `target` (a dotted path) via griffe.

    Unlike `obj.__doc__` at runtime, griffe parses the AST and picks up
    "trailing" docstrings on attribute assignments (e.g., on `TypeAlias`
    aliases), which Python discards as module-level string literals.
    """
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        mod = _griffe_load(".".join(parts[:i]))
        if mod is None:
            continue
        obj: griffe.Object = mod
        try:
            for attr in parts[i:]:
                obj = obj.members[attr]  # type: ignore[assignment]
        except KeyError:
            continue
        if obj.docstring is None:
            return ""
        return obj.docstring.value
    return ""


def _first_sentence(target: str) -> str:
    doc = _griffe_docstring(target)
    if not doc:
        # Fallback to runtime __doc__ for anything griffe couldn't resolve.
        try:
            obj = _resolve(target)
        except (ImportError, AttributeError):
            return ""
        doc = getattr(obj, "__doc__", None) or ""
    if not doc:
        return ""
    cleaned = doc.strip()
    if not cleaned:
        return ""
    # Dataclasses auto-fill __doc__ with "ClassName(field: type, ...)" when no
    # real docstring is given — skip that, it's not a real description.
    obj_name = target.rsplit(".", 1)[-1]
    if cleaned.startswith(f"{obj_name}("):
        return ""
    sentence = cleaned
    # Try ".\n" before ". " — Google-style docstrings end the summary line with a period
    # followed by a newline (and a blank line). Trying ". " first would match an in-body period
    # such as "...one per corpus. Each..." inside the Args block, pulling the whole Args section
    # into the description.
    for sep in (".\n", ". "):
        idx = cleaned.find(sep)
        if idx != -1:
            sentence = cleaned[: idx + 1]
            break
    else:
        sentence = cleaned.splitlines()[0]
    # Collapse internal whitespace (including newlines from wrapped first-sentence lines) so the
    # result fits in a single Markdown table cell.
    sentence = " ".join(sentence.split())
    # Strip mkdocstrings cross-references like `[text][target]` down to just `text`. These would
    # otherwise be mangled by the bracket-escaping below, which exists to prevent residual
    # bracketed text (e.g., `list[int]`) from being parsed as reference-style links.
    sentence = re.sub(r"\[([^\[\]]+)\]\[[^\[\]]*\]", r"\1", sentence)
    return sentence.replace("|", r"\|").replace("[", r"\[").replace("]", r"\]")


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def _slug(name: str) -> str:
    """Slug for directory names — snake_case so CamelCase reads cleanly in URLs."""
    return _snake_case(name).replace(" ", "_")


def _leaf_index_md(name: str, leaves: list[Leaf]) -> str:
    header = _leaf_column_header(leaves)
    lines = [f"# {name}", "", f"| {header} | Description |", "|---|---|"]
    for leaf in leaves:
        desc = _first_sentence(leaf.target)
        lines.append(f"| [`{leaf.title}`]({leaf.filename}) | {desc} |")
    return "\n".join(lines) + "\n"


def _leaf_column_header(leaves: list[Leaf]) -> str:
    """Infer a column header for the leaf index from what the leaves resolve to."""
    if not leaves:
        return ""
    try:
        sample = _resolve(leaves[0].target)
    except (ImportError, AttributeError):
        return ""
    if inspect.isclass(sample):
        return "Class"
    if inspect.isroutine(sample):
        # Functions, methods, classmethods, staticmethods — not type aliases (which are callable
        # but not "routine" per inspect's classification).
        return "Method"
    return "Type"


def _section_index_md(name: str, subsections: list[str]) -> str:
    lines = [f"# {name}", "", "| | |", "|---|---|"]
    for sub in subsections:
        lines.append(f"| [{sub}]({_slug(sub)}/index.md) | |")
    return "\n".join(lines) + "\n"


def _sections_in_order(buckets: list[Bucket]) -> dict[str, list[Bucket]]:
    """Group buckets by section, preserving declaration order."""
    result: dict[str, list[Bucket]] = {}
    for b in buckets:
        result.setdefault(b.section, []).append(b)
    return result


def write_files(buckets: list[Bucket]) -> tuple[set[Path], int]:
    expected: set[Path] = set()
    written = 0

    for section, section_buckets in _sections_in_order(buckets).items():
        section_dir = API_ABSOLUTE_DIR / _slug(section)

        has_direct = any(b.subsection is None for b in section_buckets)
        has_subsections = any(b.subsection is not None for b in section_buckets)

        if section not in SECTIONS_WITHOUT_INDEX:
            section_index_path = section_dir / "index.md"
            expected.add(section_index_path)
            if section in SECTION_DOCS:
                index_content = CLASS_OVERVIEW_TEMPLATE.format(
                    title=section, target=SECTION_DOCS[section]
                )
            elif has_direct and not has_subsections:
                all_direct = [leaf for b in section_buckets for leaf in b.leaves]
                index_content = _leaf_index_md(section, all_direct)
            elif has_subsections and not has_direct:
                sub_names = _ordered_unique(
                    b.subsection for b in section_buckets if b.subsection is not None
                )
                index_content = _section_index_md(section, sub_names)
            else:
                raise ValueError(
                    f"Section {section!r} mixes direct-leaf and subsection rules but "
                    f"is not in SECTION_DOCS; cannot generate a section index."
                )
            if _write_if_changed(section_index_path, index_content):
                written += 1

        emitted_subs: set[str] = set()
        for b in section_buckets:
            if b.subsection is None:
                for leaf in b.leaves:
                    leaf_path = section_dir / leaf.filename
                    expected.add(leaf_path)
                    content = LEAF_TEMPLATE.format(title=leaf.title, target=leaf.target)
                    if _write_if_changed(leaf_path, content):
                        written += 1
                continue
            if b.subsection in emitted_subs:
                continue
            emitted_subs.add(b.subsection)

            sub_buckets = [x for x in section_buckets if x.subsection == b.subsection]
            sub_dir = section_dir / _slug(b.subsection)
            direct_sub_leaves = [leaf for x in sub_buckets if x.group is None for leaf in x.leaves]
            grouped_sub_buckets = [x for x in sub_buckets if x.group is not None]

            if (section, b.subsection) not in SUBSECTIONS_WITHOUT_INDEX:
                sub_index_path = sub_dir / "index.md"
                expected.add(sub_index_path)
                if direct_sub_leaves and not grouped_sub_buckets:
                    sub_index_content = _leaf_index_md(b.subsection, direct_sub_leaves)
                elif grouped_sub_buckets and not direct_sub_leaves:
                    group_names = _ordered_unique(
                        x.group for x in grouped_sub_buckets if x.group is not None
                    )
                    sub_index_content = _section_index_md(b.subsection, group_names)
                else:
                    raise ValueError(
                        f"Subsection {b.subsection!r} of {section!r} mixes direct and "
                        f"grouped leaves; cannot generate a subsection index."
                    )
                if _write_if_changed(sub_index_path, sub_index_content):
                    written += 1

            for leaf in direct_sub_leaves:
                leaf_path = sub_dir / leaf.filename
                expected.add(leaf_path)
                content = LEAF_TEMPLATE.format(title=leaf.title, target=leaf.target)
                if _write_if_changed(leaf_path, content):
                    written += 1

            for gb in grouped_sub_buckets:
                if gb.group is None:
                    continue
                group_dir = sub_dir / _slug(gb.group)
                for leaf in gb.leaves:
                    leaf_path = group_dir / leaf.filename
                    expected.add(leaf_path)
                    content = LEAF_TEMPLATE.format(title=leaf.title, target=leaf.target)
                    if _write_if_changed(leaf_path, content):
                        written += 1

    return expected, written


def _ordered_unique(items: Iterable[str]) -> list[str]:
    """Return the unique items in first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def remove_orphans(expected: set[Path]) -> int:
    if not API_ABSOLUTE_DIR.exists():
        return 0
    removed = 0
    for md in list(API_ABSOLUTE_DIR.rglob("*.md")):
        if md not in expected:
            md.unlink()
            removed += 1
    # Remove empty directories, deepest first
    dirs = sorted(
        (p for p in API_ABSOLUTE_DIR.rglob("*") if p.is_dir()),
        key=lambda p: -len(p.parts),
    )
    for d in dirs:
        if not any(d.iterdir()):
            d.rmdir()
    return removed


# =============================================================================
# Output: zensical.toml nav rewrite via tomlkit
# =============================================================================


_API_OPENER = re.compile(r'\{\s*"?API"?\s*=\s*\[')


def _find_api_span(text: str) -> tuple[int, int]:
    """Locate the span `{"API" = [...]}` in text by bracket matching.

    Returns (start, end_exclusive). Raises if not found or unbalanced.
    """
    match = _API_OPENER.search(text)
    if not match:
        raise ValueError(
            'Could not find {"API" = [ entry in zensical.toml. '
            "The auto-generator expects this entry to exist."
        )
    start = match.start()
    pos = match.end()  # just after the opening `[`
    depth = 1  # we're already inside the `[` from the match
    i = pos
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while j < len(text):
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            i = j + 1
        elif c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
        elif c == "[":
            depth += 1
            i += 1
        elif c == "]":
            depth -= 1
            i += 1
            if depth == 0:
                if i < len(text) and text[i] == "}":
                    return (start, i + 1)
                raise ValueError(f"Found closing ] at {i - 1} but no matching }} in zensical.toml")
        else:
            i += 1
    raise ValueError("Reached end of zensical.toml without closing the API array")


def _format_api_block(buckets: list[Bucket], base_indent: str) -> str:
    """Render the {"API" = [...]} block with consistent indentation.

    `base_indent` is the leading whitespace on the line where the opening
    `{"API"` appears. Subsequent lines use that as their baseline plus 2
    spaces per nesting level. Buckets are emitted in declaration order, so
    direct leaves and subsections can interleave freely within a section.
    """
    INDENT = "  "
    lines: list[str] = ['{"API" = [']

    for section, section_buckets in _sections_in_order(buckets).items():
        section_slug = _slug(section)

        lines.append(f'{base_indent}{INDENT}{{"{section}" = [')
        # Index page first — Material's `navigation.indexes` feature binds
        # this to the section header so it becomes the click target. Sections
        # in SECTIONS_WITHOUT_INDEX skip this; the header is non-clickable
        # and just toggles its children.
        if section not in SECTIONS_WITHOUT_INDEX:
            lines.append(f'{base_indent}{INDENT * 2}"{API_RELATIVE_DIR}/{section_slug}/index.md",')

        emitted_subs: set[str] = set()
        for b in section_buckets:
            if b.subsection is None:
                for leaf in b.leaves:
                    path = f"{API_RELATIVE_DIR}/{section_slug}/{leaf.filename}"
                    if leaf.nav_label:
                        lines.append(f'{base_indent}{INDENT * 2}{{"{leaf.nav_label}" = "{path}"}},')
                    else:
                        lines.append(f'{base_indent}{INDENT * 2}"{path}",')
                continue
            if b.subsection in emitted_subs:
                continue
            emitted_subs.add(b.subsection)

            sub_buckets = [x for x in section_buckets if x.subsection == b.subsection]
            sub_slug = _slug(b.subsection)
            sub_path_prefix = f"{API_RELATIVE_DIR}/{section_slug}/{sub_slug}"

            lines.append(f'{base_indent}{INDENT * 2}{{"{b.subsection}" = [')
            if (section, b.subsection) not in SUBSECTIONS_WITHOUT_INDEX:
                lines.append(f'{base_indent}{INDENT * 3}"{sub_path_prefix}/index.md",')

            for x in sub_buckets:
                if x.group is None:
                    for leaf in x.leaves:
                        path = f"{sub_path_prefix}/{leaf.filename}"
                        if leaf.nav_label:
                            lines.append(
                                f'{base_indent}{INDENT * 3}{{"{leaf.nav_label}" = "{path}"}},'
                            )
                        else:
                            lines.append(f'{base_indent}{INDENT * 3}"{path}",')
                else:
                    group_slug = _slug(x.group)
                    lines.append(f'{base_indent}{INDENT * 3}{{"{x.group}" = [')
                    for leaf in x.leaves:
                        path = f"{sub_path_prefix}/{group_slug}/{leaf.filename}"
                        if leaf.nav_label:
                            lines.append(
                                f'{base_indent}{INDENT * 4}{{"{leaf.nav_label}" = "{path}"}},'
                            )
                        else:
                            lines.append(f'{base_indent}{INDENT * 4}"{path}",')
                    lines.append(f"{base_indent}{INDENT * 3}]}},")

            lines.append(f"{base_indent}{INDENT * 2}]}},")
        lines.append(f"{base_indent}{INDENT}]}},")

    lines.append(f"{base_indent}]}}")
    return "\n".join(lines)


def update_nav(buckets: list[Bucket], output_path: Path) -> None:
    """Read `zensical.toml`, populate the API subtree, and write to `output_path`.

    Validates the structure with tomlkit (so a malformed nav fails early)
    then surgically replaces the `{"API" = [...]}` span in the source text
    so we control formatting precisely. The committed `zensical.toml` is read
    only — it's never mutated.
    """
    original_text = ZENSICAL_CONFIG.read_text()

    # Sanity check: parse must succeed and the expected nav shape must exist
    doc = tomlkit.parse(original_text)
    nav = doc["project"]["nav"]  # type: ignore[index]
    found = False
    for entry in nav:  # type: ignore[union-attr]
        if isinstance(entry, dict) and "Documentation" in entry:
            for child in entry["Documentation"]:
                if isinstance(child, dict) and "API" in child:
                    found = True
                    break
            if found:
                break
    if not found:
        raise ValueError(
            "zensical.toml nav must contain a {Documentation = [{API = [...]}]} entry."
        )

    start, end = _find_api_span(original_text)
    line_start = original_text.rfind("\n", 0, start) + 1
    base_indent = original_text[line_start:start]
    if base_indent.strip():
        raise ValueError(
            f'Expected only whitespace before {{"API" but found {base_indent!r} on the same line.'
        )

    new_block = _format_api_block(buckets, base_indent)
    new_text = original_text[:start] + new_block + original_text[end:]

    output_path.write_text(new_text)


# =============================================================================
# Entry point
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-config",
        type=Path,
        default=DEFAULT_OUTPUT_CONFIG,
        help=(
            "Path to write the populated build config. Reads zensical.toml as a "
            "template; the source file is never modified. "
            f"(default: {DEFAULT_OUTPUT_CONFIG.name})"
        ),
    )
    args = parser.parse_args()

    buckets, errors = discover()

    if errors:
        print("Coverage check failed — uncovered public members:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    expected, n_written = write_files(buckets)
    n_removed = remove_orphans(expected)
    update_nav(buckets, args.output_config)

    print(
        f"wrote {n_written} file(s); removed {n_removed} orphan(s); "
        f"build config written to {args.output_config}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
