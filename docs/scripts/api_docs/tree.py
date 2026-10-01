"""The node types the API tree is declared with, and the passes that resolve and check it.

A tree is *declared* in `structure.py` out of five kinds of child:

    Page         one generated markdown file, named outright
    ClassPage    expands to a Page for a class, with the methods no Discovery claims beneath it
    Discovery    expands to Pages, one per public member of a class or module
    SectionsPer  expands to Sections, one per public member of a class
    Section      a container, holding any of the above

`resolve` replaces the three expanding kinds with what they expand to, so a resolved tree holds
only Sections and Pages. `validate` reports every way the declaration and the source can disagree.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Literal

from .introspect import (
    MemberKind,
    all_public_members,
    docstring,
    public_members,
    resolve_object,
)


Slug = Literal["snake", "verbatim"]


# =============================================================================
# Nodes
# =============================================================================


@dataclass(frozen=True)
class Page:
    """One generated markdown file: an mkdocstrings stub for a single object.

    `members` are the object's methods rendered beneath it on the same page. A page without any
    documents the object alone.
    """

    target: str
    title: str
    filename: str
    nav_label: str | None = None
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Discovery:
    """Expands to one Page per public member of `source`.

    `returns` narrows the members to methods whose return annotation is written as that name.
    """

    source: str
    kind: MemberKind
    prefix: str = ""
    returns: str | None = None
    title: str = "{name}"
    slug: Slug = "snake"


@dataclass(frozen=True)
class ClassPage:
    """Expands to one Page for a class, rendering its public methods beneath it.

    The methods that `exclude` discovers are left off, since they are given pages of their own.
    """

    target: str
    title: str
    filename: str
    nav_label: str | None = None
    exclude: Discovery | None = None


@dataclass(frozen=True)
class SectionsPer:
    """Expands to one Section per public method of `source`.

    Each section's pages are the public attributes of a companion module, located by formatting
    `module` with the member's name (`{name}`) or its lowercased form (`{lower}`). `returns`
    narrows the methods to those whose return annotation is written as that name.
    """

    source: str
    module: str
    returns: str | None = None
    strip_prefix: bool = False
    slug: Slug = "snake"


@dataclass(frozen=True)
class Overview:
    """An index page that is an mkdocstrings page for one object, members and all."""

    target: str


@dataclass(frozen=True)
class SummaryTable:
    """An index page that is a table of the section's own pages."""


SUMMARY_TABLE = SummaryTable()

Index = Overview | SummaryTable


@dataclass(frozen=True)
class Section:
    """A node in the tree, rendering as a directory and (optionally) an index page."""

    label: str
    index: Index | None
    children: tuple[Section | Page | ClassPage | Discovery | SectionsPer, ...]
    slug: Slug = "snake"


# =============================================================================
# Declaration helpers
# =============================================================================


def page(target: str, *, title: str | None = None, nav_label: str | None = None) -> Page:
    """A single page, named after the object it documents unless `title` says otherwise."""
    name = target.rsplit(".", 1)[-1]

    return Page(
        target=target,
        title=title or name,
        filename=f"{snake_case(name)}.md",
        nav_label=nav_label,
    )


def class_page(
    target: str,
    *,
    title: str | None = None,
    nav_label: str | None = None,
    exclude: Discovery | None = None,
) -> ClassPage:
    """A page for a class and its methods, leaving off the methods `exclude` gives pages of."""
    name = target.rsplit(".", 1)[-1]

    return ClassPage(
        target=target,
        title=title or name,
        filename=f"{snake_case(name)}.md",
        nav_label=nav_label,
        exclude=exclude,
    )


def methods(
    source: str,
    *,
    prefix: str = "",
    returns: str | None = None,
    title: str = "{name}",
    slug: Slug = "snake",
) -> Discovery:
    """A page per public method a class declares, optionally narrowed by name or return type."""
    return Discovery(
        source=source, kind="methods", prefix=prefix, returns=returns, title=title, slug=slug
    )


def classes(source: str, *, title: str = "{name}", slug: Slug = "snake") -> Discovery:
    """A page per public class a module defines."""
    return Discovery(source=source, kind="classes", title=title, slug=slug)


def sections_per(
    source: str,
    *,
    module: str,
    returns: str | None = None,
    strip_prefix: bool = False,
    slug: Slug = "snake",
) -> SectionsPer:
    """A sub-section per public method of a class, filled from a companion module."""
    return SectionsPer(
        source=source, module=module, returns=returns, strip_prefix=strip_prefix, slug=slug
    )


_SNAKE_FIRST = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_SECOND = re.compile(r"([a-z0-9])([A-Z])")


def snake_case(name: str) -> str:
    """CamelCase to snake_case, so mixed-case identifiers read cleanly in a URL."""
    return _SNAKE_SECOND.sub(r"\1_\2", _SNAKE_FIRST.sub(r"\1_\2", name)).lower().replace(" ", "_")


def slug_name(name: str, slug: Slug) -> str:
    """The path component for a name. `verbatim` is for names that are canonical as written."""
    return name if slug == "verbatim" else snake_case(name)


# =============================================================================
# Resolution
# =============================================================================


def resolve(sections: tuple[Section, ...]) -> tuple[Section, ...]:
    """Expand every Discovery and SectionsPer, leaving a tree of only Sections and Pages."""
    return tuple(_resolve_section(section) for section in sections)


def _resolve_section(section: Section) -> Section:
    children: list[Section | Page] = []

    for child in section.children:
        if isinstance(child, Discovery):
            children.extend(_expand_discovery(child))
        elif isinstance(child, SectionsPer):
            children.extend(_expand_sections_per(child))
        elif isinstance(child, ClassPage):
            children.append(_expand_class_page(child))
        elif isinstance(child, Section):
            children.append(_resolve_section(child))
        else:
            children.append(child)

    return replace(section, children=tuple(children))


def _discovered_names(discovery: Discovery) -> list[str]:
    return public_members(discovery.source, discovery.kind, discovery.prefix, discovery.returns)


def _expand_discovery(discovery: Discovery) -> list[Page]:
    pages = [
        Page(
            target=f"{discovery.source}.{name}",
            title=discovery.title.format(name=name),
            filename=f"{slug_name(name, discovery.slug)}.md",
        )
        for name in _discovered_names(discovery)
    ]

    return sorted(pages, key=lambda page: page.title.lower())


def _expand_class_page(spec: ClassPage) -> Page:
    excluded = set(_discovered_names(spec.exclude)) if spec.exclude is not None else set()

    return Page(
        target=spec.target,
        title=spec.title,
        filename=spec.filename,
        nav_label=spec.nav_label,
        members=tuple(
            name for name in public_members(spec.target, "methods") if name not in excluded
        ),
    )


def _expand_sections_per(spec: SectionsPer) -> list[Section]:
    sections = []

    for name in public_members(spec.source, "methods", returns=spec.returns):
        module = _companion_module(spec, name)
        # The section is named for the member, so it takes the spec's slug. Its pages are named for
        # the parameters inside, which are ordinary identifiers and slug the ordinary way.
        pages = tuple(
            Page(
                target=f"{module}.{attribute}",
                title=_page_title(spec, name, attribute),
                filename=f"{snake_case(_page_title(spec, name, attribute))}.md",
            )
            for attribute in public_members(module, "attributes")
        )
        # A companion module with nothing public means a member that takes no parameters, which
        # should contribute no section rather than an empty one.
        if pages:
            sections.append(Section(label=name, index=None, children=pages, slug=spec.slug))

    return sorted(sections, key=lambda section: section.label.lower())


def _companion_module(spec: SectionsPer, name: str) -> str:
    return spec.module.format(name=name, lower=name.lower())


def _page_title(spec: SectionsPer, name: str, attribute: str) -> str:
    if not spec.strip_prefix:
        return attribute

    return attribute.removeprefix(name) or attribute


# =============================================================================
# Walking a resolved tree
# =============================================================================


# The directory the tree is rooted at, which every walk starts from
ROOT = PurePosixPath()


def walk_sections(
    sections: tuple[Section, ...], parent: PurePosixPath = ROOT
) -> Iterator[tuple[PurePosixPath, Section]]:
    """Every section in a resolved tree, paired with the directory it renders into."""
    for section in sections:
        directory = parent / slug_name(section.label, section.slug)
        yield directory, section
        yield from walk_sections(
            tuple(child for child in section.children if isinstance(child, Section)), directory
        )


def walk_pages(
    sections: tuple[Section, ...], parent: PurePosixPath = ROOT
) -> Iterator[tuple[PurePosixPath, Page]]:
    """Every page in a resolved tree, paired with the directory it renders into."""
    for directory, section in walk_sections(sections, parent):
        for child in section.children:
            if isinstance(child, Page):
                yield directory, child


def index_paths(sections: tuple[Section, ...]) -> set[PurePosixPath]:
    """The path of every index page a resolved tree generates."""
    return {
        directory / "index.md"
        for directory, section in walk_sections(sections)
        if section.index is not None
    }


# =============================================================================
# Validation
# =============================================================================


def validate(declared: tuple[Section, ...], resolved: tuple[Section, ...]) -> list[str]:
    """Every way the declaration and the source disagree, as human-readable lines."""
    return [
        *_uncovered_members(declared, resolved),
        *_unresolvable_targets(declared),
        *_colliding_paths(resolved),
        *_tables_holding_sections(resolved),
        *_broken_companion_prefixes(declared),
        *_broken_docstring_links(resolved),
    ]


def _declared_nodes(
    sections: tuple[Section, ...],
) -> Iterator[Section | Page | ClassPage | Discovery | SectionsPer]:
    for section in sections:
        yield section
        for child in section.children:
            if isinstance(child, Section):
                yield from _declared_nodes((child,))
            else:
                yield child


def _uncovered_members(declared: tuple[Section, ...], resolved: tuple[Section, ...]) -> list[str]:
    """A public member of a declared source that no page documents is almost always an oversight.

    Only `Discovery` sources are held to this. A `SectionsPer` source contributes sections rather
    than pages, and its members are covered by whichever `Discovery` also reads that source.
    """
    documented = {page.target for _, page in walk_pages(resolved)}
    documented |= {
        f"{page.target}.{member}" for _, page in walk_pages(resolved) for member in page.members
    }
    # Grouped by source rather than checked per discovery: a prefix narrows what a discovery
    # claims, not what the source owes, so `methods(Catalog, prefix="from_")` alone must still
    # answer for `with_comparisons`.
    sources: set[tuple[str, MemberKind]] = {
        (node.source, node.kind)
        for node in _declared_nodes(declared)
        if isinstance(node, Discovery)
    }

    problems = []
    for source, kind in sorted(sources):
        for name in public_members(source, kind):
            if f"{source}.{name}" not in documented:
                problems.append(
                    f"{source}.{name}: public but no page documents it. Either give it a page, "
                    f"or rename it to _{name} if it was never meant to be public."
                )

    return problems


def _unresolvable_targets(declared: tuple[Section, ...]) -> list[str]:
    """Hand-written targets are the ones a rename can silently invalidate."""
    targets = [
        node.target for node in _declared_nodes(declared) if isinstance(node, Page | ClassPage)
    ]
    targets += [
        section.index.target
        for section in _declared_nodes(declared)
        if isinstance(section, Section) and isinstance(section.index, Overview)
    ]

    return [
        f"{target}: declared in the API tree but does not exist"
        for target in targets
        if resolve_object(target) is None
    ]


def _colliding_paths(resolved: tuple[Section, ...]) -> list[str]:
    """Two pages cannot share a path. Compared case-insensitively, since macOS filesystems are."""
    seen: dict[str, tuple[PurePosixPath, str]] = {}

    problems = []
    for directory, page in walk_pages(resolved):
        path = directory / page.filename
        key = path.as_posix().lower()
        if key in seen:
            other_path, other_target = seen[key]
            problems.append(
                f"{path}: collides with {other_path} (documenting {other_target} and "
                f"{page.target}); paths that differ only in case are the same file on macOS"
            )
        seen[key] = (path, page.target)

    return problems


def _tables_holding_sections(resolved: tuple[Section, ...]) -> list[str]:
    """A summary table lists pages, so it cannot describe a section that holds sub-sections."""
    return [
        f"{directory}: has a summary-table index but also holds sub-sections"
        for directory, section in walk_sections(resolved)
        if isinstance(section.index, SummaryTable)
        and any(isinstance(child, Section) for child in section.children)
    ]


def _broken_companion_prefixes(declared: tuple[Section, ...]) -> list[str]:
    """`strip_prefix` silently no-ops when a companion module breaks the `{Name}{Parameter}` rule.

    The page would just be titled `NRCEmotion` instead of `Emotion`, which is easy to miss in
    review, so the convention is checked rather than assumed.
    """
    problems = []
    for node in _declared_nodes(declared):
        if not isinstance(node, SectionsPer) or not node.strip_prefix:
            continue
        for name in public_members(node.source, "methods", returns=node.returns):
            module = _companion_module(node, name)
            for attribute in public_members(module, "attributes"):
                if not attribute.startswith(name):
                    problems.append(
                        f"{module}.{attribute}: expected to start with {name!r} so the page can be "
                        f"titled after the parameter alone"
                    )

    return problems


_MARKDOWN_LINK = re.compile(r"\]\((?!\w+:)([^)\s]+\.md)(?:#[^)\s]*)?\)")


def _broken_docstring_links(resolved: tuple[Section, ...]) -> list[str]:
    """Docstrings carry hand-written relative links between API pages; renames break them."""
    generated = {directory / page.filename for directory, page in walk_pages(resolved)}
    generated |= index_paths(resolved)

    problems = []
    for directory, page in walk_pages(resolved):
        problems += _check_links(page.target, directory, generated)
        for member in page.members:
            problems += _check_links(f"{page.target}.{member}", directory, generated)

    for directory, section in walk_sections(resolved):
        if not isinstance(section.index, Overview):
            continue
        # An overview renders the object's own docstring plus every member's, so all of them
        # resolve their links against the section directory.
        problems += _check_links(section.index.target, directory, generated)
        for name in all_public_members(section.index.target):
            problems += _check_links(f"{section.index.target}.{name}", directory, generated)

    return problems


def _check_links(target: str, directory: PurePosixPath, generated: set[PurePosixPath]) -> list[str]:
    problems = []
    for link in _MARKDOWN_LINK.findall(docstring(target)):
        destination = PurePosixPath(posixpath.normpath(directory / link))
        if destination not in generated:
            problems.append(
                f"{target}: links to '{link}', which resolves to {destination}, "
                f"which is not a generated page"
            )

    return problems
