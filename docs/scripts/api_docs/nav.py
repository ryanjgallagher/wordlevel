"""Splicing the generated API navigation into the Zensical config.

The config is rewritten as text rather than through tomlkit, which would collapse the whole
navigation onto one line and drop the quoted keys. The template is only ever read.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import tomlkit

from .tree import Page, Section, slug_name


INDENT = "  "

# The template always holds an empty placeholder. Anchoring to the start of a line with nothing but
# whitespace before it means a commented-out placeholder cannot match, which would otherwise splice
# the navigation into a comment and emit broken TOML.
_PLACEHOLDER = re.compile(r'^(?P<indent>[ \t]*)\{"API" = \[\]\}', re.MULTILINE)


def splice(template: str, sections: tuple[Section, ...], api_dir: str) -> str:
    """Return the config text with the API placeholder replaced by the generated navigation.

    Raises:
        ValueError: If the placeholder is missing or ambiguous, or the result is not valid TOML
    """
    matches = list(_PLACEHOLDER.finditer(template))
    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one `{{"API" = []}}` placeholder in the Zensical config, on a line '
            f"of its own; found {len(matches)}. The generator needs it to know where the API "
            f"navigation goes."
        )

    match = matches[0]
    indent = match["indent"]
    start = match.start() + len(indent)
    spliced = template[:start] + _render(sections, api_dir, indent) + template[match.end() :]

    try:
        tomlkit.parse(spliced)
    except Exception as error:
        raise ValueError(f"Splicing the API navigation produced invalid TOML: {error}") from error

    return spliced


def _render(sections: tuple[Section, ...], api_dir: str, base_indent: str) -> str:
    lines = ['{"API" = [']

    for section in sections:
        _render_section(section, PurePosixPath(api_dir), base_indent, 1, lines)

    lines.append(f"{base_indent}]}}")

    return "\n".join(lines)


def _render_section(
    section: Section,
    parent: PurePosixPath,
    base_indent: str,
    depth: int,
    lines: list[str],
) -> None:
    directory = parent / slug_name(section.label, section.slug)
    indent = base_indent + INDENT * depth

    lines.append(f'{indent}{{"{section.label}" = [')
    # The index page comes first so Material's `navigation.indexes` binds it to the section header,
    # making the header itself the click target.
    if section.index is not None:
        lines.append(f'{indent}{INDENT}"{directory}/index.md",')

    for child in section.children:
        if isinstance(child, Section):
            _render_section(child, directory, base_indent, depth + 1, lines)
        elif isinstance(child, Page):
            lines.append(_page_line(child, directory, indent + INDENT))
        else:
            raise TypeError(
                f"{section.label!r} still holds a {type(child).__name__}; "
                f"pass the tree through resolve() before rendering it"
            )

    lines.append(f"{indent}]}},")


def _page_line(page: Page, directory: PurePosixPath, indent: str) -> str:
    path = directory / page.filename
    if page.nav_label:
        return f'{indent}{{"{page.nav_label}" = "{path}"}},'

    return f'{indent}"{path}",'
