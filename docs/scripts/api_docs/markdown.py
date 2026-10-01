"""Turning a resolved tree into the markdown files it describes.

Every page is an mkdocstrings directive; the only variation is whether a directory's `index.md` is
a full page for one object or a table of the pages beside it.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from textwrap import dedent

from .introspect import column_header, summary
from .tree import Overview, Page, Section, walk_pages, walk_sections


PAGE_TEMPLATE = dedent(
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


OVERVIEW_TEMPLATE = dedent(
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


# A page for a class with some of its methods beneath it, set out like an overview
CLASS_PAGE_TEMPLATE = dedent(
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
            members: [{members}]
            members_order: alphabetical
            separate_signature: true
            show_signature_annotations: true
            docstring_section_style: table
            merge_init_into_class: true
            summary: true
            show_category_heading: true
    """
)


def render(sections: tuple[Section, ...]) -> dict[PurePosixPath, str]:
    """Every file a resolved tree generates, keyed by its path relative to the API directory."""
    rendered = {
        directory / "index.md": _render_index(section)
        for directory, section in walk_sections(sections)
        if section.index is not None
    }
    rendered.update(
        {directory / page.filename: _render_page(page) for directory, page in walk_pages(sections)}
    )

    return rendered


def _render_page(page: Page) -> str:
    if page.members:
        return CLASS_PAGE_TEMPLATE.format(
            title=page.title, target=page.target, members=", ".join(page.members)
        )

    return PAGE_TEMPLATE.format(title=page.title, target=page.target)


def _render_index(section: Section) -> str:
    if isinstance(section.index, Overview):
        return OVERVIEW_TEMPLATE.format(title=section.label, target=section.index.target)

    return _render_summary_table(section)


def _render_summary_table(section: Section) -> str:
    pages = [child for child in section.children if isinstance(child, Page)]
    header = column_header([page.target for page in pages])

    lines = [f"# {section.label}", "", f"| {header} | Description |", "|---|---|"]
    lines += [f"| [`{page.title}`]({page.filename}) | {summary(page.target)} |" for page in pages]

    return "\n".join(lines) + "\n"
