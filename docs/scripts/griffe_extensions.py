"""Griffe extensions used during mkdocstrings collection.

`SourceLinkExtension` appends a "View source" GitHub link to each documented
function, class, and attribute. It computes the URL from the object's file
path and line number reported by Griffe.

To wire this up, list it under mkdocstrings' python handler:

    [project.plugins.mkdocstrings.handlers.python.options]
    extensions = ["docs/scripts/griffe_extensions.py:SourceLinkExtension"]
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import griffe


REPO_ROOT = Path(__file__).resolve().parents[2]
_ZENSICAL_CONFIG = REPO_ROOT / "zensical.toml"
BRANCH = "main"


def _load_repo_url() -> str | None:
    if not _ZENSICAL_CONFIG.is_file():
        return None
    with _ZENSICAL_CONFIG.open("rb") as f:
        return tomllib.load(f).get("project", {}).get("repo_url")


REPO_URL = _load_repo_url()


class SourceLinkExtension(griffe.Extension):
    """Appends a `[Source]` GitHub link to each documented object's docstring.

    The link is appended (not prepended) so it doesn't replace the docstring's
    first paragraph in mkdocstrings' `summary: true` tables. CSS then absolutely
    positions the link to the top-right of the docstring block so it visually
    sits next to the heading.
    """

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        self._append_source_link(obj)

    def _append_source_link(self, obj: griffe.Object) -> None:
        if REPO_URL is None:
            return
        if obj.is_module:
            return
        # Skip dunders. With `merge_init_into_class: true`, an __init__ link
        # would overlap the class's link at the same absolute position.
        if obj.name.startswith("_"):
            return
        filepath = obj.filepath
        lineno = obj.lineno
        if filepath is None or lineno is None:
            return
        try:
            rel = Path(filepath).resolve().relative_to(REPO_ROOT)
        except ValueError:
            return
        url = f"{REPO_URL}/blob/{BRANCH}/{rel.as_posix()}#L{lineno}"
        link = f"[Source]({url}){{ .source-link }}"
        if obj.docstring is None:
            obj.docstring = griffe.Docstring(link, parent=obj)
        else:
            obj.docstring.value = f"{obj.docstring.value.rstrip()}\n\n{link}"
