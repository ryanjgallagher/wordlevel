"""Generate the API reference pages and the API navigation block.

The whole pipeline is the three steps in `main`: resolve the tree declared in `structure.py`
against the source, render it to markdown, and splice its navigation into a build config. Every
filesystem write happens in this module; everything it calls is pure.

Usage:
    uv run python -m docs.scripts.api_docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

from . import structure
from .markdown import render
from .nav import splice
from .tree import resolve, validate


REPO_ROOT = Path(__file__).resolve().parents[3]
ZENSICAL_CONFIG = REPO_ROOT / "zensical.toml"
DEFAULT_OUTPUT_CONFIG = REPO_ROOT / "zensical-build.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / structure.API_DIR


def main() -> int:
    args = _parse_args()

    tree = resolve(structure.API)
    problems = validate(structure.API, tree)
    if problems:
        print("The API tree does not match the source:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    pages = render(tree)
    # Orphans go first so that a rename differing only in case succeeds. On a case-insensitive
    # filesystem the old and new names are one file, so writing first would find the content
    # already correct, skip the write, and leave the file under its old name for the orphan pass
    # to then delete.
    removed = _remove_orphans(args.output_dir, pages)
    written = _write_pages(args.output_dir, pages)
    args.output_config.write_text(splice(ZENSICAL_CONFIG.read_text(), tree, structure.API_DIR))

    print(
        f"wrote {written} file(s); removed {removed} orphan(s); "
        f"build config written to {args.output_config}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the API reference pages and the API navigation block."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to write the generated pages into. Point this at a scratch directory to "
            f"compare a run against the live tree. (default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=DEFAULT_OUTPUT_CONFIG,
        help=(
            "Path to write the populated build config. Reads zensical.toml as a template; the "
            f"source file is never modified. (default: {DEFAULT_OUTPUT_CONFIG.name})"
        ),
    )

    return parser.parse_args()


def _write_pages(output_dir: Path, pages: dict[PurePosixPath, str]) -> int:
    written = 0

    for relative, content in pages.items():
        path = output_dir / relative
        if path.exists() and path.read_text() == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written += 1

    return written


def _remove_orphans(output_dir: Path, pages: dict[PurePosixPath, str]) -> int:
    """Delete markdown the tree no longer describes, then any directory it emptied."""
    if not output_dir.exists():
        return 0

    expected = {output_dir / relative for relative in pages}

    removed = 0
    for markdown in list(output_dir.rglob("*.md")):
        if markdown not in expected:
            markdown.unlink()
            removed += 1

    for directory in sorted(output_dir.rglob("*"), key=lambda path: -len(path.parts)):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    return removed


if __name__ == "__main__":
    sys.exit(main())
