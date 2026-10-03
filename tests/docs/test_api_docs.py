from collections import Counter
from collections.abc import Iterator
from typing import Any

import tomlkit

from docs.scripts.api_docs import structure
from docs.scripts.api_docs.introspect import _escape_cell, summary
from docs.scripts.api_docs.markdown import render
from docs.scripts.api_docs.nav import splice
from docs.scripts.api_docs.tree import (
    SUMMARY_TABLE,
    Section,
    methods,
    resolve,
    sections_per,
    snake_case,
    validate,
    walk_pages,
)


# The smallest config the splicer accepts: a nav holding the placeholder it replaces
TEMPLATE = '[project]\nnav = [\n    {"API" = []}\n]\n'


def _nav(sections: tuple[Section, ...]) -> Any:
    return tomlkit.parse(splice(TEMPLATE, sections, structure.API_DIR))["project"]["nav"]


def _nav_page_paths(node: Any) -> Iterator[str]:
    """Every page path in the nav, duplicates kept, in the order they appear."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_page_paths(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_page_paths(value)


def _nav_arrays(node: Any) -> Iterator[list[Any]]:
    """Every array in the nav, which is to say every section's list of children."""
    if isinstance(node, list):
        yield node
        for item in node:
            yield from _nav_arrays(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_arrays(value)


# =============================================================================
# The declared tree against the source
# =============================================================================


def test_every_public_member_is_documented() -> None:
    """
    Checks that no public member of a source the API tree introspects is missing a page.

    This is the one check that can fail because of in-progress work on the package rather than a
    bug in the generator: adding a public member of a shape no section claims will fail here until
    the member is given a page.
    """
    assert validate(structure.API, resolve(structure.API)) == []


def test_undocumented_member_of_a_documented_source_is_reported() -> None:
    """
    Checks that a source is answerable for all of its public members, not just the ones a prefix
    narrows a section down to
    """
    only_constructors = (
        Section(
            "Catalog",
            index=None,
            children=(
                Section(
                    "Construct",
                    index=SUMMARY_TABLE,
                    children=(methods(structure.CATALOG, prefix="from_"),),
                ),
            ),
        ),
    )

    problems = validate(only_constructors, resolve(only_constructors))

    assert [problem for problem in problems if "with_comparisons" in problem]


def test_companion_module_must_prefix_its_attributes_with_the_member_name() -> None:
    """
    Checks that a companion module breaking the `{Name}{Parameter}` convention is reported.

    `strip_prefix` silently no-ops otherwise, leaving a page titled `labMTLanguage` rather than
    `Language`, which is easy to miss in review.
    """
    # Point the lexicons at a module that is not their companion, so no attribute matches a prefix
    mismatched = (
        Section(
            "Lexicons",
            index=None,
            children=(
                sections_per(
                    structure.LEXICONS,
                    module="wordlevel.core.comparison.normalization",
                    strip_prefix=True,
                ),
            ),
        ),
    )

    problems = validate(mismatched, resolve(mismatched))

    assert [problem for problem in problems if "NormalizeBy" in problem]


# =============================================================================
# Discovery
# =============================================================================


def test_lexicon_types_are_discovered_from_the_companion_module() -> None:
    """
    Checks that a lexicon's parameter types are found without being named in the API tree.

    The path exercises all three conventions at once: the companion module is located from the
    lexicon's name, the lexicon name slugs verbatim, and the parameter name is left once the
    lexicon's prefix is stripped.
    """
    paths = {
        (directory / page.filename).as_posix()
        for directory, page in walk_pages(resolve(structure.API))
    }

    assert "configuration/lexicons/labMT/language.md" in paths


def test_loader_page_holds_only_the_methods_that_are_not_lexicons() -> None:
    """
    Checks that the loader's methods are split by what they return: each lexicon gets a page of its
    own, while the methods that manage the cache are rendered on the loader's page rather than being
    listed as lexicons
    """
    pages = {page.target: page for _, page in walk_pages(resolve(structure.API))}

    assert f"{structure.LEXICONS}.labMT" in pages
    assert f"{structure.LEXICONS}.update_cached" not in pages
    assert pages[structure.LEXICONS].members == ("update_cached",)


# =============================================================================
# Navigation
# =============================================================================


def test_nav_lists_every_generated_page_exactly_once() -> None:
    """
    Checks that the navigation and the generated files describe the same set of pages, so that no
    page is orphaned from the sidebar and no sidebar entry points at a file that was never written
    """
    tree = resolve(structure.API)
    generated = {f"{structure.API_DIR}/{path}" for path in render(tree)}

    listed = Counter(_nav_page_paths(_nav(tree)))

    assert set(listed) == generated
    assert [path for path, count in listed.items() if count > 1] == []


def test_section_index_is_the_first_entry_in_its_nav_array() -> None:
    """
    Checks the ordering Material's `navigation.indexes` depends on to make a section header the
    link to its own index page
    """
    for array in _nav_arrays(_nav(resolve(structure.API))):
        indexes = [item for item in array if isinstance(item, str) and item.endswith("index.md")]
        if indexes:
            assert array[0] == indexes[0]


def test_splice_rejects_a_config_without_the_placeholder() -> None:
    """
    Checks that a commented-out placeholder is not spliced into, which would write the navigation
    inside a comment and emit broken TOML
    """
    commented = TEMPLATE.replace('    {"API" = []}', '    # {"API" = []}')

    try:
        splice(commented, resolve(structure.API), structure.API_DIR)
    except ValueError:
        return

    raise AssertionError("Expected a commented-out placeholder to be rejected")


# =============================================================================
# Naming and summaries
# =============================================================================


def test_snake_case_slugs_the_names_that_become_filenames() -> None:
    """
    Checks the identifier shapes that become page filenames: a snake_case method, and a CamelCase
    class or type alias. `SocialSent` is the case that `slug="verbatim"` exists to avoid.
    """
    assert snake_case("from_polars") == "from_polars"
    assert snake_case("ListOfFreqs") == "list_of_freqs"
    assert snake_case("SocialSent") == "social_sent"


def test_summary_takes_the_first_sentence_across_wrapped_lines() -> None:
    """
    Checks that a summary spanning several source lines collapses into one table cell, and stops
    at the end of the sentence rather than running into the docstring body
    """
    assert summary(f"{structure.CATALOG}.from_polars") == (
        "Creates a `Catalog` of corpora from a Polars dataframe."
    )


def test_summary_strips_cross_references() -> None:
    """
    Checks that a mkdocstrings cross-reference is reduced to its text, since the bracket escaping
    that follows would otherwise mangle it
    """
    assert summary(f"{structure.COMPARISON_MODULE}.ComparisonResult") == (
        "The result of a `Comparison` when a plan is applied to corpus data."
    )


def test_summary_escapes_characters_that_would_break_a_table_row() -> None:
    """
    Checks that a pipe in a docstring is escaped rather than read as a column separator
    """
    assert _escape_cell("Sums $|x|$ per token.") == r"Sums $\|x\|$ per token."


def test_summary_is_empty_for_an_undocumented_object() -> None:
    """
    Checks that a generated type alias with no docstring yields no description, rather than
    falling back to something synthesised
    """
    assert summary("wordlevel.lexicons.types.labmt.labMTLanguage") == ""
