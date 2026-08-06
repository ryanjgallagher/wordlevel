"""What the API reference contains, in the order the sidebar shows it.

This is the file to edit when the public API grows. Everything else in the package is machinery
for turning the tree below into markdown files and a navigation block.
"""

from __future__ import annotations

from .tree import (
    SUMMARY_TABLE,
    Overview,
    Section,
    classes,
    methods,
    page,
    sections_per,
)


CATALOG = "wordlevel.api.catalog.catalog.Catalog"
PLOTTER = "wordlevel.api.catalog.plot.Plotter"
COMPARISON = "wordlevel.api.comparison.comparison.Comparison"
COMPARISON_MODULE = "wordlevel.api.comparison.comparison"
SCORER = "wordlevel.api.comparison.score.Scorer"
EXCLUDER = "wordlevel.api.comparison.exclude.ComparisonExcluder"
LEXICONS = "wordlevel.api.lexicons.lexicon.LexiconLoader"

# Where the generated tree lives, relative to the docs directory
API_DIR = "documentation/api"


API: tuple[Section, ...] = (
    Section(
        "Catalog",
        index=Overview(CATALOG),
        children=(
            Section(
                "Construct",
                index=SUMMARY_TABLE,
                children=(methods(CATALOG, prefix="from_", title="Catalog.{name}"),),
            ),
            Section(
                "Manipulate",
                index=SUMMARY_TABLE,
                children=(methods(CATALOG, prefix="with_", title="Catalog.{name}"),),
            ),
            Section(
                "Plot",
                index=SUMMARY_TABLE,
                children=(methods(PLOTTER, title="Catalog.plot.{name}"),),
            ),
        ),
    ),
    Section(
        "Comparison",
        index=Overview(COMPARISON),
        children=(
            page(f"{COMPARISON_MODULE}.comp", title="wordlevel.comp", nav_label="Construct"),
            Section(
                "Manipulate",
                index=SUMMARY_TABLE,
                children=(methods(COMPARISON, title="Comparison.{name}"),),
            ),
            Section(
                "Score",
                index=SUMMARY_TABLE,
                children=(methods(SCORER, title="Comparison.score.{name}"),),
            ),
            Section(
                "Exclude",
                index=SUMMARY_TABLE,
                children=(methods(EXCLUDER, title="Comparison.exclude.{name}"),),
            ),
            page(f"{COMPARISON_MODULE}.ComparisonResult", nav_label="Result"),
        ),
    ),
    Section(
        "Lexicons",
        index=SUMMARY_TABLE,
        children=(methods(LEXICONS, title="Lexicon.{name}", slug="verbatim"),),
    ),
    Section(
        "Configuration",
        index=None,
        children=(
            Section(
                "Plots",
                index=SUMMARY_TABLE,
                children=(classes("wordlevel.configs"),),
            ),
            # One sub-section per lexicon, holding that lexicon's parameter types. Both halves of
            # the naming are set by `lexicons/processor.py`, which writes each lexicon's types to
            # `types/{name.lower()}.py` as `{Name}{Parameter}` - so labMT gives types/labmt.py,
            # giving labMTLanguage, giving a page titled "Language".
            Section(
                "Lexicons",
                index=None,
                children=(
                    sections_per(
                        LEXICONS,
                        module="wordlevel.api.lexicons.types.{lower}",
                        strip_prefix=True,
                        slug="verbatim",
                    ),
                ),
            ),
        ),
    ),
    Section(
        "Types",
        index=SUMMARY_TABLE,
        children=(
            page("wordlevel.api.catalog.constructors.DictOfFreqs"),
            page("wordlevel.api.catalog.constructors.ListOfFreqs"),
            page("wordlevel.core.comparison.normalization.NORMALIZE_BY"),
        ),
    ),
)
