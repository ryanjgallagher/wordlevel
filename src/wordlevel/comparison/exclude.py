from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Literal, get_args

import polars as pl

from wordlevel.comparison.core.exclude import (
    ExcludeCorpus,
    exclude_by_range,
    exclude_tokens,
)
from wordlevel.constants import (
    FREQ_CMP_COL,
    FREQ_REF_COL,
    LEXICON_SCORE_CMP_COL,
    LEXICON_SCORE_REF_COL,
    NORMED_FREQ_CMP_COL,
    NORMED_FREQ_REF_COL,
)


if TYPE_CHECKING:
    from wordlevel.comparison.comparison import Comparison


EXCLUSION_RANGE_KIND = Literal["lexicon_score", "frequency", "proportion"]


class ComparisonExcluder:
    """
    Namespace for excluding tokens from comparisons.
    """

    def __init__(self, comparison: Comparison):
        """ """
        self._comparison = comparison

    def tokens(self, tokens: Collection[str]) -> Comparison:
        """
        Removes the given tokens from the reference and comparison corpora.

        Tokens are excluded from both corpora, before they are scored, so the frequencies the
        scoring measure works with are normalized over the remaining vocabulary.

        Example:
            ```python
            --8<-- "snippets/comparison/exclude.py:setup"

            --8<-- "snippets/comparison/exclude.py:tokens"
            ```

        Scoring a comparison after excluding gives the same result as excluding after scoring.
        Excluding more than once removes the union of the tokens, and a token that is not in the
        vocabulary is ignored.

        Warning:
            Excluding tokens changes the normalized frequencies of every token that remains.

        Args:
            tokens:
                The tokens to remove from both corpora

        Returns:
            A new `Comparison` whose vocabulary excludes the given tokens
        """
        state = self._comparison._state

        def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
            return exclude_tokens(df, token_col, tokens)

        excluded_tokens = state.exclusions.get("tokens", set()) | set(tokens)

        return self._comparison._update_state(
            exclusion_plan=state.compose_plans(state.exclusion_plan, plan),
            exclusions={**state.exclusions, "tokens": excluded_tokens},
        )

    def unshared_tokens(self) -> Comparison:
        """
        Removes the tokens that do not appear in both of the corpora being compared.

        Tokens are excluded from both corpora, before they are scored, so the frequencies the
        scoring measure works with are normalized over the remaining vocabulary.

        Example:
            ```python
            --8<-- "snippets/comparison/exclude.py:setup"

            --8<-- "snippets/comparison/exclude.py:unshared_tokens"
            ```

        Warning:
            Excluding tokens changes the normalized frequencies of every token that remains.

            A token that only one corpus uses can be a stark difference between them. Prefer using a
            measure that can handle these absences, rather than excluding them entirely.

        Returns:
            A new `Comparison` whose vocabulary is the one the two corpora share
        """
        state = self._comparison._state

        def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
            # A token that a corpus does not use has a frequency of zero in it, not a missing one
            return exclude_by_range(
                df,
                value_col_ref=FREQ_REF_COL,
                value_col_cmp=FREQ_CMP_COL,
                less_than=0,
                corpus="either",
            )

        return self._comparison._update_state(
            exclusion_plan=state.compose_plans(state.exclusion_plan, plan),
            exclusions={**state.exclusions, "unshared_tokens": True},
        )

    def by_frequency(
        self,
        less_than: float | None = None,
        greater_than: float | None = None,
        between: tuple[float, float] | None = None,
        inclusive: bool = True,
        corpus: ExcludeCorpus = "both",
    ) -> Comparison:
        """
        Removes tokens whose frequency falls in the given range from the reference and comparison
        corpora.

        The range is tested against the frequencies the catalog was built from, so its bounds are
        counts. Tokens are excluded from both corpora, before they are scored, so the frequencies
        the scoring measure works with are normalized over the remaining vocabulary.

        Example:
            ```python
            --8<-- "snippets/comparison/exclude.py:setup"

            --8<-- "snippets/comparison/exclude.py:by_frequency"
            ```

        Exactly one of `less_than`, `greater_than`, and `between` gives the range to exclude. To
        exclude by multiple disjoint criteria, call the method multiple times with different
        parameters. A count means different things in corpora of different sizes, so use
        `by_proportion` to exclude by a share of each corpus instead.

        Warning:
            Excluding tokens changes the normalized frequencies of every token that remains.

            A token missing from one of the corpora has a frequency of zero, so `corpus="either"`
            paired with `less_than` removes every token that the two corpora do not share.

        Args:
            less_than:
                Exclude tokens with a frequency below this bound
            greater_than:
                Exclude tokens with a frequency above this bound
            between:
                Exclude tokens with frequencies between the lower and upper bound
            inclusive:
                Whether the bounds themselves are part of the range being excluded
            corpus:
                Which corpus's frequency the range is evaluated against. `"reference"` and
                `"comparison"` evaluate against a single corpus; `"both"` excludes a token only
                when the range holds in both corpora; `"either"` excludes a token when the range
                holds in either one

        Returns:
            A new `Comparison` whose vocabulary excludes tokens with frequencies in the given range

        Raises:
            ValueError:
                If none of `less_than`, `greater_than`, and `between` is given, or if more than one
                of them is.
            ValueError: If the lower bound of `between` is greater than its upper bound.
            ValueError: If `corpus` is not a recognized corpus to test the range against.
        """
        return self._exclude_by_range(
            kind="frequency",
            value_col_ref=FREQ_REF_COL,
            value_col_cmp=FREQ_CMP_COL,
            less_than=less_than,
            greater_than=greater_than,
            between=between,
            inclusive=inclusive,
            corpus=corpus,
        )

    def by_proportion(
        self,
        less_than: float | None = None,
        greater_than: float | None = None,
        between: tuple[float, float] | None = None,
        inclusive: bool = True,
        corpus: ExcludeCorpus = "both",
    ) -> Comparison:
        """
        Removes tokens whose proportion of a corpus falls in the given range from the reference and
        comparison corpora.

        The range is tested against each token's share of its corpus rather than a count, so that
        the same bounds mean the same thing in corpora of different sizes. Tokens are excluded from
        both corpora, before they are scored, so the frequencies the scoring measure works with are
        normalized over the remaining vocabulary.

        Example:
            ```python
            --8<-- "snippets/comparison/exclude.py:setup"

            --8<-- "snippets/comparison/exclude.py:by_proportion"
            ```

        Exactly one of `less_than`, `greater_than`, and `between` gives the range to exclude. To
        exclude by multiple disjoint criteria, call the method multiple times with different
        parameters. Proportions are taken over each corpus as the catalog gave it, before anything
        the comparison excludes, so the bounds mean the same thing no matter what else the
        comparison excludes or how it is scored.

        Warning:
            Excluding tokens changes the normalized frequencies of every token that remains.

            A token missing from one of the corpora has a proportion of zero, so `corpus="either"`
            paired with `less_than` removes every token that the two corpora do not share.

        Args:
            less_than:
                Exclude tokens making up less than this share of a corpus
            greater_than:
                Exclude tokens making up more than this share of a corpus
            between:
                Exclude tokens whose share of a corpus is between the lower and upper bound
            inclusive:
                Whether the bounds themselves are part of the range being excluded
            corpus:
                Which corpus's proportion the range is evaluated against. `"reference"` and
                `"comparison"` evaluate against a single corpus; `"both"` excludes a token only
                when the range holds in both corpora; `"either"` excludes a token when the range
                holds in either one

        Returns:
            A new `Comparison` whose vocabulary excludes tokens with proportions in the given range

        Raises:
            ValueError:
                If none of `less_than`, `greater_than`, and `between` is given, or if more than one
                of them is.
            ValueError: If the lower bound of `between` is greater than its upper bound.
            ValueError: If `corpus` is not a recognized corpus to test the range against.
            ValueError:
                If a bound is outside $[0, 1]$
        """
        given_bounds = [
            bound
            for bound in (less_than, greater_than, *(between if between is not None else ()))
            if bound is not None
        ]
        for bound in given_bounds:
            if not (0 <= bound <= 1):
                raise ValueError(
                    f"Proportions must be in [0, 1], got {bound}. Use `by_frequency` to exclude by "
                    "the frequencies the catalog was built from."
                )

        return self._exclude_by_range(
            kind="proportion",
            value_col_ref=NORMED_FREQ_REF_COL,
            value_col_cmp=NORMED_FREQ_CMP_COL,
            less_than=less_than,
            greater_than=greater_than,
            between=between,
            inclusive=inclusive,
            corpus=corpus,
        )

    def by_lexicon_score(
        self,
        less_than: float | None = None,
        greater_than: float | None = None,
        between: tuple[float, float] | None = None,
        inclusive: bool = True,
        corpus: ExcludeCorpus = "both",
    ) -> Comparison:
        """
        Removes tokens whose lexicon score falls in the given range from the reference and
        comparison corpora.

        The range is tested against the scores the comparison's lexicon assigns. Tokens are excluded
        from both corpora, before they are scored, so the frequencies the scoring measure works with
        are normalized over the remaining vocabulary.

        Example:
            ```python
            --8<-- "snippets/comparison/exclude.py:setup"

            --8<-- "snippets/comparison/exclude.py:by_lexicon_score"
            ```

        Exactly one of `less_than`, `greater_than`, and `between` gives the range to exclude. To
        exclude by multiple disjoint criteria, call the method multiple times with different
        parameters.

        Warning:
            Excluding tokens changes the normalized frequencies of every token that remains.

        Args:
            less_than:
                Exclude tokens with a lexicon score below this bound
            greater_than:
                Exclude tokens with a lexicon score above this bound
            between:
                Exclude tokens with lexicon scores between the lower and upper bound
            inclusive:
                Whether the bounds themselves are part of the range being excluded
            corpus:
                Which corpus's lexicon score the range is evaluated against. `"reference"` and
                `"comparison"` evaluate against the lexicons associated with each corpus; `"both"`
                excludes a token only when the range holds for the lexicons for both corpora;
                `"either"` excludes a token when the condition holds for either lexicon. All four
                specifications are equivalent when a single lexicon scores both corpora

        Returns:
            A new `Comparison` whose vocabulary excludes tokens scoring in the given range

        Raises:
            ValueError:
                If none of `less_than`, `greater_than`, and `between` is given, or if more than one
                of them is.
            ValueError: If the lower bound of `between` is greater than its upper bound.
            ValueError: If `corpus` is not a recognized corpus to test the range against.
            ValueError:
                If the comparison is not scored by a lexicon, so it has no lexicon scores to
                exclude by.
            ValueError:
                If tokens scored by only one lexicon are imputed rather than borrowed or dropped,
                because an imputed score is a modeling choice rather than what a lexicon says about
                a token.
        """
        self._comparison._state.validate_lexicon_score_exclusion()

        return self._exclude_by_range(
            kind="lexicon_score",
            value_col_ref=LEXICON_SCORE_REF_COL,
            value_col_cmp=LEXICON_SCORE_CMP_COL,
            less_than=less_than,
            greater_than=greater_than,
            between=between,
            inclusive=inclusive,
            corpus=corpus,
        )

    def _exclude_by_range(
        self,
        kind: EXCLUSION_RANGE_KIND,
        value_col_ref: str,
        value_col_cmp: str,
        less_than: float | None,
        greater_than: float | None,
        between: tuple[float, float] | None,
        inclusive: bool,
        corpus: ExcludeCorpus,
    ) -> Comparison:
        """
        Adds a plan excluding tokens whose values in the given columns fall in the given range.

        `kind` names what the range was tested against, and is the key the range is recorded under
        in the comparison's exclusions.
        """
        self._validate_range(
            less_than=less_than, greater_than=greater_than, between=between, corpus=corpus
        )
        state = self._comparison._state

        def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
            return exclude_by_range(
                df,
                value_col_ref=value_col_ref,
                value_col_cmp=value_col_cmp,
                less_than=less_than,
                greater_than=greater_than,
                between=between,
                inclusive=inclusive,
                corpus=corpus,
            )

        # Validation leaves exactly one bound, so the condition names the one that was given
        given_bounds = {
            name: bound
            for name, bound in (
                ("less_than", less_than),
                ("greater_than", greater_than),
                ("between", between),
            )
            if bound is not None
        }
        condition: dict[str, Any] = {**given_bounds, "inclusive": inclusive, "corpus": corpus}
        conditions = [*state.exclusions.get(kind, []), condition]

        return self._comparison._update_state(
            exclusion_plan=state.compose_plans(state.exclusion_plan, plan),
            exclusions={**state.exclusions, kind: conditions},
        )

    @staticmethod
    def _validate_range(
        less_than: float | None,
        greater_than: float | None,
        between: tuple[float, float] | None,
        corpus: ExcludeCorpus,
    ) -> None:
        """
        Checks that exactly one bound describes a well formed range, tested against a recognized
        corpus.

        Raises:
            ValueError:
                If none of `less_than`, `greater_than`, and `between` is given, or if more than one
                of them is.
            ValueError: If the lower bound of `between` is greater than its upper bound.
            ValueError: If `corpus` is not a recognized corpus to test the range against.
        """
        given_bounds = [
            name
            for name, bound in (
                ("less_than", less_than),
                ("greater_than", greater_than),
                ("between", between),
            )
            if bound is not None
        ]

        if len(given_bounds) == 0:
            raise ValueError("Must specify one of `less_than`, `greater_than`, or `between`")
        if len(given_bounds) > 1:
            raise ValueError(
                "`less_than`, `greater_than`, and `between` are mutually exclusive, got "
                f"{given_bounds}. Exclude more than one range by calling the method more than once."
            )
        if (between is not None) and (between[0] > between[1]):
            raise ValueError(
                f"The lower bound of `between` cannot exceed its upper bound, got {between}"
            )
        if corpus not in get_args(ExcludeCorpus):
            raise ValueError(f"`corpus` must be one of {get_args(ExcludeCorpus)}, got {corpus}")
