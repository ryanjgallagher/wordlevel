from __future__ import annotations

from typing import Any, Collection, get_args, TYPE_CHECKING

import polars as pl

from wordlevel.constants import LEXICON_SCORE_CMP_COL, LEXICON_SCORE_REF_COL
from wordlevel.core.comparison.exclude import (
    EXCLUDE_CORPUS,
    exclude_by_lexicon_score,
    exclude_tokens,
)

if TYPE_CHECKING:
    from wordlevel.api.comparison.comparison import Comparison


class ComparisonExcluder:
    """
    Namespace for excluding tokens from comparisons.
    """

    def __init__(self, comparison: Comparison):
        """ """
        self._comparison = comparison

    def tokens(self, tokens: Collection[str]) -> Comparison:
        """
        Removes the given tokens from the the reference and comparison corpora.

        Excluding tokens is a way of narrowing a comparison to the vocabulary that is of interest,
        such as removing stop words that carry little meaning of their own. Tokens are excluded
        from both corpora, before they are scored, so the frequencies the scoring measure works
        with are normalized over the remaining vocabulary.

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

    def by_lexicon_score(
        self,
        less_than: float | None = None,
        greater_than: float | None = None,
        between: tuple[float, float] | None = None,
        inclusive: bool = True,
        corpus: EXCLUDE_CORPUS = "both",
    ) -> Comparison:
        """
        Removes tokens whose lexicon score falls in the given range from the reference and
        comparison corpora.

        Excluding by lexicon score narrows a comparison to the part of a lexicon's scale that is of
        interest. Tokens are excluded from both corpora, before they are scored, so the frequencies
        the scoring measure works with are normalized over the remaining vocabulary.

        Example:
            ```python
            --8<-- "snippets/comparison/exclude.py:setup"

            --8<-- "snippets/comparison/exclude.py:by_lexicon_score"
            ```

        Exactly one of `less_than`, `greater_than`, and `between` gives the range to exclude. To
        exclude by multiple disjoint criteria, call the method multiple times with differnet
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
                specificiations are equivalent when a single lexicon scores both corpora

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
                If the lexicon scores are imputed by anything other than `"borrow"`, because an
                imputed score is a modeling choice rather than what a lexicon says about a token.
        """
        state = self._comparison._state
        state.validate_lexicon_score_exclusion()

        bounds = [
            (name, bound)
            for name, bound in (
                ("less_than", less_than),
                ("greater_than", greater_than),
                ("between", between),
            )
            if bound is not None
        ]
        if len(bounds) == 0:
            raise ValueError("Must specify one of `less_than`, `greater_than`, or `between`")
        if len(bounds) > 1:
            raise ValueError(
                "`less_than`, `greater_than`, and `between` are mutually exclusive, got "
                f"{[name for name, _ in bounds]}. Exclude more than one range of scores by "
                "calling `by_lexicon_score` more than once."
            )
        if between is not None and between[0] > between[1]:
            raise ValueError(
                f"The lower bound of `between` cannot exceed its upper bound, got {between}"
            )
        if corpus not in get_args(EXCLUDE_CORPUS):
            raise ValueError(f"`corpus` must be one of {get_args(EXCLUDE_CORPUS)}, got {corpus!r}")

        def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
            return exclude_by_lexicon_score(
                df,
                score_col_ref=LEXICON_SCORE_REF_COL,
                score_col_cmp=LEXICON_SCORE_CMP_COL,
                less_than=less_than,
                greater_than=greater_than,
                between=between,
                inclusive=inclusive,
                corpus=corpus,
            )

        bound_name, bound_value = bounds[0]
        condition: dict[str, Any] = {
            bound_name: bound_value,
            "inclusive": inclusive,
            "corpus": corpus,
        }
        conditions = [*state.exclusions.get("lexicon_score", []), condition]

        return self._comparison._update_state(
            exclusion_plan=state.compose_plans(state.exclusion_plan, plan),
            exclusions={**state.exclusions, "lexicon_score": conditions},
        )
