from __future__ import annotations

from typing import Collection, TYPE_CHECKING

import polars as pl

from wordlevel.core.comparison.exclude import exclude_tokens

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
