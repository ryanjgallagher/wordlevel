from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, get_args

import polars as pl

from wordlevel.core.comparison.normalization import NORMALIZE_BY
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_TYPE
from wordlevel.core.comparison.state import (
    renormalize_freqs,
    restore_corpus_names,
    select_corpora,
)


class PlanStage(Protocol):
    """
    A plan for a single stage of the pipeline that a
    [`Comparison`][wordlevel.api.comparison.comparison.Comparison] executes.

    A plan takes corpus data and transforms it, so that plans can be applied one after another to
    build up the full comparison.
    """

    def __call__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        token_col: str,
    ) -> pl.DataFrame | pl.LazyFrame: ...


@dataclass(frozen=True)
class ComparisonState:
    """
    The state of a [`Comparison`][wordlevel.api.comparison.comparison.Comparison], indicating the
    corpora it compares, the metadata describing how they are scored, and the plans that transform
    and score them.

    A `Comparison` is immutable, so each of its operations returns a new state rather than
    modifying the existing one. Each operation writes to the plan it belongs to, so the order in
    which the user calls operations does not affect the order they execute in.

    Attributes:
        reference:
            Name of the reference corpus
        comparison:
            Name of the comparison corpus
        alias:
            The name of the comparison
        method:
            The name of a scoring method
        is_weighted_avg:
            Whether the scoring method can be interpreted as a weighted average or difference in
            weighted averages
        is_nonnegative:
            Whether the scoring method is inherently non-negative, i.e. all token-level
            contributions are non-negative
        attributed_by:
            The attribution type used to split token contributions between the two bars of a bar
            plot (`"frequency"` and `"rank"` attribute by corpus; `"score_sign"` attributes by
            whether a token reinforces or offsets the overall difference)
        parameters:
            Any parameters specific to the scoring method when making the comparison
        normalized_by:
            The normalization applied to token contributions, or `None` if they are not normalized
        exclusions:
            The exclusions applied to the vocabulary, keyed by the kind of exclusion. Each value
            describes what that kind of exclusion removed, e.g. `{"tokens": {"the", "a"}}`
        lexicon_join_plan:
            The plan joining lexicon scores onto the corpus data
        exclusion_plan:
            The plan restricting the vocabulary of the two corpora
        score_plan:
            The plan scoring the reference and comparison corpora according to the specified method
        score_normalization_plan:
            The plan rescaling token contributions onto a common scale
    """

    reference: str
    comparison: str
    alias: str | None = None
    method: str | None = None
    is_weighted_avg: bool | None = None
    is_nonnegative: bool | None = None
    attributed_by: ATTRIBUTION_TYPE | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    normalized_by: NORMALIZE_BY | None = None
    exclusions: dict[str, Any] = field(default_factory=dict)
    lexicon_join_plan: PlanStage | None = None
    exclusion_plan: PlanStage | None = None
    score_plan: PlanStage | None = None
    score_normalization_plan: PlanStage | None = None

    def __post_init__(self) -> None:
        # Copy the parameters so a caller mutating the mapping they passed does not affect the state
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "exclusions", dict(self.exclusions))

    @staticmethod
    def compose_plans(base_plan: PlanStage | None, wrapping_plan: PlanStage) -> PlanStage:
        """
        Composes one plan onto another, so that the wrapping plan is applied to the base plan's
        output. If there is no base plan, the wrapping plan becomes the plan.

        Args:
            base_plan:
                The existing plan, or `None` if there is not one yet
            wrapping_plan:
                The plan to apply to the existing plan ("wrapping" it)

        Returns:
            A plan first applying the `base_plan` and then the `wrapping_plan`
        """
        if base_plan is None:
            return wrapping_plan

        def composed(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            return wrapping_plan(base_plan(df, token_col), token_col)

        return composed

    def resolve(self) -> PlanStage:
        """
        Composes the comparison's different transformation plans into a single plan.

        The transformation plans are always composed in the same order, regardless of the order in
        which the user called the operations that populated them. Alongside the plans the user's
        operations populate, the comparison contributes three plans of its own: it selects the two
        corpora being compared, renormalizes their frequencies once the vocabulary is settled, and
        renames their columns back to names derived from the corpora.

        Returns:
            A plan applying every populated plan of the comparison in the specified order

        Raises:
            ValueError: If the comparison has not been scored, so there is no plan to resolve.
            ValueError: If the normalization is not valid for the scoring method.
            ValueError: If excluding by lexicon score is not valid for the scoring method.
        """
        if self.score_plan is None:
            raise ValueError("Cannot resolve the plan of a comparison that has not been scored")

        if self.normalized_by is not None:
            self.validate_normalization(self.normalized_by)
        if "lexicon_score" in self.exclusions:
            self.validate_lexicon_score_exclusion()

        def select_corpora_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            return select_corpora(df, token_col, self.reference, self.comparison)

        def renormalize_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            return renormalize_freqs(df)

        def restore_corpus_names_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            return restore_corpus_names(df, self.reference, self.comparison)

        plans = [
            plan
            for plan in (
                select_corpora_plan,
                # Excluding by proportion needs normalized frequencies before narrowing the vocab
                renormalize_plan if "proportion" in self.exclusions else None,
                self.lexicon_join_plan,
                self.exclusion_plan,
                renormalize_plan,
                self.score_plan,
                self.score_normalization_plan,
                restore_corpus_names_plan,
            )
            if plan is not None
        ]

        resolved_plan = plans[0]
        for plan in plans[1:]:
            resolved_plan = self.compose_plans(resolved_plan, plan)

        return resolved_plan

    def validate_normalization(self, by: NORMALIZE_BY) -> None:
        """
        Checks that a normalization strategy is valid for how the comparison is scored.

        Args:
            by:
                The normalization strategy to check

        Raises:
            ValueError: If `by` is not a recognized normalization strategy.
            ValueError:
                If `by` is `"abs_sum"` for a `proportion` comparison, whose contributions sum to
                zero by construction, so the absolute sum is always zero.
        """
        if by not in get_args(NORMALIZE_BY):
            raise ValueError(f"`by` must be one of {get_args(NORMALIZE_BY)}, got {by!r}")
        if by == "abs_sum" and self.method == "proportion":
            raise ValueError(
                "Cannot normalize a 'proportion' comparison by 'abs_sum': proportions sum to one "
                "in each corpus, so the net difference is zero by construction. Use 'sum_abs' "
                "instead."
            )

    def validate_lexicon_score_exclusion(self) -> None:
        """
        Checks that excluding by lexicon score is valid for how the comparison is scored.

        A comparison that has not been scored yet passes, because the scoring method it will be
        given is not known until every plan is in place.

        Raises:
            ValueError:
                If the comparison is not scored by a lexicon, so it has no lexicon scores to
                exclude by.
            ValueError:
                If the lexicon scores are imputed by anything other than `"borrow"`, because an
                imputed score is a modeling choice rather than what a lexicon says about a token.
        """
        if self.score_plan is None:
            return

        if self.method != "lexicon":
            raise ValueError(
                "Can only exclude by lexicon score when the comparison is scored by a lexicon, "
                f"but it is scored by {self.method!r}."
            )

        impute = self.parameters.get("impute")
        if impute is not None and not (isinstance(impute, str) and impute == "borrow"):
            raise ValueError(
                f"Cannot exclude by lexicon score when `impute` is {impute!r}: an imputed score is "
                "a modeling choice rather than what a lexicon says about a token. Only the "
                "'borrow' imputation strategy is compatible"
            )
