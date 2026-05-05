from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, Mapping, TYPE_CHECKING

import polars as pl
from polars import col

from wordlevel.configs import ShiftTypesConfig
from wordlevel.constants import (
    SCORE_COL,
    RANK_COL,
    ATTRIBUTION_COL,
    REFERENCE_ATTRIBUTION,
    COMPARISON_ATTRIBUTION,
    COUNTERACTED_SUFFIX,
)
from wordlevel.types import PolarsFrame
from wordlevel.expressions.types import (
    ScoreFunc,
    DualScoreFunc,
    DualCleanScoreFunc,
)
from wordlevel.expressions.attribution import ATTRIBUTION_TYPE
from wordlevel.expressions.entropy import (
    tsallis_entropy_dual_dist_scores,
    tsallis_entropy_clean_dual_dist_scores,
)
from wordlevel.expressions.kullback_leibler_divergence import (
    kullback_leibler_divergence_dual_dist_scores,
    kullback_leibler_divergence_clean_dual_dist_scores,
)
from wordlevel.expressions.jensen_shannon_divergence import (
    jensen_shannon_divergence_dual_dist_scores,
    jensen_shannon_divergence_clean_dual_dist_scores,
)
from wordlevel.expressions.rank_divergence import rank_divergence
from wordlevel.expressions.frequency import frequency_difference
from wordlevel.expressions.shift.plan import (
    basic_score_plan,
    weighted_average_score_plan,
    lexicon_join_plan,
)
from wordlevel.lexicons.lexicon import Lexicon

if TYPE_CHECKING:
    from wordlevel.compare.comparisons.comparison import Comparison


ScorePlan = Callable[[PolarsFrame, str, str], PolarsFrame]


def _fixed_dual_score_func(expr1: pl.Expr, expr2: pl.Expr) -> DualScoreFunc:
    def func(freqs1: pl.Expr, freqs2: pl.Expr, alias1: str, alias2: str) -> tuple[pl.Expr, pl.Expr]:
        return expr1.alias(alias1), expr2.alias(alias2)

    return func


def _passthrough_clean_score_func(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    alias1: str,
    alias2: str,
) -> tuple[pl.Expr, pl.Expr]:
    return scores1.alias(alias1), scores2.alias(alias2)


class Scorer:
    """ """

    def __init__(self, comparison: Comparison):
        """ """
        self._comparison = comparison

    def _get_default_alias(self, method: str) -> str:
        """ """
        return (
            f"reference:{self._comparison._ref}_comparison:{self._comparison._cmp}_method:{method}"
        )

    def _make_basic_score_plan(
        self,
        score_func: ScoreFunc,
        attribute_by: ATTRIBUTION_TYPE,
        requires_normed_freqs: bool = True,
    ) -> ScorePlan:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison
        """
        ref = self._comparison._ref
        cmp = self._comparison._cmp

        def plan(df: PolarsFrame, label_col: str, normalization_suffix: str) -> PolarsFrame:
            if requires_normed_freqs:
                ref_col = f"{ref}{normalization_suffix}"
                cmp_col = f"{cmp}{normalization_suffix}"
            else:
                ref_col = ref
                cmp_col = cmp

            return basic_score_plan(
                df,
                freqs_ref=col(ref_col),
                freqs_cmp=col(cmp_col),
                label=col(label_col),
                score_func=score_func,
                attribute_by=attribute_by,
                score_alias=SCORE_COL,
                rank_alias=RANK_COL,
                attribution_alias=ATTRIBUTION_COL,
                reference_attribution=REFERENCE_ATTRIBUTION,
                comparison_attribution=COMPARISON_ATTRIBUTION,
            )

        return plan

    def _make_weighted_avg_score_plan(
        self,
        score_func: DualScoreFunc,
        clean_score_func: DualCleanScoreFunc,
        attribute_by: ATTRIBUTION_TYPE,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
        requires_normed_freqs: bool = True,
        share_component_freqs_by: Literal["reference", "comparison"] | None = None,
    ) -> ScorePlan:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison when the
        scoring is done using a weighted average method
        """
        ref = self._comparison._ref
        cmp = self._comparison._cmp

        SHIFT_TYPES = ShiftTypesConfig()

        def plan(df: PolarsFrame, label_col: str, normalization_suffix: str) -> PolarsFrame:
            if requires_normed_freqs:
                freqs_ref = f"{ref}{normalization_suffix}"
                freqs_cmp = f"{cmp}{normalization_suffix}"
            else:
                freqs_ref = ref
                freqs_cmp = cmp

            component_freqs_ref = freqs_ref
            component_freqs_cmp = freqs_cmp
            if share_component_freqs_by == "reference":
                component_freqs_cmp = component_freqs_ref
            elif share_component_freqs_by == "comparison":
                component_freqs_ref = component_freqs_cmp

            return weighted_average_score_plan(
                df,
                score_freqs_ref=col(freqs_ref),
                score_freqs_cmp=col(freqs_cmp),
                component_freqs_ref=col(component_freqs_ref),
                component_freqs_cmp=col(component_freqs_cmp),
                label=col(label_col),
                score_func=score_func,
                clean_score_func=clean_score_func,
                attribute_by=attribute_by,
                reference_score=reference_score,
                score_alias=SCORE_COL,
                rank_alias=RANK_COL,
                more_of_positive_alias=SHIFT_TYPES.more_of_positive.name,
                less_of_negative_alias=SHIFT_TYPES.less_of_negative.name,
                more_of_negative_alias=SHIFT_TYPES.more_of_negative.name,
                less_of_positive_alias=SHIFT_TYPES.less_of_positive.name,
                higher_score_alias=SHIFT_TYPES.higher_score.name,
                lower_score_alias=SHIFT_TYPES.lower_score.name,
                counteracted_suffix=COUNTERACTED_SUFFIX,
                attribution_alias=ATTRIBUTION_COL,
                reference_attribution=REFERENCE_ATTRIBUTION,
                comparison_attribution=COMPARISON_ATTRIBUTION,
            )

        return plan

    def _return_comparison(
        self,
        plan: ScorePlan,
        method: str,
        parameters: dict[str, Any],
        is_weighted_avg: bool,
        is_nonnegative: bool,
    ) -> Comparison:
        """ """
        if self._comparison._alias is None:
            alias = self._get_default_alias(method)
        else:
            alias = self._comparison._alias

        return self._comparison.__class__(
            reference=self._comparison._ref,
            comparison=self._comparison._cmp,
            alias=alias,
            method=method,
            parameters=parameters,
            is_weighted_avg=is_weighted_avg,
            is_nonnegative=is_nonnegative,
            plan=plan,
        )

    def frequency(self) -> Comparison:
        """ """
        plan = self._make_basic_score_plan(
            frequency_difference,
            attribute_by="frequency",
            requires_normed_freqs=False,
        )

        return self._return_comparison(
            plan, method="frequency", parameters=dict(), is_weighted_avg=False, is_nonnegative=False
        )

    def proportion(self, reference_score: float | Literal["mean"] | pl.Expr = 0) -> Comparison:
        """ """
        plan = self._make_weighted_avg_score_plan(
            score_func=_fixed_dual_score_func(pl.lit(1), pl.lit(1)),
            clean_score_func=_passthrough_clean_score_func,
            attribute_by="frequency",
            reference_score=reference_score,
            requires_normed_freqs=True,
        )

        return self._return_comparison(
            plan,
            method="proportion",
            parameters={"reference_score": reference_score},
            is_weighted_avg=True,
            is_nonnegative=False,
        )

    def lexicon(
        self,
        lexicon: Lexicon | Mapping[str, float] | PolarsFrame | None = None,
        lexicon_reference: Lexicon | Mapping[str, float] | PolarsFrame | None = None,
        lexicon_comparison: Lexicon | Mapping[str, float] | PolarsFrame | None = None,
        lexicon_label_col: str | None = None,
        lexicon_score_col: str | None = None,
        reference_score: float | Literal["mean", "center"] | pl.Expr | None = None,
        impute: float | Literal["borrow", "mean"] | pl.Expr = "borrow",
    ) -> Comparison:
        """ """
        score_alias_ref = "lexicon_score_ref"
        score_alias_cmp = "lexicon_score_cmp"

        if reference_score is None:
            is_lexicon = (
                isinstance(lexicon, Lexicon)
                or isinstance(lexicon_reference, Lexicon)
                or isinstance(lexicon_comparison, Lexicon)
            )
            if is_lexicon:
                reference_score = "center"
            else:
                reference_score = 0

        if not isinstance(reference_score, pl.Expr) and (reference_score == "center"):
            if isinstance(lexicon, Lexicon):
                reference_score = lexicon.center
            elif isinstance(lexicon_reference, Lexicon):
                reference_score = lexicon_reference.center
            elif isinstance(lexicon_comparison, Lexicon):
                reference_score = lexicon_comparison.center
            else:
                raise ValueError(
                    "Cannot determine the 'center' to use as a reference score because a "
                    "pre-defined lexicon was not provided"
                )

        score_plan = self._make_weighted_avg_score_plan(
            score_func=_fixed_dual_score_func(col(score_alias_ref), col(score_alias_cmp)),
            clean_score_func=_passthrough_clean_score_func,
            attribute_by="score_sign",
            reference_score=reference_score,
            requires_normed_freqs=True,
        )

        def plan(df: PolarsFrame, label_col: str, normalization_suffix: str) -> PolarsFrame:
            df = lexicon_join_plan(
                df,
                df_label_col=label_col,
                lexicon=lexicon,
                lexicon_ref=lexicon_reference,
                lexicon_cmp=lexicon_comparison,
                lexicon_label_col=lexicon_label_col,
                lexicon_score_col=lexicon_score_col,
                score_alias_ref=score_alias_ref,
                score_alias_cmp=score_alias_cmp,
                impute=impute,
                drop_null_scores=True,
            )

            return score_plan(df, label_col=label_col, normalization_suffix=normalization_suffix)

        return self._return_comparison(
            plan,
            method="lexicon",
            parameters={"reference_score": reference_score, "impute": impute},
            is_weighted_avg=True,
            is_nonnegative=False,
        )

    def tsallis_entropy(
        self,
        alpha: float = 1,
        base: float = 2,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
    ) -> Comparison:
        """ """
        plan = self._make_weighted_avg_score_plan(
            score_func=partial(
                tsallis_entropy_dual_dist_scores, alpha=alpha, base=base, normalize=False
            ),
            clean_score_func=tsallis_entropy_clean_dual_dist_scores,
            attribute_by="score_sign",
            reference_score=reference_score,
            requires_normed_freqs=True,
        )

        return self._return_comparison(
            plan,
            method="tsallis_entropy",
            parameters={"alpha": alpha, "base": base, "reference_score": reference_score},
            is_weighted_avg=True,
            is_nonnegative=False,
        )

    def shannon_entropy(
        self, base: float = 2, reference_score: float | Literal["mean"] | pl.Expr = 0
    ) -> Comparison:
        """ """
        return self.tsallis_entropy(alpha=1, base=base, reference_score=reference_score)

    def kullback_leibler_divergence(
        self, base: float = 2, reference_score: float | Literal["mean"] | pl.Expr = 0
    ) -> Comparison:
        """ """
        plan = self._make_weighted_avg_score_plan(
            score_func=partial(
                kullback_leibler_divergence_dual_dist_scores, base=base, normalize=False
            ),
            clean_score_func=kullback_leibler_divergence_clean_dual_dist_scores,
            attribute_by="score_sign",
            reference_score=reference_score,
            requires_normed_freqs=True,
            share_component_freqs_by="reference",
        )

        return self._return_comparison(
            plan,
            method="kullback_leibler_divergence",
            parameters={"base": base, "reference_score": reference_score},
            is_weighted_avg=True,
            is_nonnegative=False,
        )

    def kld(
        self, base: float = 2, reference_score: float | Literal["mean"] | pl.Expr = 0
    ) -> Comparison:
        """ """
        return self.kullback_leibler_divergence(base=base, reference_score=reference_score)

    def jensen_shannon_divergence(
        self,
        alpha: float = 1,
        base: float = 2,
        weight_ref: float = 0.5,
        weight_cmp: float = 0.5,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
    ) -> Comparison:
        """ """
        plan = self._make_weighted_avg_score_plan(
            score_func=partial(
                jensen_shannon_divergence_dual_dist_scores,
                alpha=alpha,
                base=base,
                weight1=weight_ref,
                weight2=weight_cmp,
                normalize=False,
            ),
            clean_score_func=jensen_shannon_divergence_clean_dual_dist_scores,
            attribute_by="frequency",
            reference_score=reference_score,
            requires_normed_freqs=True,
        )

        return self._return_comparison(
            plan,
            method="jensen_shannon_divergence",
            parameters={
                "alpha": alpha,
                "base": base,
                "weight_ref": weight_ref,
                "weight_cmp": weight_cmp,
                "reference_score": reference_score,
            },
            is_weighted_avg=True,
            is_nonnegative=True,
        )

    def jsd(
        self,
        alpha: float = 1,
        base: float = 2,
        weight_ref: float = 0.5,
        weight_cmp: float = 0.5,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
    ) -> Comparison:
        """ """
        return self.jensen_shannon_divergence(
            alpha=alpha,
            base=base,
            weight_ref=weight_ref,
            weight_cmp=weight_cmp,
            reference_score=reference_score,
        )

    def rank_divergence(self, alpha: float = 1.0) -> Comparison:
        """ """
        score_func = partial(rank_divergence, alpha=alpha)
        plan = self._make_basic_score_plan(
            score_func,
            attribute_by="rank",
            requires_normed_freqs=False,
        )

        return self._return_comparison(
            plan,
            method="rank_divergence",
            parameters={"alpha": alpha},
            is_weighted_avg=False,
            is_nonnegative=True,
        )
