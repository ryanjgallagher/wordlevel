from __future__ import annotations
from functools import partial
from typing import Any, Literal, TYPE_CHECKING

import polars as pl
from polars import col

from wordlevel.types import PolarsFrame
from wordlevel.compare.comparisons.types import (
    ScorePlan,
    ScoreFunc,
    DualScoreFunc,
    DualCleanScoreFunc,
)
from wordlevel.expressions.attribution import (
    frequency_attribution,
    rank_attribution,
    score_sign_attribution,
)
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
from wordlevel.expressions.proportions import (
    proportion_dual_dist_scores,
    proportion_clean_dual_dist_scores,
)
from wordlevel.expressions.rank_divergence import rank_divergence
from wordlevel.expressions.frequency import frequency_difference
from wordlevel.expressions.shift_components import get_shift_components
from wordlevel.expressions.shift_plot_components import get_shift_plot_components

if TYPE_CHECKING:
    from wordlevel.compare.comparisons.comparison import Comparison


ATTRIBUTION_TYPE = Literal["frequency", "rank", "score_sign"]


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

    def _make_score_plan(
        self,
        score_func: ScoreFunc,
        attribute_by: ATTRIBUTION_TYPE,
        requires_normed_freqs: bool = True,
    ) -> ScorePlan:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison
        """

        def plan(df: PolarsFrame) -> PolarsFrame:
            """ """
            if requires_normed_freqs:
                ref = self._comparison._ref_normed
                cmp = self._comparison._cmp_normed
            else:
                ref = self._comparison._ref
                cmp = self._comparison._cmp

            df = df.with_columns(
                score_func(freqs1=col(ref), freqs2=col(cmp)).alias("comparison_score")
            ).with_columns(col("comparison_score").abs().rank(descending=True).alias("rank"))

            df = self._attribute_scores(
                df=df, by=attribute_by, requires_normed_freqs=requires_normed_freqs
            )

            return df

        return plan

    def _make_weighted_avg_score_plan(
        self,
        freqs_ref: str,
        freqs_cmp: str,
        score_func: DualScoreFunc,
        clean_score_func: DualCleanScoreFunc,
        attribute_by: ATTRIBUTION_TYPE,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
        requires_normed_freqs: bool = True,
    ) -> ScorePlan:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison when the
        scoring is done using a weighted average method

        TODO:
        - Make some better source of truth for the column names, especially the shift component ones
        - Move all aliases to an object that can be used as a source of truth and configuration
        """

        def score_plan(df: PolarsFrame) -> PolarsFrame:
            """ """
            if requires_normed_freqs:
                ref = self._comparison._ref_normed
                cmp = self._comparison._cmp_normed
            else:
                ref = self._comparison._ref
                cmp = self._comparison._cmp

            df_score = df.with_columns(
                *score_func(
                    freqs1=col(ref), freqs2=col(cmp), alias1="raw_score_ref", alias2="raw_score_cmp"
                ),
            ).with_columns(
                *clean_score_func(
                    freqs1=col(ref),
                    freqs2=col(cmp),
                    scores1=col("raw_score_ref"),
                    scores2=col("raw_score_cmp"),
                    alias1="clean_score_ref",
                    alias2="clean_score_cmp",
                )
            )

            return df_score

        def plot_plan(df: PolarsFrame) -> PolarsFrame:
            """ """
            component_ref = f"{freqs_ref}{self._comparison._normalization_suffix}"
            component_cmp = f"{freqs_cmp}{self._comparison._normalization_suffix}"

            df_score = get_shift_components(
                df,
                freqs1=col(component_ref),
                freqs2=col(component_cmp),
                scores1=col("clean_score_ref"),
                scores2=col("clean_score_cmp"),
                reference_score=reference_score,
                normalize=False,
                include_total_difference=True,
                include_rank=True,
                frequency_diff_alias="freq_diff",
                weighted_score_diff_alias="weighted_score_diff",
                reference_score_diff_alias="ref_score_diff",
                total_difference_alias="comparison_score",
            )
            df_score = get_shift_plot_components(
                df_score,
                frequency_diff=col("freq_diff"),
                weighted_score_diff=col("weighted_score_diff"),
                reference_score_diff=col("ref_score_diff"),
            )

            return df_score.drop("clean_score_ref", "clean_score_cmp").rename(
                {"raw_score_ref": "score_ref", "raw_score_cmp": "score_cmp"}
            )

        def full_plan(df: PolarsFrame) -> PolarsFrame:
            """ """
            df = plot_plan(score_plan(df))

            df = self._attribute_scores(
                df=df, by=attribute_by, requires_normed_freqs=requires_normed_freqs
            )

            return df

        return full_plan

    def _attribute_scores(
        self,
        df: PolarsFrame,
        by: ATTRIBUTION_TYPE,
        requires_normed_freqs: bool,
    ) -> PolarsFrame:
        """ """
        if requires_normed_freqs:
            ref = self._comparison._ref_normed
            cmp = self._comparison._cmp_normed
        else:
            ref = self._comparison._ref
            cmp = self._comparison._cmp

        if by == "frequency":
            expr = frequency_attribution(reference=col(ref), comparison=col(cmp), normalize=False)
        elif by == "rank":
            expr = rank_attribution(reference=col(ref), comparison=col(cmp))
        elif by == "score_sign":
            expr = score_sign_attribution(score=col("comparison_score"))
        else:
            raise ValueError("Invalid specification of how to attribute scores")

        return df.with_columns(expr.alias("attribution"))

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
            normalization_suffix=self._comparison._normalization_suffix,
        )

    def rank_divergence(self, alpha: float = 1.0) -> Comparison:
        """ """
        score_func = partial(rank_divergence, alpha=alpha)
        plan = self._make_score_plan(score_func, attribute_by="rank", requires_normed_freqs=False)

        return self._return_comparison(
            plan,
            method="rank_divergence",
            parameters={"alpha": alpha},
            is_weighted_avg=False,
            is_nonnegative=True,
        )

    def frequency(self) -> Comparison:
        """ """
        plan = self._make_score_plan(
            frequency_difference, attribute_by="frequency", requires_normed_freqs=False
        )

        return self._return_comparison(
            plan, method="frequency", parameters=dict(), is_weighted_avg=False, is_nonnegative=False
        )

    def proportion(self, reference_score: float | Literal["mean"] | pl.Expr = 0) -> Comparison:
        """ """
        plan = self._make_weighted_avg_score_plan(
            freqs_ref=self._comparison._ref,
            freqs_cmp=self._comparison._cmp,
            score_func=proportion_dual_dist_scores,
            clean_score_func=proportion_clean_dual_dist_scores,
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

    def tsallis_entropy(
        self,
        alpha: float = 1,
        base: float = 2,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
    ) -> Comparison:
        """ """
        dual_score_func = partial(
            tsallis_entropy_dual_dist_scores, alpha=alpha, base=base, normalize=False
        )

        plan = self._make_weighted_avg_score_plan(
            freqs_ref=self._comparison._ref,
            freqs_cmp=self._comparison._cmp,
            score_func=dual_score_func,
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
        dual_score_func = partial(
            kullback_leibler_divergence_dual_dist_scores, base=base, normalize=False
        )

        plan = self._make_weighted_avg_score_plan(
            freqs_ref=self._comparison._ref,
            freqs_cmp=self._comparison._ref,
            score_func=dual_score_func,
            clean_score_func=kullback_leibler_divergence_clean_dual_dist_scores,
            attribute_by="score_sign",
            reference_score=reference_score,
            requires_normed_freqs=True,
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
        dual_score_func = partial(
            jensen_shannon_divergence_dual_dist_scores,
            alpha=alpha,
            base=base,
            weight1=weight_ref,
            weight2=weight_cmp,
            normalize=False,
        )

        plan = self._make_weighted_avg_score_plan(
            freqs_ref=self._comparison._ref,
            freqs_cmp=self._comparison._cmp,
            score_func=dual_score_func,
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
