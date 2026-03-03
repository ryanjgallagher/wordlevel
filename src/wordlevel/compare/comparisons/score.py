from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TYPE_CHECKING

import polars as pl
from polars import col

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
from wordlevel.expressions.proportions import (
    proportion_dual_dist_scores,
    proportion_clean_dual_dist_scores,
)
from wordlevel.expressions.rank_divergence import rank_divergence
from wordlevel.expressions.frequency import frequency_difference
from wordlevel.expressions.shift.plan import basic_score_plan, weighted_average_score_plan

if TYPE_CHECKING:
    from wordlevel.compare.comparisons.comparison import Comparison


ScorePlan = Callable[[PolarsFrame], PolarsFrame]


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
        if requires_normed_freqs:
            ref = self._comparison._ref_normed
            cmp = self._comparison._cmp_normed
        else:
            ref = self._comparison._ref
            cmp = self._comparison._cmp

        plan = partial(
            basic_score_plan,
            freqs_ref=col(ref),
            freqs_cmp=col(cmp),
            score_func=score_func,
            attribute_by=attribute_by,
        )

        return plan

    def _make_weighted_avg_score_plan(
        self,
        score_func: DualScoreFunc,
        clean_score_func: DualCleanScoreFunc,
        attribute_by: ATTRIBUTION_TYPE,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
        requires_normed_freqs: bool = True,
    ) -> ScorePlan:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison when the
        scoring is done using a weighted average method
        """
        ref_normed = self._comparison._ref_normed
        cmp_normed = self._comparison._cmp_normed
        if requires_normed_freqs:
            ref = ref_normed
            cmp = cmp_normed
        else:
            ref = self._comparison._ref
            cmp = self._comparison._cmp

        plan = partial(
            weighted_average_score_plan,
            freqs_to_score_ref=col(ref),
            freqs_to_score_cmp=col(cmp),
            normed_freqs_ref=col(ref_normed),
            normed_freqs_cmp=col(cmp_normed),
            score_func=score_func,
            clean_score_func=clean_score_func,
            attribute_by=attribute_by,
            reference_score=reference_score,
        )

        return plan

    def _return_comparison(
        self,
        plan: PolarsFrame,
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
        plan = self._make_basic_score_plan(
            score_func, requires_normed_freqs=False, attribute_by="rank"
        )

        return self._return_comparison(
            plan,
            method="rank_divergence",
            parameters={"alpha": alpha},
            is_weighted_avg=False,
            is_nonnegative=True,
        )

    def frequency(self) -> Comparison:
        """ """
        plan = self._make_basic_score_plan(
            frequency_difference, requires_normed_freqs=False, attribute_by="frequency"
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
            reference_score=reference_score,
            requires_normed_freqs=True,
            attribute_by="frequency",
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
            reference_score=reference_score,
            requires_normed_freqs=True,
            attribute_by="score_sign",
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
            reference_score=reference_score,
            requires_normed_freqs=True,
            attribute_by="score_sign",
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
            reference_score=reference_score,
            requires_normed_freqs=True,
            attribute_by="frequency",
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
