from __future__ import annotations
from functools import partial
from typing import Any, Literal, Mapping, Protocol, TYPE_CHECKING

import polars as pl
from polars import col

from wordlevel.configs import SHIFT_BAR_NAMES
from wordlevel.constants import (
    SCORE_COL,
    RANK_COL,
    ATTRIBUTION_COL,
    REFERENCE_ATTRIBUTION,
    COMPARISON_ATTRIBUTION,
    COUNTERACTED_SUFFIX,
)
from wordlevel.core.comparison.score.plan import (
    ScoreFunc,
    DualScoreFunc,
    DualCleanScoreFunc,
)
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_TYPE
from wordlevel.core.comparison.score.metrics.entropy import (
    tsallis_entropy_dual_dist_scores,
    tsallis_entropy_clean_dual_dist_scores,
)
from wordlevel.core.comparison.score.metrics.kullback_leibler_divergence import (
    kullback_leibler_divergence_dual_dist_scores,
    kullback_leibler_divergence_clean_dual_dist_scores,
)
from wordlevel.core.comparison.score.metrics.jensen_shannon_divergence import (
    jensen_shannon_divergence_dual_dist_scores,
    jensen_shannon_divergence_clean_dual_dist_scores,
)
from wordlevel.core.comparison.score.metrics.rank_divergence import rank_divergence
from wordlevel.core.comparison.score.metrics.frequency import frequency_difference
from wordlevel.core.comparison.score.plan import (
    basic_score_plan,
    weighted_average_score_plan,
    lexicon_join_plan,
)
from wordlevel.core.lexicons.load import Lexicon

if TYPE_CHECKING:
    from wordlevel.api.comparison.comparison import Comparison


class ScorePlan(Protocol):
    def __call__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        token_col: str,
        normalization_suffix: str,
    ) -> pl.DataFrame | pl.LazyFrame: ...


def _fixed_dual_score_func(expr1: pl.Expr, expr2: pl.Expr) -> DualScoreFunc:
    """ """

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
    """ """
    return scores1.alias(alias1), scores2.alias(alias2)


class Scorer:
    """
    Namespace for scoring comparisons.
    """

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

        def plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str, normalization_suffix: str
        ) -> pl.DataFrame | pl.LazyFrame:
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
                token=col(token_col),
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

        def plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str, normalization_suffix: str
        ) -> pl.DataFrame | pl.LazyFrame:
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
                token=col(token_col),
                score_func=score_func,
                clean_score_func=clean_score_func,
                attribute_by=attribute_by,
                reference_score=reference_score,
                score_alias=SCORE_COL,
                rank_alias=RANK_COL,
                more_of_positive_alias=SHIFT_BAR_NAMES.more_of_positive,
                less_of_negative_alias=SHIFT_BAR_NAMES.less_of_negative,
                more_of_negative_alias=SHIFT_BAR_NAMES.more_of_negative,
                less_of_positive_alias=SHIFT_BAR_NAMES.less_of_positive,
                higher_score_alias=SHIFT_BAR_NAMES.higher_score,
                lower_score_alias=SHIFT_BAR_NAMES.lower_score,
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

        return type(self._comparison)._with_state(
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
        r"""
        Calculates the frequency difference between the reference and comparison corpora.

        Each token's contribution is the difference in its raw frequency between each corpus.
        Specifically, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau = f_\tau^{(C)} - f_\tau^{(R)}
        $$

        where $f_\tau^{(R)}$ and $f_\tau^{(C)}$ are the raw frequencies of token $\tau$ in $R$ and
        $C$ respectively, the reference and comparison corpora. Positive contributions identify
        tokens that occur more often in the comparison; negative contributions identify tokens that
        occur more often in the reference.

        Returns:
            A new `Comparison` with the corpora scored by frequency differences
        """
        plan = self._make_basic_score_plan(
            frequency_difference,
            attribute_by="frequency",
            requires_normed_freqs=False,
        )

        return self._return_comparison(
            plan, method="frequency", parameters=dict(), is_weighted_avg=False, is_nonnegative=False
        )

    def proportion(self, reference_score: float | Literal["mean"] | pl.Expr = 0) -> Comparison:
        r"""
        Calculates the proportion difference between the reference and comparison corpora.

        Each token's contribution is the difference in its normalized frequency (i.e. proportion)
        between each corpus. When the `reference_score` is 0 (the default), the contribution
        $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau = p_\tau^{(C)} - p_\tau^{(R)}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora. Positive contributions
        identify tokens that occur relatively more often in the comparison; negative contributions
        identify tokens that occur relatively more often in the reference.

        When the `reference_score` is non-zero, the contribution of the token is:

        $$
        \delta_\tau =
            \left(
                p_\tau^{(C)} - p_\tau^{(R)}
            \right)
            \left(
                1 - \Phi^{(\text{ref})}
            \right)
        $$

        where $\Phi^{(\text{ref})}$ is the user-provided reference score.

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        Args:
            reference_score:
                Baseline against which per-token contributions are interpreted in the
                weighted-average decomposition. Defaults to 0

        Returns:
            A new `Comparison` with the corpora scored by proportion differences
        """
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
        lexicon: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame | None = None,
        lexicon_reference: Lexicon
        | Mapping[str, float]
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        lexicon_comparison: Lexicon
        | Mapping[str, float]
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        lexicon_token_col: str | None = None,
        lexicon_score_col: str | None = None,
        reference_score: float | Literal["mean", "center"] | pl.Expr | None = None,
        impute: float | Literal["borrow", "mean"] | pl.Expr = "borrow",
    ) -> Comparison:
        r"""
        Calculates the difference in lexicon-weighted averages between the reference and comparison
        corpora.

        The supplied lexicon associates a score with each token. The user can provide a single
        lexicon to use for both corpora, or one lexicon for each. The overall score for each corpus
        individually is the lexicon-weighted average. Specifically, if $p_\tau$ is the normalized
        frequency of a token $\tau$ in a corpus, and $\phi_\tau$ is the score assigned to it by
        the lexicon, the the corpus-level score is $\sum_\tau p_\tau \phi_\tau$. This method
        takes the difference in weighted averages between the reference and comparison corpora.

        When the `reference_score` is 0, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau = \phi^{(C)}_\tau p_\tau^{(C)} - \phi^{(R)}_\tau p_\tau^{(R)}
        $$

        where $\phi_\tau^{(C)}$ is the score of token $\tau$ in the comparison corpus, and
        $p_\tau^{(C)}$ is its normalized frequency. Similarly, $\phi_\tau^{(R)}$ and $p_\tau^{(R)}$
        are the score and normalized frequency of $\tau$ in the reference corpus. Positive
        contributions identify tokens that have a higher score-weighted normalized frequency in the
        comparison than the reference.

        If the `reference_score` is non-zero, and the score of a word is the same in both corpora,
        i.e. $\phi_\tau = \phi^{(C)}_\tau = \phi^{(R)}_\tau$, then a token's contribution is:

        $$
        \delta_\tau =
            \left(
                p_\tau^{(C)} - p_\tau^{(R)}
            \right)
            \left(
                \phi_\tau - \Phi^{(\text{ref})}
            \right)
        $$

        where $\Phi^{(\text{ref})}$ is the user-provided reference score. If the `reference_score`
        is non-zero, and the score of a word is corpus dependent, then a token's contribution is:

        $$
        \delta_\tau =
        \left(
            p_\tau^{(C)} - p_\tau^{(R)}
        \right)
        \left[
            \frac{1}{2}
            \left(
                \phi_\tau^{(C)} + \phi_\tau^{(R)}
            \right)
            - \Phi^{(\text{ref})}
        \right]
        +
        \frac{1}{2}
        \left(
            p_\tau^{(C)} + p_\tau^{(R)}
        \right)
        \left(
            \phi_\tau^{(C)} - \phi_\tau^{(R)}
        \right)
        $$

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        Args:
            lexicon:
                Lexicon associating tokens with scores. Accepts a mapping `{token: score}` or a
                Polars dataframe with columns indicating the tokens and scores. Also accepts a
                [pre-defined lexicon][wordlevel.core.lexicons.load.Lexicon]
            lexicon_reference:
                Lexicon associating tokens with scores for the reference corpus
            lexicon_comparison:
                Lexicon associating tokens with scores foro the comparison corpus
            lexicon_token_col:
                Name of the token column when a dataframe is supplied for the lexicon
            lexicon_score_col:
                Name of the score column when a dataframe is supplied for the lexicon
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to the
                "center" of a lexicon's numerical scale when a `Lexicon` is supplied; otherwise it
                defaults to 0
            impute:
                How to score tokens when a score exists in one lexicon for the reference or
                comparison but not the other. `"borrow"` applies the score from the other corpus.
                `"mean"` uses the mean score of the corpus's lexicon (not weighted by frequency)

        Returns:
            A new `Comparison` scored by the difference in lexicon-weighted averages
        """
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

        def plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str, normalization_suffix: str
        ) -> pl.DataFrame | pl.LazyFrame:
            df = lexicon_join_plan(
                df,
                df_token_col=token_col,
                lexicon=lexicon,
                lexicon_ref=lexicon_reference,
                lexicon_cmp=lexicon_comparison,
                lexicon_token_col=lexicon_token_col,
                lexicon_score_col=lexicon_score_col,
                score_alias_ref=score_alias_ref,
                score_alias_cmp=score_alias_cmp,
                impute=impute,
                drop_null_scores=True,
            )

            return score_plan(df, token_col=token_col, normalization_suffix=normalization_suffix)

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
        r"""
        Calculates the difference in Tsallis entropy between the reference and comparison corpora.

        Tsallis entropy is a generalized measure of textual diversity, parameterized by an
        order $\alpha$ that interpolates between counting the number of non-zero tokens ($\alpha
        \to 0$) and measuring the most frequent token ($\alpha \to \infty$). At $\alpha = 1$, it
        reduces to the [Shannon entropy][wordlevel.api.comparison.score.Scorer.shannon_entropy].

        When the `reference_score` is 0, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            - \frac{\left( p_\tau^{(C)} \right)^\alpha}{\alpha - 1}
            + \frac{\left( p_\tau^{(R)} \right)^\alpha}{\alpha - 1}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora.

        When the `reference_score` is non-zero, the contribution of the token is:

        $$
        \delta_\tau =
        \left(
            p_\tau^{(C)} - p_\tau^{(R)}
        \right)
        \left[
            \frac{1}{2}
            \left(
                \phi_\tau^{(C)} + \phi_\tau^{(R)}
            \right)
            - \Phi^{(\text{ref})}
        \right]
        +
        \frac{1}{2}
        \left(
            p_\tau^{(C)} + p_\tau^{(R)}
        \right)
        \left(
            \phi_\tau^{(C)} - \phi_\tau^{(R)}
        \right)
        $$

        where $\Phi^{(\text{ref})}$ is the reference score, and

        $$
            \phi_\tau^{(C)} = - \frac{\left( p_\tau^{(C)} \right)^{\alpha - 1}}{\alpha - 1}
            \, , \,
            \phi_\tau^{(R)} = - \frac{\left( p_\tau^{(R)} \right)^{\alpha - 1}}{\alpha - 1}
        $$

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        Args:
            alpha:
                Order of the Tsallis entropy. $\alpha = 1$ recovers Shannon entropy; larger
                values place more weight on high-probability tokens, smaller values place distribute
                weight among all tokens more equally
            base:
                Base of the logarithms used to calculate the entropy when $\alpha = 1$. Ignored
                otherwise
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0

        Returns:
            A new `Comparison` scored by the difference in Tsallis entropies
        """
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
        r"""
        Calculates the difference in Shannon entropy between the reference and comparison corpora.

        Shannon entropy is a measure of textual diversity and "surprise." If $p_\tau$ is the
        normalized frequency of a token $\tau$ ina  corpus, then its Shannon entropy is
        $\sum_\tau p_\tau \log \frac{1}{p_\tau}$. The factor $\log \frac{1}{p_\tau}$ is known as
        the "surprisal" of a token: the less frequent a token, the higher its surprisal.

        When the `reference_score` is 0, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            p_\tau^{(C)} \log \frac{1}{p_\tau^{(C)}}
            - p_\tau^{(R)} \log \frac{1}{p_\tau^{(R)}}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora. Positive contributions
        identify tokens that are relatively more "surprising" in the comparison than the reference.

        When the `reference_score` is non-zero, the contribution of the token is:

        $$
        \delta_\tau =
        \left(
            p_\tau^{(C)} - p_\tau^{(R)}
        \right)
        \left[
            \frac{1}{2}
            \left(
                \phi_\tau^{(C)} + \phi_\tau^{(R)}
            \right)
            - \Phi^{(\text{ref})}
        \right]
        +
        \frac{1}{2}
        \left(
            p_\tau^{(C)} + p_\tau^{(R)}
        \right)
        \left(
            \phi_\tau^{(C)} - \phi_\tau^{(R)}
        \right)
        $$

        where $\Phi^{(\text{ref})}$ is the reference score, and

        $$
            \phi_\tau^{(C)} = \log \frac{1}{p_\tau^{(C)}}
            \, , \,
            \phi_\tau^{(R)} = \log \frac{1}{p_\tau^{(R)}}
        $$

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        Args:
            base:
                Base of the logarithms used to calculate the entropy
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0

        Returns:
            A new `Comparison` scored by the difference in Shannon entropies
        """
        return self.tsallis_entropy(alpha=1, base=base, reference_score=reference_score)

    def kullback_leibler_divergence(self, base: float = 2) -> Comparison:
        r"""
        Calculates the Kullback-Leibler divergence (KLD) between the reference and comparison corpora.

        The KLD is an asymmetric measure of how two corpora differ. From an information-theoretic
        perspective, it measures how much more information is needed to encode comparison corpus
        when using the reference as the "optimal coding scheme." It is only well-defined when
        all of the words in the comparison also appear in the reference. Otherwise, the KLD diverges
        to infinity.

        The contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            p_\tau^{(C)} \log \frac{1}{p_\tau^{(R)}} - p_\tau^{(C)} \log \frac{1}{p_\tau^{C}}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora. The total divergence is
        always non-negative, but individual per-token contributions can be either sign. Positive
        contributions indicate tokens that appear more in the comparison corpus; negative
        contributions indicate tokens that appear more in the reference corpus.

        Args:
            base:
                Logarithm base. Defaults to 2 (divergence in bits)

        Returns:
            A new `Comparison` scored by the Kullback-Leibler divergence

        Raises:
            ValueError:
                If the comparison corpus contains any token absent from the reference
                corpus, which causes the KLD to be infinite. Filter to the intersection of tokens
                or use a symmetric measure like the Jensen-Shannon divergence.
        """
        # The reference score does not affect token contributions for the KLD.
        plan = self._make_weighted_avg_score_plan(
            score_func=partial(
                kullback_leibler_divergence_dual_dist_scores, base=base, normalize=False
            ),
            clean_score_func=kullback_leibler_divergence_clean_dual_dist_scores,
            attribute_by="score_sign",
            reference_score=0,
            requires_normed_freqs=True,
            share_component_freqs_by="comparison",
        )

        ref = self._comparison._ref
        cmp = self._comparison._cmp

        def validated_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str, normalization_suffix: str
        ) -> pl.DataFrame | pl.LazyFrame:
            ref_col = f"{ref}{normalization_suffix}"
            cmp_col = f"{cmp}{normalization_suffix}"
            diverges = (
                df.lazy()
                .select(((col(cmp_col) > 0) & (col(ref_col) == 0)).any().alias("diverges"))
                .collect()
                .item()
            )
            if diverges:
                raise ValueError(
                    f"Kullback-Leibler divergence is infinite: the comparison corpus "
                    f"({cmp!r}) contains at least one token absent from the reference "
                    f"corpus ({ref!r}). The KLD is not well-defined."
                )
            return plan(df, token_col, normalization_suffix)

        return self._return_comparison(
            validated_plan,
            method="kullback_leibler_divergence",
            parameters={"base": base},
            is_weighted_avg=True,
            is_nonnegative=False,
        )

    def kld(self, base: float = 2) -> Comparison:
        r"""
        Calculates the Kullback-Leibler divergence between the reference and comparison corpora.

        Alias for
        [`kullback_leibler_divergence`][wordlevel.api.comparison.score.Scorer.kullback_leibler_divergence];
        see that method for more detials.

        Args:
            base:
                Logarithm base. Defaults to 2 (divergence in bits)

        Returns:
            A new `Comparison` scored by the Kullback-Leibler divergence
        """
        return self.kullback_leibler_divergence(base=base)

    def jensen_shannon_divergence(
        self,
        alpha: float = 1,
        base: float = 2,
        weight_ref: float = 0.5,
        weight_cmp: float = 0.5,
        reference_score: float | Literal["mean"] | pl.Expr = 0,
    ) -> Comparison:
        r"""
        Calculates the Jensen-Shannon divergence between the reference and comparison corpora.

        Jensen-Shannon divergence is a symmetric, bounded measure of how two distributions
        differ, built by averaging two KL divergences against the mixture distribution. Unlike
        [`kullback_leibler_divergence`][wordlevel.api.comparison.score.Scorer.kullback_leibler_divergence],
        JSD is symmetric in $R$ and $C$, is always non-negative, and each per-token contribution
        is also non-negative.

        The Jensen-Shannon divergence between $R$ and $C$ is:

        $$
        D_{\text{JS}}(R, C)
            = \pi_R \, D_{\text{KL}}(R \,\|\, M) + \pi_C \, D_{\text{KL}}(C \,\|\, M)
        $$

        where $M = \pi_R \, p^{(R)} + \pi_C \, p^{(C)}$ is the mixture distribution and
        $\pi_R, \pi_C$ are the mixture weights (defaulting to $\pi_R = \pi_C = 1/2$). Each
        token's contribution is $\geq 0$, so JSD naturally ranks tokens by how much they pull
        the two corpora apart.

        Args:
            alpha:
                Tsallis order generalizing the underlying divergences. $\alpha = 1$ recovers the
                standard, Shannon-based JSD
            base:
                Base of the logarithms used to calculate the JSD
            weight_ref:
                Mixture weight $\pi_R$ for the reference corpus
            weight_cmp:
                Mixture weight $\pi_C$ for the comparison corpus
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0

        Returns:
            A new `Comparison` scored by the Jensen-Shannon divergence
        """
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
        r"""
        Calculates the Jensen-Shannon divergence between the reference and comparison corpora.

        Alias for
        [`jensen_shannon_divergence`][wordlevel.api.comparison.score.Scorer.jensen_shannon_divergence];
        see that method for more details.

        Args:
            alpha:
                Tsallis order generalizing the underlying divergences. $\alpha = 1$ recovers
                standard JSD
            base:
                Base of the logarithms used to calculate the JSD
            weight_ref:
                Mixture weight $\pi_R$ for the reference corpus
            weight_cmp:
                Mixture weight $\pi_C$ for the comparison corpus
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0

        Returns:
            A new `Comparison` scored by the Jensen-Shannon divergence
        """
        return self.jensen_shannon_divergence(
            alpha=alpha,
            base=base,
            weight_ref=weight_ref,
            weight_cmp=weight_cmp,
            reference_score=reference_score,
        )

    def rank_divergence(self, alpha: float = 1.0) -> Comparison:
        r"""
        Calculates the rank-turbulence divergence between the reference and comparison corpora.

        Rank-turbulence divergence compares corpora by token rank instead of frequency. Each
        token's contribution is the absolute difference of inverse-rank powers, summed over the
        union of tokens in $R$ and $C$. The measure is symmetric in $R$ and $C$ and robust to
        differences in overall corpus size.

        The rank divergence between $R$ and $C$ at order $\alpha$ is:

        $$
        D^R_\alpha(R, C)
            = \frac{1}{N_\alpha}
                \sum_\tau
                \left|
                    \frac{1}{\left[r_\tau^{(R)}\right]^\alpha}
                    - \frac{1}{\left[r_\tau^{(C)}\right]^\alpha}
                \right|
        $$

        where $r_\tau^{(X)}$ is the rank of token $\tau$ in corpus $X$ (rank 1 being the most
        frequent) and $N_\alpha$ is a normalization constant that makes divergences comparable
        across values of $\alpha$. Each per-token contribution is non-negative.

        The parameter $\alpha$ controls which ranks dominate:

        - $\alpha \to 0$: the per-token contribution approaches
          $\left|\ln r_\tau^{(R)} - \ln r_\tau^{(C)}\right|$, giving equal weight to changes in
          rank order across the full distribution and surfacing rare tokens at the long tail
        - $\alpha = 1$: standard inverse-rank weighting
        - $\alpha \to \infty$: contributions are dominated by the top-ranked tokens and the
          divergence reduces to comparing the heads of the two corpora

        Args:
            alpha:
                Order of the rank divergence. Defaults to 1.0

        Returns:
            A new `Comparison` scores by the rank divergence
        """
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
