from __future__ import annotations
from functools import partial
from typing import Any, Literal, Mapping, TYPE_CHECKING

import polars as pl
from polars import col

from wordlevel.configs import SHIFT_BAR_NAMES
from wordlevel.constants import (
    SCORE_COL,
    RANK_COL,
    ATTRIBUTION_COL,
    COUNTERACTED_SUFFIX,
    FREQ_CMP_COL,
    FREQ_REF_COL,
    IS_BORROWED_COL,
    LEXICON_SCORE_CMP_COL,
    LEXICON_SCORE_REF_COL,
    NORMED_FREQ_CMP_COL,
    NORMED_FREQ_REF_COL,
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
    from wordlevel.api.comparison.state import PlanStage


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
        state = self._comparison._state

        return f"reference:{state.reference}_comparison:{state.comparison}_method:{method}"

    def _make_basic_score_plan(
        self,
        score_func: ScoreFunc,
        attribute_by: ATTRIBUTION_TYPE,
        requires_normed_freqs: bool = True,
    ) -> PlanStage:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison
        """
        if requires_normed_freqs:
            ref_col = NORMED_FREQ_REF_COL
            cmp_col = NORMED_FREQ_CMP_COL
        else:
            ref_col = FREQ_REF_COL
            cmp_col = FREQ_CMP_COL

        def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
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
        passthrough_cols: list[str] | None = None,
    ) -> PlanStage:
        """
        Creates a set of instructions (i.e. a function) for how to score a comparison when the
        scoring is done using a weighted average method
        """
        if requires_normed_freqs:
            freqs_ref = NORMED_FREQ_REF_COL
            freqs_cmp = NORMED_FREQ_CMP_COL
        else:
            freqs_ref = FREQ_REF_COL
            freqs_cmp = FREQ_CMP_COL

        component_freqs_ref = freqs_ref
        component_freqs_cmp = freqs_cmp
        if share_component_freqs_by == "reference":
            component_freqs_cmp = component_freqs_ref
        elif share_component_freqs_by == "comparison":
            component_freqs_ref = component_freqs_cmp

        def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
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
                passthrough_cols=passthrough_cols,
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
            )

        return plan

    def _return_comparison(
        self,
        score_plan: PlanStage,
        method: str,
        parameters: dict[str, Any],
        is_weighted_avg: bool,
        is_nonnegative: bool,
        attributed_by: ATTRIBUTION_TYPE,
        lexicon_join_plan: PlanStage | None = None,
    ) -> Comparison:
        """ """
        # TODO: `lexicon_join_plan` shadows the core function of the same name imported above, fix
        state = self._comparison._state

        if state.alias is None:
            alias = self._get_default_alias(method)
        else:
            alias = state.alias

        return self._comparison._update_state(
            alias=alias,
            method=method,
            parameters=parameters,
            is_weighted_avg=is_weighted_avg,
            is_nonnegative=is_nonnegative,
            attributed_by=attributed_by,
            lexicon_join_plan=lexicon_join_plan,
            score_plan=score_plan,
        )

    def frequency(self) -> Comparison:
        r"""
        Calculates the frequency difference between the reference and comparison corpora.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:frequency"
            ```

        Each token's contribution is the difference in its raw frequency between the corpora.
        Specifically, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau = f_\tau^{(C)} - f_\tau^{(R)}
        $$

        where $f_\tau^{(R)}$ and $f_\tau^{(C)}$ are the raw frequencies of token $\tau$ in $R$ and
        $C$ respectively, the reference and comparison corpora. Positive contributions identify
        tokens that occur more often in the comparison; negative contributions identify tokens that
        occur more often in the reference.

        See Also:
            | Score | Description |
            |-------|-------------|
            | [`Comparison.score.proportion`][wordlevel.api.comparison.score.Scorer.proportion] | Scores corpora by normalized frequencies |
            | [`Comparison.score.shannon_entropy`][wordlevel.api.comparison.score.Scorer.shannon_entropy] | Scores corpora by normalized frequencies weighted by surprisals |

        Returns:
            A new `Comparison` with the corpora scored by frequency differences
        """
        plan = self._make_basic_score_plan(
            frequency_difference,
            attribute_by="frequency",
            requires_normed_freqs=False,
        )

        return self._return_comparison(
            plan,
            method="frequency",
            parameters=dict(),
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
        )

    def proportion(self, reference_score: float | Literal["mean"] | pl.Expr = 0) -> Comparison:
        r"""
        Calculates the proportion difference between the reference and comparison corpora.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:proportion"
            ```

        Each token's contribution is the difference in its normalized frequency (i.e. proportion)
        within each corpus. When the `reference_score` is 0 (the default), the contribution
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

        See Also:
            | Score | Description |
            |-------|-------------|
            | [`Comparison.score.frequency`][wordlevel.api.comparison.score.Scorer.frequency] | Scores corpora by raw frequencies |
            | [`Comparison.score.shannon_entropy`][wordlevel.api.comparison.score.Scorer.shannon_entropy] | Scores corpora by normalized frequencies weighted by surprisals |

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
            attributed_by="frequency",
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
        the lexicon, then the corpus-level score is $\Phi = \sum_\tau p_\tau \phi_\tau$. This method
        takes the difference in weighted averages between the reference and comparison corpora,
        $\Phi^{(C)} - \Phi^{(R)}$.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:lexicon"
            ```

        When the `reference_score` is 0, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau = \phi^{(C)}_\tau p_\tau^{(C)} - \phi^{(R)}_\tau p_\tau^{(R)}
        $$

        where $\phi_\tau^{(C)}$ is the score of token $\tau$ in the comparison corpus, and
        $p_\tau^{(C)}$ is its normalized frequency. Similarly, $\phi_\tau^{(R)}$ and $p_\tau^{(R)}$
        are the score and normalized frequency of $\tau$ in the reference corpus.

        Each token is interpreted by whether it reinforces or offsets the corpus-level difference in
        weighted averages $\Phi^{(C)} - \Phi^{(R)}$. A contribution with the same sign as the overall
        difference reinforces it, making its magnitude larger; a contribution with the opposite sign
        offsets it, pulling the difference back toward zero. For example, if
        $\Phi^{(C)} - \Phi^{(R)} > 0$ (the comparison's weighted average is higher), positive
        contributions reinforce the difference and negative contributions offset it; the
        interpretation flips when $\Phi^{(C)} - \Phi^{(R)} < 0$.

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
                Lexicon associating tokens with scores for the comparison corpus
            lexicon_token_col:
                Name of the token column when a dataframe is supplied for the lexicon
            lexicon_score_col:
                Name of the score column when a dataframe is supplied for the lexicon
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to the
                "center" of a lexicon's numerical scale when a `Lexicon` is supplied; otherwise it
                defaults to 0. If `"mean"`, uses the weighted average of the reference corpus as
                the reference score
            impute:
                How to score tokens when a score exists in one lexicon for the reference or
                comparison but not the other. `"borrow"` applies the score from the other corpus.
                `"mean"` uses the mean score of the corpus's lexicon (not weighted by frequency). A
                float or Polars expression can also be passed to use a specific value as the imputed
                score

        Returns:
            A new `Comparison` scored by the difference in lexicon-weighted averages

        Raises:
            ValueError:
                If no lexicon is supplied (`lexicon`, `lexicon_reference`, and `lexicon_comparison`
                are all `None`).
            ValueError:
                If `lexicon` is supplied alongside `lexicon_reference` or `lexicon_comparison` (the
                single-lexicon and per-corpus-lexicon paths are mutually exclusive).
        """
        score_alias_ref = LEXICON_SCORE_REF_COL
        score_alias_cmp = LEXICON_SCORE_CMP_COL

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

        passthrough_cols = None
        if isinstance(impute, str) and impute == "borrow":
            passthrough_cols = [IS_BORROWED_COL]

        score_plan = self._make_weighted_avg_score_plan(
            score_func=_fixed_dual_score_func(col(score_alias_ref), col(score_alias_cmp)),
            clean_score_func=_passthrough_clean_score_func,
            attribute_by="score_sign",
            reference_score=reference_score,
            requires_normed_freqs=True,
            passthrough_cols=passthrough_cols,
        )

        def join_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            return lexicon_join_plan(
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

        return self._return_comparison(
            score_plan,
            method="lexicon",
            parameters={"reference_score": reference_score, "impute": impute},
            is_weighted_avg=True,
            is_nonnegative=False,
            attributed_by="score_sign",
            lexicon_join_plan=join_plan,
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
        order $\alpha$. When $\alpha = 1$, the Tsallis entropy reduces to the [Shannon
        entropy][wordlevel.api.comparison.score.Scorer.shannon_entropy]; when $\alpha < 1$, more
        weight is given to rare tokens; and when $\alpha > 1$, more weight is given to common
        tokens.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:tsallis_entropy"
            ```

        When the `reference_score` is 0, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            \frac{p_\tau^{(C)} - \left[ p_\tau^{(C)} \right]^\alpha}{\alpha - 1}
            - \frac{p_\tau^{(R)} - \left[ p_\tau^{(R)} \right]^\alpha}{\alpha - 1}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora.

        Each token is interpreted by whether it reinforces or offsets the corpus-level difference in
        entropies $\Phi^{(C)} - \Phi^{(R)}$. A contribution with the same sign as the overall
        difference reinforces it, making its magnitude larger; a contribution with the opposite sign
        offsets it, pulling the difference back toward zero. For example, if
        $\Phi^{(C)} - \Phi^{(R)} > 0$ (the comparison's entropy is higher), positive contributions
        reinforce the difference and negative contributions offset it; the interpretation flips when
        $\Phi^{(C)} - \Phi^{(R)} < 0$.

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

        where $\Phi^{(\text{ref})}$ is the user-provided reference score, and

        $$
            \phi_\tau^{(C)} = \frac{1 - \left[ p_\tau^{(C)} \right]^{\alpha - 1}}{\alpha - 1}
            \, , \,
            \phi_\tau^{(R)} = \frac{1 - \left[ p_\tau^{(R)} \right]^{\alpha - 1}}{\alpha - 1}
        $$

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        See Also:
            | Score | Description |
            |-------|-------------|
            | [`Comparison.score.shannon_entropy`][wordlevel.api.comparison.score.Scorer.shannon_entropy] | Scores corpora in the special case when $\alpha=1$ |

        Args:
            alpha:
                Order of the Tsallis entropy. $\alpha = 1$ recovers Shannon entropy. Larger
                values place more weight on common tokens, and smaller values place more weight
                on rare tokens
            base:
                Base of the logarithms used to calculate the entropy when $\alpha = 1$. Ignored
                otherwise
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0.
                If `"mean"`, uses the Tsallis entropy of the reference corpus as the reference score

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
            attributed_by="score_sign",
        )

    def shannon_entropy(
        self, base: float = 2, reference_score: float | Literal["mean"] | pl.Expr = 0
    ) -> Comparison:
        r"""
        Calculates the difference in Shannon entropy between the reference and comparison corpora.

        Shannon entropy is a measure of textual diversity and "surprise." If $p_\tau$ is the
        normalized frequency of a token $\tau$ in a corpus, then its Shannon entropy is
        $\sum_\tau p_\tau \log \frac{1}{p_\tau}$. The factor $\log \frac{1}{p_\tau}$ is known as
        the "surprisal" of a token: the less frequent a token, the higher its surprisal.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:shannon_entropy"
            ```

        When the `reference_score` is 0, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            p_\tau^{(C)} \log \frac{1}{p_\tau^{(C)}}
            - p_\tau^{(R)} \log \frac{1}{p_\tau^{(R)}}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora.

        Each token is interpreted by whether it reinforces or offsets the corpus-level difference in
        entropies $\Phi^{(C)} - \Phi^{(R)}$. A contribution with the same sign as the overall
        difference reinforces it, making its magnitude larger; a contribution with the opposite sign
        offsets it, pulling the difference back toward zero. For example, if
        $\Phi^{(C)} - \Phi^{(R)} > 0$ (the comparison's entropy is higher), positive contributions
        reinforce the difference and negative contributions offset it; the interpretation flips when
        $\Phi^{(C)} - \Phi^{(R)} < 0$.

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

        where $\Phi^{(\text{ref})}$ is the user-provided reference score, and

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

        See Also:
            | Score | Description |
            |-------|-------------|
            | [`Comparison.score.tsallis_entropy`][wordlevel.api.comparison.score.Scorer.tsallis_entropy] | Scores corpora using generalized entropy |

        Args:
            base:
                Base of the logarithms used to calculate the entropy
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0. If
                `"mean"`, uses the Shannon entropy of the reference corpus as the reference score

        Returns:
            A new `Comparison` scored by the difference in Shannon entropies
        """
        return self.tsallis_entropy(alpha=1, base=base, reference_score=reference_score)

    def kullback_leibler_divergence(self, base: float = 2) -> Comparison:
        r"""
        Calculates the Kullback-Leibler divergence (KLD) between the reference and comparison corpora.

        The KLD is an asymmetric measure of how two corpora differ. From an information-theoretic
        perspective, it measures how much more information is needed to encode the comparison corpus
        when using the reference as the "codebook." It is only well-defined when all of the words in
        the comparison also appear in the reference. Otherwise, the KLD diverges to infinity.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:kullback_leibler_divergence"
            ```

        Warning:
            Using this measure forces the comparison to be evaluated eagerly rather than lazily,
            because validating it requires computing over the full vocabulary.

        The contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            p_\tau^{(C)} \log \frac{1}{p_\tau^{(R)}} - p_\tau^{(C)} \log \frac{1}{p_\tau^{(C)}}
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora. The total divergence is
        always non-negative, but individual per-token contributions can be either sign. A positive
        contribution identifies a token that appears relatively more in the comparison corpus; a
        negative contribution, more in the reference. Because the total is non-negative, positive
        contributions reinforce the divergence and negative contributions offset it.

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        See Also:
            | Score | Description |
            |-------|-------------|
            | [`Comparison.score.jensen_shannon_divergence`][wordlevel.api.comparison.score.Scorer.jensen_shannon_divergence] | Scores corpora using a bounded, symmetric, entropy-based divergence |

        Args:
            base:
                Base of the logarithms used to calculate the KLD

        Returns:
            A new `Comparison` scored by the Kullback-Leibler divergence

        Raises:
            ValueError:
                If the comparison corpus contains any token absent from the reference
                corpus, which causes the KLD to be infinite. Filter to the intersection of tokens
                or use a symmetric measure like the [Jensen-Shannon
                divergence][wordlevel.api.comparison.score.Scorer.jensen_shannon_divergence]
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

        ref = self._comparison._state.reference
        cmp = self._comparison._state.comparison

        def validated_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            diverges = (
                df.lazy()
                .select(
                    ((col(NORMED_FREQ_CMP_COL) > 0) & (col(NORMED_FREQ_REF_COL) == 0))
                    .any()
                    .alias("diverges")
                )
                .collect()
                .item()
            )
            if diverges:
                raise ValueError(
                    f"Kullback-Leibler divergence is infinite: the comparison corpus "
                    f"({cmp!r}) contains at least one token absent from the reference "
                    f"corpus ({ref!r}). The KLD is not well-defined."
                )
            return plan(df, token_col)

        return self._return_comparison(
            validated_plan,
            method="kullback_leibler_divergence",
            parameters={"base": base},
            is_weighted_avg=True,
            is_nonnegative=False,
            attributed_by="score_sign",
        )

    def kld(self, base: float = 2) -> Comparison:
        r"""
        Calculates the Kullback-Leibler divergence (KLD) between the reference and comparison corpora.

        Alias for
        [`kullback_leibler_divergence`][wordlevel.api.comparison.score.Scorer.kullback_leibler_divergence];
        see that method for more details.

        Args:
            base:
                Base of the logarithms used to calculate the KLD

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
        Calculates the Jensen-Shannon divergence (JSD) between the reference and comparison corpora.

        The JSD is a symmetric, bounded measure of how two corpora differ. It is computed by
        averaging the Kullback-Leibler divergence of each corpus against a synthetic mixture corpus
        representing the combination of both corpora. This implementation generalizes the JSD
        through an order parameter $\alpha$. When $\alpha < 1$, more weight is given to rare tokens;
        when $\alpha > 1$, more weight is given to common tokens.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:jensen_shannon_divergence"
            ```

        When the `reference_score` is 0 and $\alpha = 1$, the contribution $\delta_\tau$ of a
        token $\tau$ is:

        $$
        \delta_\tau =
            p_\tau^{(M)} \log \frac{1}{p_\tau^{(M)}}
            -
            \left(
                \pi_R \, p_\tau^{(R)} \log \frac{1}{p_\tau^{(R)}}
                + \pi_C \, p_\tau^{(C)} \log \frac{1}{p_\tau^{(C)}}
            \right)
        $$

        where $p_\tau^{(R)}$ and $p_\tau^{(C)}$ are the normalized frequencies of token $\tau$ in
        $R$ and $C$ respectively, the reference and comparison corpora, and $p_\tau^{(M)} = \pi_R \,
        p_\tau^{(R)} + \pi_C \, p_\tau^{(C)}$ with weights $\pi_R$ and $\pi_C$ such that $\pi_R + \pi_C
        = 1$. For $\alpha = 1$, each token's contribution is always non-negative.

        When the `reference_score` is 0 and $\alpha \neq 1$, the contribution of the token is:

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

        where $\Phi^{(\text{ref})}$ is the user-provided reference score, and

        $$
            \phi_\tau^{(C)} =
                \pi_C
                \left(
                    \frac
                    {\left[ p_\tau^{(C)} \right]^{\alpha - 1} - \left[ p_\tau^{(M)} \right]^{\alpha - 1}}
                    {\alpha - 1}
                \right)
            \, , \,
            \phi_\tau^{(R)} =
                \pi_R
                \left(
                    \frac
                    {\left[ p_\tau^{(M)} \right]^{\alpha - 1} - \left[ p_\tau^{(R)} \right]^{\alpha - 1}}
                    {\alpha - 1}
                \right)
        $$

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        See Also:
            | Score | Description |
            |-------|-------------|
            | [`Comparison.score.kullback_leibler_divergence`][wordlevel.api.comparison.score.Scorer.kullback_leibler_divergence] | Scores corpora using an asymmetric entropy-based divergence |

        Args:
            alpha:
                Order of the generalized entropy. $\alpha = 1$ recovers the standard JSD. Larger
                values place more weight on common tokens, and smaller values place more weight
                on rare tokens
            base:
                Base of the logarithms used to calculate the JSD when $\alpha = 1$. Ignored
                otherwise
            weight_ref:
                Mixture weight $\pi_R$ for the reference corpus
            weight_cmp:
                Mixture weight $\pi_C$ for the comparison corpus
            reference_score:
                Baseline against which per-token contributions are interpreted. Defaults to 0. If
                `"mean"`, uses the weighted average of the reference corpus, where weights are the
                values described in the general formula above

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
            attributed_by="frequency",
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
        Calculates the Jensen-Shannon divergence (JSD) between the reference and comparison corpora.

        Alias for
        [`jensen_shannon_divergence`][wordlevel.api.comparison.score.Scorer.jensen_shannon_divergence];
        see that method for more details.

        Args:
            alpha:
                Order of the generalized entropy. $\alpha = 1$ recovers the standard JSD. Larger
                values place more weight on common tokens, and smaller values place more weight
                on rare tokens
            base:
                Base of the logarithms used to calculate the JSD when $\alpha = 1$. Ignored
                otherwise
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

        Rank-turbulence divergence compares corpora by comparing the ranks of tokens in each corpus
        when ordered by relative frequency. The measure is symmetric and bounded between 0 and 1,
        where 0 indicates that the corpora are identical, and 1 indicates that they are maximally
        divergent. It is also parameterized by an order parameter $\alpha$. When $\alpha < 1$, more
        weight is given to rare tokens; when $\alpha > 1$, more weight is given to common tokens.

        Example:
            ```python
            --8<-- "snippets/comparison/score.py:setup"

            --8<-- "snippets/comparison/score.py:rank_divergence"
            ```

        When $\alpha > 0$, the contribution $\delta_\tau$ of a token $\tau$ is:

        $$
        \delta_\tau =
            \frac{1}{\mathcal{N}_\alpha}
            \,
            \left|
                \frac{1}{\left[ r_\tau^{(R)} \right]^\alpha}
                - \frac{1}{\left[ r_\tau^{(C)} \right]^\alpha}
            \right|
            ^{ 1 / (\alpha + 1)}
        $$

        where $r_\tau^{(R)}$ and $r_\tau^{(C)}$ are the ranks of token $\tau$ in $R$ and $C$
        respectively, the reference and comparison corpora, and $\mathcal{N}_\alpha$ is a
        normalization constant based on $\alpha$.

        When $\alpha = 0$, the contribution of the token is:

        $$
        \delta_\tau =
            \frac{1}{N_0}
            \,
            \left|
                \ln \frac{r_\tau^{(R)}}{r_\tau^{(C)}}
            \right|
        $$

        Reference:
            For details, see the following paper:
            > Dodds, P. S., Minot, J. R., Arnold, M. V., Alshaabi, T., Adams, J. L.,
            Dewhurst, D. R., Gray, T. J., Frank, M. R., Reagan, A. J., & Danforth, C. M. (2023).
            [Allotaxonometry and rank-turbulence divergence: A universal instrument for comparing
            complex systems](https://link.springer.com/article/10.1140/epjds/s13688-023-00400-x).
            EPJ Data Science, 12(1), 37.

        Args:
            alpha:
                Order of the rank divergence. Larger values place more weight on common tokens, and
                smaller values place more weight on rare tokens

        Returns:
            A new `Comparison` scored by the rank-turbulence divergence
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
            attributed_by="rank",
        )
