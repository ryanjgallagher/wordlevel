from functools import cached_property
from typing_extensions import Self

from wordlevel.compare.constructors import NORMALIZATION_SUFFIX
from wordlevel.compare.comparisons.score import Scorer
from wordlevel.compare.comparisons.types import ScorePlan


class Comparison:
    """
    tc = (
        Comparator(data=data, subsets=['1990', '2000', '2010'])
        .with_comparsions(
            comp('1990', '2000').score.rank_divergence().filter(min_score=20).alias('my_rank_div'),
            comp('2000', '2010').score.jensen_shannon()
        )
        .plot.bars('my_rank_div')
    )
    """

    def __init__(
        self,
        reference: str,
        comparison: str,
        alias: str | None = None,
        method: str | None = None,
        is_weighted_avg: bool | None = None,
        plan: ScorePlan | None = None,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
        parameters: dict | None = None,
    ):
        self._ref = reference
        self._cmp = comparison
        self._alias = alias
        self._method = method
        self._is_weighted_avg = is_weighted_avg
        self._plan = plan
        self._normalization_suffix = normalization_suffix

        if parameters is None:
            self._parameters = dict()
        else:
            self._parameters = parameters.copy()

    @property
    def _ref_normed(self) -> str:
        return f"{self._ref}{self._normalization_suffix}"

    @property
    def _cmp_normed(self) -> str:
        return f"{self._cmp}{self._normalization_suffix}"

    @cached_property
    def score(self) -> Scorer:
        """ """
        return Scorer(self)

    def alias(self, name: str) -> Self:
        """Return a new Comparison with the specified alias."""
        return self.__class__(
            reference=self._ref,
            comparison=self._cmp,
            alias=name,
            method=self._method,
            is_weighted_avg=self._is_weighted_avg,
            plan=self._plan,
            normalization_suffix=self._normalization_suffix,
            parameters=self._parameters,
        )


def comp(reference: str, comparison: str) -> Comparison:
    """ """
    return Comparison(reference=reference, comparison=comparison)
