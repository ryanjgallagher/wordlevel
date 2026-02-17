from typing import TYPE_CHECKING

import altair as alt

from wordlevel.plots.shift.bars import RankAxisConfig, ScoreAxisConfig, BarConfig
from wordlevel.plots.shift.labels import LabelConfig
from wordlevel.plots.shift.shift import shift

if TYPE_CHECKING:
    from wordlevel.compare.comparator import CorpusComparator


class Plotter:
    """ """

    def __init__(self, comparator: CorpusComparator):
        """ """
        self._comparator = comparator

    def shift(
        self,
        comparison: str,
        max_rank: int = 50,
        detailed: bool = False,
        vertical: bool = True,
        height: float | None = None,
        width: float | None = None,
        autoscale_axes: bool = True,
        rank_axis: RankAxisConfig | dict | None = None,
        score_axis: ScoreAxisConfig | dict | None = None,
        bar_config: BarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
        show_border: bool = False,
    ) -> alt.LayerChart:
        """ """
        comp = self._comparator.comparisons[comparison]

        if detailed:
            if comp.is_weighted_avg:
                # Pivot
                pass
            else:
                # TODO: make better error  message
                raise ValueError("Cannot make a detailed shift plot for non-weighted averages")
        else:
            # Annotate bars with types (e.g. positive, negative)
            pass

        # TODO: for semi-positive measures, is this where we should direct the bars? Or should it happen in the comparison plan?
        #       add an attribute to comparisons indicating if it's semi positive (like we do with weighted averages?)

        shift(df=comp._lf)
