from dataclasses import dataclass
from typing import cast

import altair as alt
import polars as pl
from typing_extensions import Self

from wordlevel.configs import BorderConfig, RankAxisConfig, ScoreAxisConfig
from wordlevel.plots.core.utils import to_altair_optional


_TRANSLATE_PX = 0.5


@dataclass
class InternalBorderConfig(BorderConfig):
    """
    Internal configuration of which border edges to draw and how.

    Attributes:
        draw_score_axis:
            Whether to draw the score axis border
        draw_rank_axis:
            Whether to draw the rank axis border
        draw_border_opposite_score_axis:
            Whether to draw the border opposite the score axis
        draw_border_opposite_rank_axis:
            Whether to draw the border opposite the rank axis
    """

    draw_score_axis: bool = True
    draw_rank_axis: bool = True
    draw_border_opposite_score_axis: bool = False
    draw_border_opposite_rank_axis: bool = False

    @classmethod
    def from_config(
        cls,
        config: BorderConfig,
        draw_score_axis: bool = True,
        draw_rank_axis: bool = True,
        draw_border_opposite_score_axis: bool = False,
        draw_border_opposite_rank_axis: bool = False,
        width: float | None = None,
    ) -> Self:
        """
        Derives an internal config from a user-facing border config

        Args:
            config:
                A user-facing config for the border
            draw_score_axis:
                Whether to draw the score axis border
            draw_rank_axis:
                Whether to draw the rank axis border
            draw_border_opposite_score_axis:
                Whether to draw the border opposite the score axis
            draw_border_opposite_rank_axis:
                Whether to draw the border opposite the rank axis
            width:
                Width of the drawn edges, in pixels. Overrides `config.width` when given, letting a
                caller draw an edge that the user-facing border width would otherwise leave at zero
        """
        return cls(
            color=config.color,
            width=config.width if width is None else width,
            draw_score_axis=draw_score_axis,
            draw_rank_axis=draw_rank_axis,
            draw_border_opposite_score_axis=draw_border_opposite_score_axis,
            draw_border_opposite_rank_axis=draw_border_opposite_rank_axis,
        )


def shift_borders(
    charts: list[alt.Chart | alt.LayerChart],
    score_col: str,
    rank_col: str,
    rank_axis_config: RankAxisConfig,
    score_axis_config: ScoreAxisConfig,
    border_config: InternalBorderConfig,
    vertical: bool = True,
) -> alt.LayerChart:
    """
    Draws borders around a set of provided charts by drawing them as their own axes.

    The "border" consists of the non-axis edges, i.e. those that are opposite the axes. Drawing the
    border as axes gives more control over how they appear, and helps ensure that they are aligned
    perfectly with the main axes and any ticks

    Args:
        charts:
            The charts around which to draw borders
        score_col:
            Column in the dataframe indicating the "score" associated with each bar, i.e. its length
        rank_col:
            Column in the dataframe indicating the rank of each token, taking into account all of
            its contributing scores
        rank_axis_config:
            Configuration for the rank axis. Its `color` overrides `border_config.color`
        score_axis_config:
            Configuration for the score axis. Its `color` overrides `border_config.color`
        border_config:
            Configuration for the border
        vertical:
            Whether to orient the chart vertically, where tokens are listed top-to-bottom and bars
            extend horizontally. If `False`, tokens are listed left-to-right and bars extend
            vertically

    Returns:
        A `LayerChart` of the provided charts with the newly created border "charts." `chart[0]` is
        a `LayerChat` of the original `charts`, where as `chart[1:]` consists of the borders
    """
    dim2params = {
        rank_col: {
            "field": f"{rank_col}:O",
            "color": rank_axis_config.axis_color
            if rank_axis_config.axis_color is not None
            else border_config.color,
            "orient": "right" if vertical else "top",
        },
        score_col: {
            "field": f"{score_col}:Q",
            "color": score_axis_config.axis_color
            if score_axis_config.axis_color is not None
            else border_config.color,
            "orient": "top" if vertical else "right",
        },
    }

    dimension_to_draw: list[str] = []
    if border_config.draw_border_opposite_rank_axis:
        dimension_to_draw.append(rank_col)
    if border_config.draw_border_opposite_score_axis:
        dimension_to_draw.append(score_col)

    borders: list[alt.Chart] = []
    for dim in dimension_to_draw:
        axis = alt.Axis(
            domain=True,
            domainColor=to_altair_optional(dim2params[dim]["color"]),
            domainWidth=border_config.width,
            orient=dim2params[dim]["orient"],  # type: ignore[arg-type]
            title=None,
            titlePadding=0,
            labels=False,
            labelPadding=0,
            ticks=False,
            tickSize=0,
            grid=False,
            offset=0,
            minExtent=0,
            maxExtent=0,
            translate=_TRANSLATE_PX,
        )

        if dim2params[dim]["orient"] == "right":
            encoding = {"y": alt.Y(dim2params[dim]["field"], axis=axis)}
        else:
            encoding = {"x": alt.X(dim2params[dim]["field"], axis=axis)}

        borders.append(
            alt.Chart(pl.DataFrame(schema={rank_col: pl.Int64, score_col: pl.Float64}))
            .mark_point(opacity=0)
            .encode(**encoding, tooltip=alt.value(None))  # type: ignore[arg-type]
        )

    # Nest the base layers into one chart before layering the border edges on top so that the `Step`
    # sizing of the main charts is not affected by the newly drawn borders
    # The axes need to be resolved so that the new border "charts" are rendered independently
    chart_with_border = alt.layer(alt.layer(*charts), *borders).resolve_axis(
        x="independent", y="independent"
    )

    return cast(alt.LayerChart, chart_with_border)
