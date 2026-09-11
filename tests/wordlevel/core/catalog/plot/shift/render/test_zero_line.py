import pytest

from wordlevel.configs import BorderConfig, ZeroLineConfig
from wordlevel.core.catalog.plot.shift.render.border import _TRANSLATE_PX
from wordlevel.core.catalog.plot.shift.render.shift import shift, zero_line


@pytest.mark.parametrize("vertical, channel", [(True, "x"), (False, "y")])
def test_zero_line_orientation(vertical, channel):
    """
    The zero line is a rule at datum 0 on the score axis (x when vertical, y when horizontal). The
    rank-axis channel is left unencoded so the rule spans the full extent of that axis.
    """
    spec = zero_line(ZeroLineConfig(width=1), vertical=vertical).to_dict()

    assert spec["mark"]["type"] == "rule"
    assert spec["encoding"][channel]["datum"] == 0

    other = "y" if channel == "x" else "x"
    assert other not in spec["encoding"]


@pytest.mark.parametrize("vertical, offset_key", [(True, "xOffset"), (False, "yOffset")])
def test_zero_line_carries_score_translate_offset(vertical, offset_key):
    """
    The score axis (and its zero tick) is nudged a half pixel for crisp registration with the frame.
    The zero line is a mark, not an axis, so it must carry the same offset to land on the zero tick.
    """
    mark = zero_line(ZeroLineConfig(width=1), vertical=vertical).to_dict()["mark"]

    assert mark[offset_key] == _TRANSLATE_PX


def test_zero_line_width_and_color():
    """The width and an explicit color pass through to the rule mark."""
    mark = zero_line(ZeroLineConfig(color="red", width=3)).to_dict()["mark"]

    assert mark["strokeWidth"] == 3
    assert mark["color"] == "red"


def test_zero_line_color_falls_back_to_border():
    """When the zero line has no color it uses the border color; an explicit color overrides it."""
    fallback = zero_line(ZeroLineConfig(width=1), border=BorderConfig(color="black"))
    assert fallback.to_dict()["mark"]["color"] == "black"

    override = zero_line(ZeroLineConfig(color="red", width=1), border=BorderConfig(color="black"))
    assert override.to_dict()["mark"]["color"] == "red"


def test_zero_line_dict_form_and_default():
    """
    Dict configs are accepted; the default falls back to the border color (Altair's axis grey) at
    width 1.0
    """
    mark = zero_line({"color": "blue", "width": 2}).to_dict()["mark"]
    assert mark["color"] == "blue"
    assert mark["strokeWidth"] == 2

    default = zero_line().to_dict()["mark"]
    assert default["strokeWidth"] == 1.0
    assert default["color"] == "#888"


def test_shift_appends_zero_line(make_mock_df, score_col, rank_col, token_col, score_type_col):
    """
    shift appends the zero line as the final layer, invisible (width 0) by default, and honors a
    provided ZeroLineConfig.
    """
    default_chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        width=100,
    )
    # The zero line is the last of the base layers grouped in layer[0] (the border edges follow)
    default_last = default_chart.layer[0].layer[-1].to_dict()
    assert default_last["mark"]["type"] == "rule"
    assert default_last["encoding"]["x"]["datum"] == 0
    assert default_last["mark"]["strokeWidth"] == 1.0

    configured_chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        width=100,
        zero_line_config=ZeroLineConfig(color="red", width=2),
    )
    configured_last = configured_chart.layer[0].layer[-1].to_dict()
    assert configured_last["mark"]["color"] == "red"
    assert configured_last["mark"]["strokeWidth"] == 2
