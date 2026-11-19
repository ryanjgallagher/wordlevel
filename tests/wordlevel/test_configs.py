import pytest

from wordlevel.configs import (
    BarColorConfig,
    BarConfig,
    BorderConfig,
    LabelConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ShiftColorConfig,
    TotalsConfig,
    ZeroLineConfig,
    _resolve_config,
)


@pytest.mark.parametrize("config_class", [LabelConfig, BorderConfig, TotalsConfig])
def test_resolve_config_none_returns_defaults(config_class):
    assert _resolve_config(None, config_class) == config_class()


@pytest.mark.parametrize(
    "config_class, fields",
    [
        (RankAxisConfig, {"title": None}),
        (ZeroLineConfig, {"width": 2.0}),
        (ShiftColorConfig, {"counteracted_opacity": 0.5}),
    ],
)
def test_resolve_config_builds_from_dict(config_class, fields):
    assert _resolve_config(fields, config_class) == config_class(**fields)


@pytest.mark.parametrize("config", [ScoreAxisConfig(), BarConfig(), BarColorConfig()])
def test_resolve_config_passes_instance_through(config):
    assert _resolve_config(config, type(config)) is config


def test_resolve_config_rejects_invalid_dict_keys():
    with pytest.raises(ValueError, match="Invalid RankAxisConfig key"):
        _resolve_config({"not_a_field": "red", "also_bad": "blue"}, RankAxisConfig)


def test_resolve_config_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="Expected a RankAxisConfig or a dict of its fields"):
        _resolve_config(LabelConfig(), RankAxisConfig)
