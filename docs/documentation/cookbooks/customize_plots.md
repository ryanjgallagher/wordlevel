# Customizing Plots

Every part of a plot's appearance is set by a configuration passed to
[`plot.bar` or `plot.shift`](plots.md). Each one takes a dict of fields, as used throughout this
page, or the equivalent config class if you prefer the type checking.

This page covers appearance. For rank limits, orientation, totals, and figure size, see
[Plotting Comparisons](plots.md).

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:setup"
```

The colors below come from a palette chosen to stay distinguishable with color vision deficiency.
WordLevel does not enforce that, so it is worth checking any palette you pick.

## Bar colors and symbols

Which fields a bar graph takes depends on how its measure attributes tokens.

Measures that attribute each token to a **corpus**, such as the rank turbulence divergence and
frequency differences, take [`BarConfig`](../api/configuration/plots/bar_config.md) fields. The
symbol fields label the totals panels, so they only appear when totals are shown.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:bar_config"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

Measures that attribute by the **sign of the contribution**, such as lexicon scores and the
entropies, take [`ScoreSignBarConfig`](../api/configuration/plots/score_sign_bar_config.md) fields
instead. They name whether a token reinforces or offsets the overall difference rather than naming a
corpus.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:score_sign_bar_config"
```

```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

!!! tip
    A comparison records which of the two it uses under its `attributed_by` attribute, so you can
    check before choosing fields.

## Word shift bar colors

A word shift splits each token's contribution further, so it takes
[`ShiftBarConfig`](../api/configuration/plots/shift_bar_config.md) fields, with a color per
contribution type: whether a word is relatively positive or negative, and whether the comparison
corpus used it more or less. Each also has a `_counteracted` variant, used when the two stacked bars
point in opposite directions.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:shift_bar_config"
```

```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

## The totals bar

When totals are shown, [`TotalShiftConfig`](../api/configuration/plots/total_shift_config.md) fields
set the color and glyph of the grand total bar. The per-type panels above it take their colors from
the bar config.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:total_shift_config"
```

```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

## Labels

[`LabelConfig`](../api/configuration/plots/label_config.md) fields control the token labels beside
each bar: their color, size, weight, rotation, and the padding between a bar and its label.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:label_config"
```

```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

## Axes

The rank axis lists the tokens and the score axis carries their contributions. Both
[`RankAxisConfig`](../api/configuration/plots/rank_axis_config.md) and
[`ScoreAxisConfig`](../api/configuration/plots/score_axis_config.md) take titles, colors, font sizes,
and grid settings. `label_tick_step` sets how often the rank axis is labeled, and `bar_width` on the
rank axis is what sizes the figure along that direction.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:axis_config"
```

```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

## Borders and the zero line

[`BorderConfig`](../api/configuration/plots/border_config.md) frames the plot and
[`ZeroLineConfig`](../api/configuration/plots/zero_line_config.md) styles the line at the score axis
origin. A border width of `0` draws only the native axis lines; anything larger closes the frame.

```python exec="true" source="above" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:border_config"
```

```python exec="true" html="true" session="customize_plots"
--8<-- "snippets/cookbooks/customize_plots.py:render"
```

For every field on every config, see the
[plot configuration documentation](../api/configuration/plots/index.md).
