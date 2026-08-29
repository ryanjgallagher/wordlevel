# Plotting Comparisons

WordLevel draws two kinds of plot, and they take almost all the same options. The difference is what
each can plot and how much it can say.

A **bar graph** works with any comparison, whatever measure scored it. It shows the tokens that
contribute most, ordered by the magnitude of their contribution.

A **word shift graph** goes further, breaking each token's contribution into its parts, but it only
works for measures built on weighted averages: lexicon scores, the entropies, and the Jensen-Shannon
divergence. For anything else, such as the rank turbulence divergence, use a bar graph.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:setup"
```

## Making a bar graph

Pass the alias of a comparison attached to the `Catalog`.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:bar"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

## Making a word shift graph

A word shift needs a weighted-average measure. A lexicon is the most direct: it gives every word a
score, so the corpus-level difference is a difference in average score.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:lexicon_comparison"
```

The `reference_score` is the baseline each word's score is measured against, and it decides which
side of the shift a word falls on. Here it is the lexicon's own neutral point. Set it deliberately
rather than relying on a default.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:shift"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

Each word gets two stacked bars, colored by what each part of its contribution represents: whether
the word is relatively positive or negative and used more or less often, and whether its score is
higher or lower. Pass `show_legend=True` to name the colors.

A lexicon is not the only option. The Shannon entropy is also a weighted average, so it shifts too,
measuring how the diversity of the two corpora differs rather than their sentiment.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:entropy_comparison"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

## Showing fewer tokens

`max_rank` sets how far down the ranking to plot. It defaults to 50, and applies to both plots.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:max_rank"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

## Changing the orientation

By default tokens run down the page and bars extend across it. Pass `vertical=False` to swap them.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:horizontal"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

## Adding the totals

`show_totals` adds panels that accumulate every token's contribution, including tokens below the
plotted ranks, so you can see how much of the overall difference is on display.

On a bar graph the totals are split by the corpus each token is attributed to.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:bar_totals"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

On a word shift they are split by contribution type instead, each labeled with the symbol for that
type, with a grand total beneath them. The symbols appear only here, not on the individual bars.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:shift_totals"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

## Sizing and legends

`scale_factor` scales the whole figure and `show_legend` adds a legend naming what each color means.

Sizing follows the score axis rather than the screen: when the chart is vertical the bars run across,
so `width` sets their length, and when it is horizontal the bars run up the page, so `height` does.
The other dimension is set by the number of tokens plotted, and passing it has no effect.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:legend_and_size"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

To size that other dimension, set how much room each token gets rather than the total. `bar_width` on
the rank axis is the space per token, so the chart grows or shrinks with the number of tokens plotted.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:rank_size"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

The rank axis has more options than this, and so do the other parts of the figure. See
[Customizing Plots](customize_plots.md).

## Non-negative measures

Some measures, such as the rank turbulence divergence, only ever produce non-negative scores. By
default WordLevel still points each bar left or right according to which corpus the token is
attributed to, so the plot reads like any other. Pass `split_nonnegative_by_corpus=False` to keep
every bar on one side and distinguish the corpora by color alone. This applies to bar graphs, since
a word shift always has a direction.

```python exec="true" source="above" session="plots"
--8<-- "snippets/how_to/plots.py:split_nonnegative"
```

```python exec="true" html="true" session="plots"
--8<-- "snippets/how_to/plots.py:render"
```

For every parameter, including axis, label, border, and color configuration, see the
[bar plot](../api/catalog/plot/bar.md) and [word shift](../api/catalog/plot/shift.md) documentation.
