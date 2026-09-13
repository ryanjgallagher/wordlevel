# Plotting Comparisons

A bar graph shows the tokens that contribute most to a comparison, ordered by the magnitude of their contribution. Bar graphs can be made with `Catalog.plot.bar` for a comparison scored with any measure.

For comparisons scored with weighted averages, such as lexicon scores and the entropies, a [word shift graph](../concepts/word_shift_graphs.md) can break each token's contribution down even further.


## Example dataset

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs.

We make two comparisons: one using the [Jensen-Shannon divergence](../api/comparison/score/jsd.md) and one using the difference in [Shannon entropy](../api/comparison/score/shannon_entropy.md). Both are [normalized](normalizing_scores.md), so each token's contribution is a percentage of the total.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:setup"
```


## Bar plot

To plot a comparison, pass its name to `plot.bar`. The name is the alias that we gave the comparison when adding it to the `Catalog`.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:bar"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


## Total contributions

Setting `show_totals=True` adds panels above the bar graph that sum up the contributions of _all_ tokens, including those that are not shown in the graph. The panels are split by how the tokens are attributed, and they also show the total overall difference.


### Attributed by corpus

Some measures attribute each token to one of the two corpora. For the Jensen-Shannon divergence, a token is attributed to the corpus that uses it relatively more often: tokens that are more common in Roosevelt's speeches are attributed to the reference corpus, and tokens that are more common in Biden's speeches are attributed to the comparison corpus. Each bar points in the direction of the corpus that its token is attributed to.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:corpus_totals"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

The colors of the bars are set with the `reference` and `comparison` keys of the `bar_config`. The `reference_symbol` and `comparison_symbol` keys set the labels in the totals panels. See [`BarConfig`](../api/configuration/plots/bar_config.md) for all options.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:corpus_colors"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

!!! tip
    A comparison records how its tokens are attributed under its `attributed_by` attribute. See [how tokens are attributed](../concepts/comparisons.md#how-tokens-are-attributed) for more details.


### Attributed by sign

Other measures attribute each token by whether it _reinforces_ or _offsets_ the overall difference between the corpora. Biden's speeches have a higher Shannon entropy than Roosevelt's speeches, so the overall difference is positive. A token reinforces that difference when its contribution is also positive, pushing Biden's entropy higher relative to Roosevelt's. A token offsets the difference when its contribution is negative, pulling back toward Roosevelt having the higher entropy.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:sign_totals"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

The colors of the bars are set with the `reinforce` and `offset` keys of the `bar_config`. The `reinforce_symbol` and `offset_symbol` keys set the labels in the totals panels. See [`ScoreSignBarConfig`](../api/configuration/plots/score_sign_bar_config.md) for all options.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:sign_colors"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


## Size and orientation


### Number of tokens

The `max_rank` parameter sets how many of the top contributing tokens are shown. By default, the top 50 tokens are shown.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:max_rank"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Height and width

By default, bar graphs are vertical: tokens are listed from top to bottom and the bars extend left and right. The `width` parameter sets the width of the graph in pixels.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:width"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

The height of a vertical graph depends on how many tokens are shown. Rather than setting the height directly, the `rank_axis` sets how much space each token occupies. Its `bar_width` key is the number of pixels given to each token, so the height is roughly `max_rank` times `bar_width`. Its `bar_padding` key is the fraction of that space left empty between bars.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:height"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Orientation

Setting `vertical=False` makes a horizontal bar graph: tokens are listed from left to right and the bars extend up and down. The roles of the dimensions swap: `height` sets the height of the graph in pixels, and the width is set by the `rank_axis`.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:horizontal"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

By default, the labels of a horizontal bar graph are turned on their side so that they read vertically. The `rotation` key of the `label_config` sets a different angle, in degrees. Labels on bars pointing in the opposite direction are rotated to the mirrored angle so they still extend away from their bars.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:rotated_labels"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


## Configuration

Every part of the bar graph can be configured by passing a dictionary of options. Each configuration can also be passed as its corresponding configuration class.


### Axes

The rank axis lists the tokens, and the score axis shows their contributions. Both can be configured with titles, colors, font sizes, and grid lines. See [`RankAxisConfig`](../api/configuration/plots/rank_axis_config.md) and [`ScoreAxisConfig`](../api/configuration/plots/score_axis_config.md) for all options.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:axes"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Labels

The token labels beside each bar can be configured with their color, size, weight, and padding. See [`LabelConfig`](../api/configuration/plots/label_config.md) for all options.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:labels"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Totals

When the totals panels are shown, the color and label of the overall total can be configured. The colors and labels of the other totals come from the `bar_config`. See [`TotalShiftConfig`](../api/configuration/plots/total_shift_config.md) for all options.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:totals"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Borders

The border around the graph can be configured with its color and width. A width of `0` only draws the axis lines, while a larger width draws a box around the whole graph. The line at zero along the score axis is configured separately. See [`BorderConfig`](../api/configuration/plots/border_config.md) and [`ZeroLineConfig`](../api/configuration/plots/zero_line_config.md) for all options.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:borders"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

For all parameters, see the [`plot.bar`](../api/catalog/plot/bar.md) and [plot configuration](../api/configuration/plots/index.md) documentation.
