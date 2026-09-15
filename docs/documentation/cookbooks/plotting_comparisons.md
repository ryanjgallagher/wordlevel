# Plotting Comparisons

Comparisons measure how much each token contributes to difference between two corpora. We can visualize those differences by plotting them as a bar graph. Plots are made by calling functions from the [`Catalog.plot`](../api/catalog/plot/index.md) namespace.


## Example dataset

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs.

We make two comparisons: one using the [Jensen-Shannon divergence](../api/comparison/score/jsd.md) and one using the difference in [Shannon entropy](../api/comparison/score/shannon_entropy.md).

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:setup"
```


## Bar plot

Once we make a comparison, we can plot it by passing its name to [`Catalog.plot.bar`](../api/catalog/plot/bar.md). The name is the alias that we gave it when adding the comparison to the `Catalog`.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:bar"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


## Total contributions

There are two things that can give more context to the comparison bar graph. First, we can add a legend indicating what each bar means. Second, we can show the total contributions, and how they are _attributed_. The attribution depends on the type of comparison we made.

Both of these aspects are added to the plot if set `show_totals=True`, which appends panels to the top of the bar graph summing of the contributions of all tokens, including those that are not shown in the graph. 

!!! tip
    A comparison records how its tokens are [attributed]((../concepts/comparisons.md#how-tokens-are-attributed)) under its `attributed_by` attribute.

### Attributed by corpus

Some comparison measures attribute tokens to a specific corpus, based whether its relatively more frequent in one or the other. For example, with the Jensen-Shannon divergence, a token used more in Roosevelt's speeches is attributed to the reference corpus, and a token used more in Biden's speeches is attributed to the comparison corpus. Each bar points in the direction of the attribution.

In the example below, "you," "ukraine," and "afghanistan" are attributed to Biden's speeches (the comparison corpus), and "of," "government," and "men" are attributed to Roosevelt's speeches (the reference). 

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:corpus_totals"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

If a comparison attributes by corpus, then its bars can be configured with a [`BarConfig`](../api/configuration/plots/bar_config.md). For example, we can change their colors and the labels of the total contributions. 

Note, we can pass a dictionary instead of a `BarConfig` instance, and we do not have to specify every field—only the ones that we want to change.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:corpus_colors"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Attributed by sign

Some comparison measures cannot attribute tokens to a specific corpus because there is a more complex interplay between how often it appears and how it "scores" in a corpus. For example, when we compare two corpora by entropy, both the frequency of a token $p_\tau$ and its surprisal $- \log p_\tau$ in both corpora affect the final token-level contribution. 

For these measures, we attribute a token by whether it _reinforces_ or _offsets_ the overall difference between the corpora. In the example below, we can see that Biden's speeches have a higher entropy than Roosevelt's speeches because the overall difference (the "Total") is positive. An individual token _reinforces_ that difference when its contribution is also positive, pushing Biden's speeches' entropy higher than Roosevelt's speeches. A token _offsets_ that difference when its contribution is negative, pulling back the difference such that is not as large as it could be otherwise.


```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:sign_totals"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

If a comparison attributes by reinforcements and offsets, which is the same as attributing it by the contribution's sign, then its bars can be configured with a [`ScoreSignBarConfig`](../api/configuration/plots/score_sign_bar_config.md). Like before, we can pass a dictionary instead of a `ScoreSignBarConfig` instance, and we do not have to specify every field—only the ones that we want to change.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:sign_colors"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


## Size and orientation


### Number of tokens

As we have seen in some of the examples already, the `max_rank` parameter can be used to set how many tokens are shown in the bar graph. By default, the top 50 contributing tokens are displayed.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:max_rank"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Orientation

By default, the bar graph is vertical. This makes it easier for a reader to interpret the tokens because they can be easily read horizontally. However, we can plot the bar graph horizontally, if we set `vertical=False`. 

The `rotation` key of a [`LabelConfig`](../api/configuration/plots/label_config.md) can be used to rotate the bar labels so that they can be ready more easily even in the horizontal layout.


```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:rotated_labels"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Height and width

When the bar graph is vertical, the `width` parameter controls the width of the graph in pixels.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:width"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

For a vertical bar graph, its height is automatically determined from two things: the number of tokens displayed, and how wide each bar is. The number of tokens is set by `max_rank`. The `bar_width` is specified as a configuration of the the rank axis, which is the vertical axis. It indicates how many pixels to give each token vertically. Other aspects of the height can be controlled through some of the other [`RankAxisConfig`](../api/configuration/plots/rank_axis_config.md) parameters.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:height"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```

For a horizontal bar graph, when `vertical=False`, width of the bars is still controlled by the `RankAxisConfig` because it is now the horizontal axis. However, the `width` parameter is ignored, and `height` becomes the tunable parameter that can be used to specify the height of the graph.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:horizontal_size"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


## Configuration

### Axes

The _score axis_ measures the magnitude of each token's contribution, and the _rank axis_ indicates how tokens are ordered by those contributions. Things like the titles, font sizes, and colors can be configured using a [`ScoreAxisConfig`](../api/configuration/plots/score_axis_config.md) and [`RankAxisConfig`](../api/configuration/plots/rank_axis_config.md) respectively.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:axes"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Labels

Each bar is _labeled_ with its corresponding token. How labels are displayed can be configured with a [`LabelConfig`](../api/configuration/plots/label_config.md).

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:labels"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Totals

When the total contributions are shown (`show_totals=True`), the color and label of the grand total can be configured with a [`TotalShiftConfig`](../api/configuration/plots/total_shift_config.md). To configure the colors and labels of the totals per attribution, [see above](#total-contributions) for using the `bar_config`.

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:totals"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```


### Borders

The border and middle line (zero line) can also be configured stylistically using a [`BorderConfig`](../api/configuration/plots/border_config.md) and [`ZeroLineConfig`](../api/configuration/plots/zero_line_config.md).

```python exec="true" source="above" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:borders"
```

```python exec="true" html="true" session="plotting_comparisons"
--8<-- "snippets/cookbooks/plotting_comparisons.py:render"
```