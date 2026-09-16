# Plotting Word Shifts

All comparisons can be visualized as a [bar graph](plotting_comparisons.md). If the comparisons is a difference in weighted averages, or can be represented as one, then it can be plotted as a [word shift graph](../concepts/word_shift_graphs.md). In addition to showing _which_ tokens contribute to the difference, a word shift graph also shows _how_ they contribute.

!!! tip
    A comparison records whether a weighted average measure was used under its `is_weighted_avg` attribute.


## Example dataset

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs.

We make two comparisons using [sentiment lexicons](scoring_with_lexicons). The first comparison uses the [labMT](../api/lexicons/labMT.md) lexicon for both corpora. The second comparison uses the [SocialSent](../api/lexicons/SocialSent.md) lexicons, which are specific to each decade. We score Roosevelt's speeches with the lexicon from the 1940s, and Biden's speeches with the lexicon from the 2000s, the most recent decade available.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:setup"
```


## Word shift graph

We can create a word shift graph it by passing a comparison's name to [`Catalog.plot.shift`](../api/catalog/plot/shift.md). The name is the alias that we gave it when adding the comparison to the `Catalog`.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:shift"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```


## Total contributions

Word shift graphs indicate the different ways that tokens contribute to the difference. A token can have been used more or less in the comparison corpus versus the reference corpus $\left( \uparrow / \downarrow \right)$, and it can have a [relatively](../concepts/word_shift_graphs.md#reference-scores) positive or negative score $(+ / -)$. This means a token can contribute in four ways:

- $\left(+ \uparrow \right)$ A relatively positive token is used more
- $\left(+ \downarrow \right)$ A relatively positive token is used less
- $\left(- \uparrow \right)$ A relatively negative token is used more
- $\left(- \downarrow \right)$ A relatively negative token  is used less

If we set `show_totals=True`, we can see the relative magnitude of the different kinds of contributions.


### Single lexicon

When both corpora are scored with the same lexicon, tokens contribute in the four ways described above.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:single_totals"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```

The overall difference (the "Total") indicates that Biden's speeches have a higher sentiment than Roosevelt's speeches. This is because Biden's speeches use relatively positive words more $\left(+ \uparrow \right)$, such as "you" and "america", and relatively negative words less $\left(- \downarrow \right)$, such as "war". The sentiment difference is partly offset by Biden's speeches using some relatively positive words less $\left(+ \downarrow \right)$, such as "great" and "peace", and some relatively negative words more $\left(- \uparrow \right)$, such as "violence." These words offset how much more positive Biden's speeches would have been otherwise.

The bar colors can be configured with a [`ShiftBarConfig`](../api/configuration/plots/shift_bar_config.md).

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:single_colors"
```

```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```


### Multiple lexicons

When each corpus is scored with its own lexicon, each token has a different score in each of those lexicons. This means there are two other ways a token can contribute to the corpora's difference:

- $\left(\bigtriangleup \right)$ A word has a higher score in the comparison corpus
- $\left(\bigtriangledown \right)$ A word has a lower score in the comparison corpus

These contributions are _additive_ to the other ways that a token can contribute, and so they are shown as an additional stacked bar. 

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:multiple_totals"
```

```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```

The total is negative, so Biden's speeches have a lower sentiment than Roosevelt's speeches when each is scored with the lexicon of its time. The totals panels show that much of this difference comes from words having lower scores in the 2000s $\left(\bigtriangledown \right)$, such as "nation", "families", and "americans".

The different kinds of contributions can partially cancel each other out. When this happens, we show it by fading the canceled out portions. For example, "violence" is relatively negative word that was used more in Biden's speeches. However, that contribution is partly offset by the score change, because "violence" also has a relatively higher sentiment score in the 2000s lexicon dictionary than the 1940s lexicon dictionary.

The higher and lower score contribution colors can also be configured with a [`ShiftBarConfig`](../api/configuration/plots/shift_bar_config.md).

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:multiple_colors"
```

```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```
