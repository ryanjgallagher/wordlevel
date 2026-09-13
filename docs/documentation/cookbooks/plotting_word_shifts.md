# Plotting Word Shifts

A [word shift graph](../concepts/word_shift_graphs.md) shows the tokens that contribute most to a comparison, and it breaks down _how_ each token contributes. Word shift graphs can be made with `Catalog.plot.shift` for comparisons scored with weighted averages, such as lexicon scores and the entropies. For comparisons scored with other measures, see [Plotting Comparisons](plotting_comparisons.md).


## Example dataset

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs.

We make two comparisons using sentiment lexicons. See the [cookbook](scoring_with_lexicons.md) for more about scoring with lexicons.

The first comparison uses the [labMT](../api/lexicons/labMT.md) lexicon for both corpora. The second comparison uses the [SocialSent](../api/lexicons/SocialSent.md) lexicons, which are specific to each decade. We score Roosevelt's speeches with the lexicon from the 1940s, and Biden's speeches with the lexicon from the 2000s, the most recent decade available. Both comparisons are [normalized](normalizing_scores.md), so each token's contribution is a percentage of the total.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:setup"
```


## Word shift graph

To plot a comparison, pass its name to `plot.shift`. The name is the alias that we gave the comparison when adding it to the `Catalog`.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:shift"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```


## Total contributions

Setting `show_totals=True` adds panels above the word shift graph that sum up the contributions of _all_ tokens, including those that are not shown in the graph. The panels are split by the type of contribution, and they also show the total overall difference.


### Single lexicon

When both corpora are scored with the same lexicon, each token can contribute in four ways:

- $\left(+ \uparrow \right)$ A relatively positive word is used more
- $\left(+ \downarrow \right)$ A relatively positive word is used less
- $\left(- \uparrow \right)$ A relatively negative word is used more
- $\left(- \downarrow \right)$ A relatively negative word is used less

A word is relatively positive or negative with respect to the reference score, which is the center of labMT's scale. Whether a word is used more or less is with respect to the comparison corpus, Biden's speeches.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:single_totals"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```

The total is positive, so Biden's speeches have a higher sentiment than Roosevelt's speeches. The totals panels show that this is mostly because Biden's speeches use relatively positive words more $\left(+ \uparrow \right)$, such as "you" and "america", and relatively negative words less $\left(- \downarrow \right)$, such as "war". The difference is partly offset by Biden's speeches using some relatively positive words less $\left(+ \downarrow \right)$, such as "great" and "peace", and some relatively negative words more $\left(- \uparrow \right)$, such as "violence".

The colors of the bars are set with the `bar_config`, which has a key for each type of contribution. See [`ShiftBarConfig`](../api/configuration/plots/shift_bar_config.md) for all options.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:single_colors"
```

```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```


### Multiple lexicons

When each corpus is scored with its own lexicon, a word's score can also change between the corpora. This adds two more ways that each token can contribute:

- $\left(\bigtriangleup \right)$ A word has a higher score in the comparison corpus
- $\left(\bigtriangledown \right)$ A word has a lower score in the comparison corpus

These contributions are shown as a second bar stacked on each token's first bar.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:multiple_totals"
```

```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```

The total is negative, so Biden's speeches have a lower sentiment than Roosevelt's speeches when each is scored with the lexicon of its time. The totals panels show that much of this difference comes from words having lower scores in the 2000s $\left(\bigtriangledown \right)$, such as "nation", "families", and "americans".

When the two bars of a token point in opposite directions, they partially cancel out, and the bars are faded. For example, "peace" is a relatively positive word that Biden's speeches use less $\left(+ \downarrow \right)$, but it has a higher score in the 2000s $\left(\bigtriangleup \right)$.

Some words only have a score in one of the two lexicons. By default, these words [borrow](scoring_with_lexicons.md#borrowing-scores) their score from the other lexicon, and their labels are marked with an asterisk, such as "well\*" and "mr\*".

The colors of the new bars are set with the `higher_score` and `lower_score` keys of the `bar_config`. Each type of contribution also has a `_counteracted` key, which sets the color of its faded bars.

```python exec="true" source="above" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:multiple_colors"
```

```python exec="true" html="true" session="plotting_word_shifts"
--8<-- "snippets/cookbooks/plotting_word_shifts.py:render"
```

Word shift graphs can be sized, oriented, and configured in the same ways as [bar graphs](plotting_comparisons.md#size-and-orientation). For all parameters, see the [`plot.shift`](../api/catalog/plot/shift.md) and [plot configuration](../api/configuration/plots/index.md) documentation.
