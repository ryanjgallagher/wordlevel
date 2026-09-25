# Make a Word Shift Graph

This guide shows you how to make a [word shift graph](../concepts/word_shift_graphs.md) with WordLevel. To follow along, make sure that you first [install the package](../install.md), and read the introduction on [making comparisons](make_a_comparison.md).

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president.

```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:setup"
```

## Load a sentiment lexicon

A _lexicon_ assigns a "score" to each token in its vocabulary. For example, the [labMT sentiment lexicon](../api/lexicons/labMT.md) maps each token to a score between 1 and 9, where 1 indicates that the token is associated with negativity ("sadness") and 9 indicates the token is associated with positivity ("happiness").

WordLevel provides [several sentiment lexicons](../api/lexicons/index.md) previously constructed by researchers. We will use the English labMT lexicon as an example. Lexicons are loaded using `wl.lex`.


```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:load_lexicon"
```

!!! note
    The first time that you load a lexicon, it is downloaded from [Hugging Face](https://huggingface.co/datasets/ryanjgallag/wordlevel-lexicons). The lexicon is cached locally, so subsequent calls do not download it again.


## Score the corpora

Like our [first comparison](make_a_comparison.md), we compare the presidential speeches using `wl.comp`. This time, we score them by passing the `labmt` sentiment lexicon to the `lexicon` method.


```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:comparison"
```

We pass a `reference_score` when scoring the words—in this case it is the center of labMT's 1-to-9 sentiment scale, which is 5. Each token's score is compared to this reference score: if it is greater than 5, then it is "relatively positive"; if it is less than 5, then it is "relatively negative."


## Plot the word shift

A [comparison bar graph](make_a_comparison.md#plot-the-scores) shows _which_ tokens contribute to the difference between two corpora. A word shift graph also shows _how_ they contribute. Here, there are four types of contributions:

- $\left(+ \uparrow \right)$ A relatively positive word is used more in the comparison corpus
- $\left(+ \downarrow \right)$ A relatively positive word is used less in the comparison corpus
- $\left(- \uparrow \right)$ A relatively negative word is used more in the comparison corpus
- $\left(- \downarrow \right)$ A relatively negative word is used less in the comparison corpus


```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:shift_chart"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:shift_chart_render"
```

The total difference is positive, which means that the comparison corpus (Biden) has a higher average sentiment than the reference corpus (Roosevelt). The tokens are interpreted with respect to this difference. For example, Biden's speeches have a higher sentiment than Roosevelt's because they use relatively positive words more ("you", "america", "thank") and relatively negative words less ("war", "fighting"). The higher sentiment is partly offset by other words, such as Biden's speeches using some relatively positive words less ("great", "peace", "victory") and some relatively negative words more ("violence", "dont", "tax").


## Save the figure

WordLevel plots are [Altair](https://altair-viz.github.io/) charts, so they can be saved as PNG, SVG, PDF, or HTML files with the `save` method. Saving requires some extra dependencies, which are installed with the [`save` extra](../install.md#saving-plots).

```python
chart.save("word_shift.png", ppi=300)
```


## Where to go next

Learn more about [word shift graphs](../concepts/word_shift_graphs.md) and how to [plot them](../cookbooks/plotting_word_shifts.md), or get started with your own data by [constructing a catalog](../cookbooks/constructing_catalogs.md) and [scoring it](../cookbooks/scoring_comparisons.md).
