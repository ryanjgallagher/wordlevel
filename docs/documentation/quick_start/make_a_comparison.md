# Make a Comparison

This guide shows you how to get started comparing texts with WordLevel. To follow along, make sure that you first [install the package](../install.md).

## Load data

A collection of documents is known as a [corpus](../concepts/corpora_and_catalogs.md#what-is-a-corpus). We compare multiple corpora by loading them into a [catalog](../concepts/corpora_and_catalogs.md#what-is-a-catalog). We will use an example dataset of US presidential speeches. Each document is a speech, and each corpus contains all of the speeches by a particular president. We will compare the speeches of Presidents Franklin D. Roosevelt and Joe Biden.

```python exec="true" source="above" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:setup"
```


## Inspect the `Catalog`

The `Catalog` stores the corpora together as a dataframe. There is one row per token, and one column per corpus, where the entry is the number of times that token appeared in that corpus. For example, the word "the" was used 9,559 times in speeches by Roosevelt.

Columns with the suffix `__wl_normed` contain the normalized frequencies of the tokens in each corpus, such that all of the frequency values within a corpus sum to 1.

```python exec="true" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:display_config"
```

```python exec="true" source="material-block" result="text" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:inspect_catalog"
```


## Compare the corpora

WordLevel lets us quantify how two corpora differ token by token (at the "word level"). We do this by creating a [comparison](../concepts/comparisons.md) between them and using a [measure](../api/comparison/score/index.md) to score their token-level differences.

We will use the [rank-turbulence divergence](../api/comparison/score/rank_divergence.md) as an example. Each token is ranked by frequency. The larger a token's difference in rank between the corpora, the more it contributes to the divergence.

We make comparisons by using `wl.comp` and providing the names of the corpora that we want to compare. We give this comparison the alias `"roosevelt_biden"`, which we will use to retrieve it from our `Catalog`.


```python exec="true" source="above" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:comparison"
```


## Retrieve the comparison scores

The comparison is stored on the `Catalog` under the alias we gave it, `"roosevelt_biden"`.

```python exec="true" source="material-block" result="text" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:top_tokens"
```

There is one row per token. Each token has a `comparison_score`, which comes directly from the scoring measure that we chose, the rank-turbulence divergence. It measures how much that token contributes to the difference between the corpora.

For the rank-turbulence divergence, the `attribution` indicates whether the token was used relatively more in the "reference" corpus or the "comparison" corpus. Because we defined our comparison as `wl.comp("Franklin D. Roosevelt", "Joe Biden")`, the reference corpus is Roosevelt's speeches, and the comparison corpus is Biden's speeches. Read more about [comparisons](../concepts/comparisons.md) for details on reference and comparison corpora.


## Plot the scores

We can visualize the comparison scores as a bar graph. There is one bar per token, and its direction indicates whether it is attributed to the reference corpus (Roosevelt, left) or the comparison corpus (Biden, right).

```python exec="true" source="above" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:bar_chart"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:bar_chart_render"
```

This provides a quick but effective summary of how the two corpora differ. For example, we see that Roosevelt's speeches are characterized by words pertaining to World War II ("japanese", "germany", "british", "axis") and industrial mobilization ("industry", "ships", "labor", "production"), whereas Biden's speeches are characterized by the COVID-19 pandemic ("covid", "vaccinated", "pandemic") and the Russian invasion of Ukraine ("ukraine", "putin", "nato").

## Where to go next

Continue to learn how to conduct sentiment analysis of Roosevelt's and Biden's speeches and [make a word shift graph](make_a_word_shift_graph.md), or get started with your own data by [constructing a catalog](../cookbooks/constructing_catalogs.md) and [scoring it](../cookbooks/scoring_comparisons.md).
