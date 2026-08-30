# Make a Comparison

This guide compares the speeches of two US presidents and finds the words that most distinguish
them. If you have not installed WordLevel yet, start with [Install](../install.md).

## Load some corpora

WordLevel ships with an example dataset of presidential speeches. Each document is a speech, and each
corpus is every speech by a particular president. We will use Franklin D. Roosevelt and Joe Biden.

```python exec="true" source="above" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:setup"
```

A `Catalog` holds the corpora we want to work with. Everything else in WordLevel starts here.

## Look at the data

The `Catalog` stores the corpora as a dataframe, with one row per token.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:display_config"
```

```python exec="true" source="material-block" result="text" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:inspect_catalog"
```

There is one column of frequencies per president, plus a `__wl_normed` column that WordLevel adds
for each corpus, holding the token's frequency as a proportion of that corpus.

## Compare the corpora

Now we ask how the two presidents differ. A **comparison** pairs two corpora and picks a measure to
score them with. Here we use the rank turbulence divergence, which asks how differently the two
presidents rank their words.

```python exec="true" source="above" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:comparison"
```

Roosevelt is the *reference* corpus and Biden is the *comparison* corpus. The order matters for
interpreting the results, so it is worth being deliberate about it.

## See which words differ

The comparison is stored on the `Catalog` under the name we gave it.

```python exec="true" source="material-block" result="text" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:top_tokens"
```

Each token gets a `comparison_score` measuring how much it contributes to the difference between the
corpora, a `score_rank` ordering the tokens by that contribution, and an `attribution` naming which
president the token is more characteristic of.

## Plot it

Reading a table of scores only goes so far. A bar graph shows the highest-contributing tokens and
which corpus each belongs to.

```python exec="true" source="above" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:bar_chart"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="make_a_comparison"
--8<-- "snippets/quick_start/make_a_comparison.py:bar_chart_render"
```

The two sides separate the presidents. Roosevelt's speeches are marked by the vocabulary of the
Second World War and industrial mobilization, while Biden's carry the language of the pandemic and
of Ukraine.

## Where to go next

- To understand what you just computed, read [Comparisons](../concepts/comparisons.md).
- To apply this to your own data, see [Constructing Catalogs](../cookbooks/constructing_catalogs.md).
- For other ways to measure a difference, see the
  [scoring methods](../api/comparison/score/index.md).

Or continue to [Make a Word Shift Graph](make_a_word_shift_graph.md).
