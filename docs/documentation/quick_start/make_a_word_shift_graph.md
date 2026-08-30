# Make a Word Shift Graph

[Make a Comparison](make_a_comparison.md) found the words that most distinguish two presidents.
This guide asks a different question: not just *which* words differ, but *how* they add up to an
overall difference. The answer is a **word shift graph**, WordLevel's signature plot.

We pick up with the same two corpora.

```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:setup"
```

## Load a sentiment lexicon

A **lexicon** assigns a score to each word. WordLevel ships with several; we will use labMT, which
rates words for how happy or unhappy they feel, from 1 at the most negative to 9 at the most
positive.

```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:load_lexicon"
```

The first time you load a lexicon it is downloaded and cached, so later loads are instant.

## Score the comparison

Now we compare the two presidents by average sentiment, and ask which words drive the difference.

```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:comparison"
```

Two arguments are doing real work here.

The **`reference_score`** is the neutral point each word is measured against. We pass the lexicon's
own center, 5, so a word scoring 7 counts as positive and a word scoring 3 counts as negative. This
is not an incidental setting: move the reference score and words change which side of the shift they
fall on, and the graph tells a different story. It is worth choosing deliberately every time.

Calling **`normalize`** rescales each word's contribution so it reads as a share of the total
difference between the corpora, rather than a raw quantity. See
[Normalization](../concepts/normalization.md) for when to use each method.

## Plot the shift

```python exec="true" source="above" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:shift_chart"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="make_a_word_shift_graph"
--8<-- "snippets/quick_start/make_a_word_shift_graph.py:shift_chart_render"
```

Each bar is one word's contribution to the overall difference in sentiment, and the bars are ordered
by how much they contribute. A word can push the difference in either direction, and the two bars are
colored by why: whether the word is relatively positive or negative and used more or less often, and
whether its score is higher or lower.

The panels above the bars are the totals. They accumulate the individual contributions into the
overall difference between the two presidents, so you can see how much of it the words on display
account for.

## Where to go next

- To work with lexicons in more depth, see [Scoring with Lexicons](../cookbooks/scoring_with_lexicons.md).
- To shape the graph, see
  [Making a word shift graph](../cookbooks/plots.md#making-a-word-shift-graph).
- For the measures behind these plots, see the
  [scoring methods](../api/comparison/score/index.md).
