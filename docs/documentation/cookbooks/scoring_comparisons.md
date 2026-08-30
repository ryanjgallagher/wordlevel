# Scoring Comparisons

For what a comparison is, how the reference and comparison corpora differ, and why that direction
matters, see [Comparisons](../concepts/comparisons.md).

## Example dataset

To work with comparisons, we will use an example dataset of presidential speeches. Each document is 
a speech by a US president, and each corpus is the set of all speeches by a particular president. 
We will use the speeches by Presidents Joe Biden and Franklin D. Roosevelt.

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:setup"
```

If you want to see which other corpora are available, you can run `speeches.as_df().columns`.


## Scoring a comparison

Let's compare Roosevelt and Biden's speeches by measuring the differences in the
[normalized frequencies](../api/comparison/score/proportion.md) of each token. Roosevelt's
speeches will be our reference corpus, and Biden's speeches will be our comparison corpus.

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:proportion"
```

Let's break this down:

- `wl.comp("Franklin D. Roosevelt", "Joe Biden")` defines which corpora we want to compare. The
first argument is the reference corpus (Roosevelt) and the second argument is the comparison corpus
(Biden). We can use the names of any two corpora in our `Catalog` to create our comparison.
- `score.proportion()` specifies that we want to compare the normalized frequencies (proportions) of
the two corpora in our comparison. All scoring methods are called from the `score` namespace; see
the [documentation](../api/comparison/score/index.md) for other measures.
- `alias("roosevelt_biden")` gives the comparison the name `"roosevelt_biden"`, which we can use to
refer to the comparison later.
- `cl.with_comparisons(...)` attaches the comparison to the `Catalog`. If you are familiar with 
[Polars], this is similar to how `df.with_columns()` attaches expressions to a dataframe.

!!! warning
    Remember, `wl.comp()` does not do anything by itself—it is a lazily evaluated plan for how to
    make the comparison. Attaching it to the `Catalog` is how the comparison is actually executed.


## Inspecting a comparison

We've made our first comparison! We can access it from the `Catalog` using the alias
`"roosevelt_biden"` we gave it.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:display_config"
```

```python exec="true" source="material-block" result="text" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:inspect"
```

The comparison dataframe `df` contains a variety of information. Above, we can see some of the most
important columns for comparing proportions.

- `Franklin D. Roosevelt__wl_normed` and `Joe Biden__wl_normed` are the normalized frequencies
that are automatically calculated for the comparison.
- The `comparison_score` is the difference in normalized frequencies (proportions) that we
calculated by calling `score.proportion()`.
- The `score_rank` ranks the `comparison_score` by magnitude. The closer a token's rank is to 1, 
the larger the difference is between the corpora.
- The `attribution` indicates which corpus each token is more closely "associated" with. For
example, because Roosevelt used the word "the" relatively more than Biden (a negative
`comparison_score`), it is associated with the reference corpus (which is Roosevelt).

The comparison also carries metadata describing how it was made. See
[Recording Steps for Reproducibility](recording_reproducibility.md).


## Making multiple comparisons

Let's compare Roosevelt and Biden's speeches with other measures, the 
[rank turbulence divergence](../api/comparison/score/rank_divergence.md), and the
[Shannon entropy](../api/comparison/score/shannon_entropy.md). Note how we can simply 
add them as additional comparisons under different aliases.

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:multiple_comparisons"
```

All of the comparisons are stored in our catalog, including the original comparison we made using 
the proportions.

```python exec="true" source="material-block" result="text" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:comparison_keys"
```


Comparison scores can also be rescaled onto a common scale; see
[Normalizing Scores](normalizing_scores.md).


  [Polars]: https://pola.rs/