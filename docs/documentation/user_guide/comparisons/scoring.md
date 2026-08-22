# Comparisons

## What is a comparison?

A **comparison** quantifies the token-level differences between two corpora. It is a scientific plan
for how to compare the corpora, including how to measure their differences, which tokens to include,
and more. If you are familiar with [Polars], this is similar to how
[expressions](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) are plans for how
to transform and analyze columns in a dataframe.

The core of every comparison is a _scoring method_ for quantifying the token-level differences. 
WordLevel implements a [full suite of measures](../../api/comparison/score/index.md), including 
basic frequency differences, entropy-based metrics like the Kullback-Leibler and Jensen-Shannon 
divergences, and other advanced measures.

Each comparison has two corpora: a **_reference_** corpus and a **_comparison_** corpus. The 
reference corpus can be considered the baseline from which the comparison corpus deviates. It is
important to be mindful of which corpus is which, since it can affect the interpretation of the
results. 

For example, say we have two corpora: the speeches of President Franklin D. Roosevelt and
the speeches of President Joe Biden. Suppose Roosevelt's speeches are the reference corpus and 
Biden's are the comparison corpus, and we want to simply measure the difference in how often 
different words appear. Since Roosevelt is our reference, this means we are calculating 
$f^{(\text{Biden})}_\tau - f^{(\text{Roosevelt})}_\tau$, where $f^{(C)}_\tau$ is the frequency of 
token $\tau$ in corpus $C$. If the difference is positive, then it means that the word was used more 
by Biden. Similarly, if it is negative, then it means it was used less by Biden (more by Roosevelt).

Some scoring methods, like the 
[Jensen-Shannon divergence](../../api/comparison/score/jensen_shannon_divergence.md) or 
[rank turbulence divergence](../../api/comparison/score/rank_divergence.md), are symmetric by 
default, meaning that the token-level difference is the same regardless of which corpus is the 
reference and which is the comparison. However, it can still be important to track which corpus is 
which, particularly for interpreting shift plots. See the 
[scoring documentation](../../api/comparison/score/index.md) for details on how to interpret each 
measure.

<!-- TODO: add documentation link to shift plots above -->


## Example dataset

To work with comparisons, we will use an example dataset of presidential speeches. Each document is 
a speech by a US president, and each corpus is the set of all speeches by a particular president. 
We will use the speeches by Presidents Joe Biden and Franklin D. Roosevelt.

```python exec="true" source="above" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:setup"
```

If you want to see which other corpora are available, you can run `speeches.as_df().columns`.


## Scoring a comparison

Let's compare Roosevelt and Biden's speeches by measuring the differences in the
[normalized frequencies](../../api/comparison/score/proportion.md) of each token. Roosevelt's
speeches will be our reference corpus, and Biden's speeches will be our comparison corpus.

```python exec="true" source="above" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:proportion"
```

Let's break this down:

- `wl.comp("Franklin D. Roosevelt", "Joe Biden")` defines which corpora we want to compare. The
first argument is the reference corpus (Roosevelt) and the second argument is the comparison corpus
(Biden). We can use the names of any two corpora in our `Catalog` to create our comparison.
- `score.proportion()` specifies that we want to compare the normalized frequencies (proportions) of
the two corpora in our comparison. All scoring methods are called from the `score` namespace; see
the [documentation](../../api/comparison/score/index.md) for other measures.
- `alias("roosevelt_biden")` gives the comparison the name `"roosevelt_biden"`, which we can use to
refer to the comparison later.
- `cl.with_comparisons(...)` attaches the comparison to the `Catalog`. If you are familiar with 
[Polars], this is similar to how `df.with_columns()` attaches expressions to a dataframe.

!!! warning
    Remember, `wl.comp()` does not do anything by itself—it is a lazily evaluated plan for how to
    make the comparison. Attaching it to the `Catalog` is how the comparison is actually executed.


## Inspecting a comparison

### Scores

We've made our first comparison! We can access it from the `Catalog` using the alias
`"roosevelt_biden"` we gave it.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:display_config"
```

```python exec="true" source="material-block" result="text" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:inspect"
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

### Reproducibility Metadata

There is also useful information on the comparison `roosevelt_biden` itself. It tracks all of the
relevant information needed for scientific reproducibility, including the reference corpus, the 
comparison corpus, the scoring method, the method's parameters, and more. See the
[documentation](../../api/comparison/comparison_result.md) for all of the tracked metadata.


## Making multiple comparisons

Let's compare Roosevelt and Biden's speeches with other measures, the 
[rank turbulence divergence](../../api/comparison/score/rank_divergence.md), and the
[Shannon entropy](../../api/comparison/score/shannon_entropy.md). Note how we can simply 
add them as additional comparisons under different aliases.

```python exec="true" source="above" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:multiple_comparisons"
```

All of the comparisons are stored in our catalog, including the original comparison we made using 
the proportions.

```python exec="true" source="material-block" result="text" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:comparison_keys"
```


## Normalizing comparison scores

It can sometimes be difficult to interpret the individual token-level scores. To make them more
interpretable, we can normalize them in different ways. 

### By Total Difference

One way is to normalize by the sum of the absolute scores—if each score is represented by $\delta$, 
then we normalize by $\sum | \delta |$. This is the _total_ difference between the corpora, 
regardless of the score signs. We can interpret each normalized score as the share of that total 
difference accounted for by an individual token. For example, a score of 0.05 means that a token 
makes up 5% of the total difference between the corpora. A score of -0.05 also means that a token 
makes up 5% of the total difference, but contributes in the negative direction. The absolute values 
of the normalized scores sum to 1.

```python exec="true" source="above" session="scoring" hl_lines="4"
--8<-- "snippets/user_guide/comparisons/scoring.py:normalize_sum_abs"
```

### By Net Difference

We can also normalize by the magnitude of the overall score, $| \sum \delta |$. This is the _net_
difference between the corpora. We can interpret each normalized score as how large a token's
contribution is relative to the overall, net difference—it is the ratio between them. For example,
a score of 0.05 means that a token's contribution is 5% the size of the net difference. A score of
-1.05 means that a token's contribution is 105% the size of the net difference, and it contributes
in the negative direction. Even though individual normalized scores can exceed 1 in magnitude,
they still sum to $\pm 1$ because opposing contributions can cancel one another out.

```python exec="true" source="above" session="scoring" hl_lines="4"
--8<-- "snippets/user_guide/comparisons/scoring.py:normalize_abs_sum"
```

!!! note
    For scoring measures that produce token-level scores that are always non-negative, like the 
    [rank divergence](../../api/comparison/score/rank_divergence.md) and 
    [Jensen-Shannon divergence](../../api/comparison/score/jensen_shannon_divergence.md), the
    different forms of normalization produce the same result. For other measures, where tokens' 
    scores can be positive or negative, the normalizations can produce different results.

!!! warning
    The net difference of normalized frequencies is always zero. This means that only `"sum_abs"` 
    can be used as a normalization method when scoring by `proportion`. Using `"abs_sum"` will
    raise an error.

!!! warning
    Normalizing scores causes them to be eagerly evaluated because the normalization needs to verify
    that the normalization constant is nonzero. 

### Reproducibility Metadata

Regardless of the normalization method, the reproducibility metadata is stored in the comparison
under the attributes `normalized`, `normalized_by`, and `normalization_constant`.

```python exec="true" source="material-block" result="text" session="scoring"
--8<-- "snippets/user_guide/comparisons/scoring.py:normalization_metadata"
```


  [Polars]: https://pola.rs/