# Scoring Comparisons


## Example data

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president.

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:setup"
```


## Scoring a comparison

We will compare Roosevelt's and Biden's speeches by measuring the difference in the [normalized frequency](../api/comparison/score/proportion.md) of each token. Roosevelt's speeches will be our reference corpus, and Biden's speeches will be our comparison corpus. This means that we are calculating the difference $p_\tau^{(\text{Biden})} - p_\tau^{(\text{Roosevelt})}$, where $p_\tau$ is the normalized frequency of token $\tau$.

We make a comparison by using the constructor `wl.comp`, and providing it to the `Catalog`'s method `with_comparisons`.

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:proportion"
```

In the comparison above, `wl.comp("Franklin D. Roosevelt", "Joe Biden")` defines which corpora we want to compare. The first argument is the reference corpus (Roosevelt) and the second argument is the comparison corpus (Biden). We can use the names of any two corpora in our `Catalog` to create our comparison.

The method `score.proportion()` specifies that we want to compare the tokens' normalized frequencies (proportions). All scoring methods are called from the `score` namespace; see the [API reference](../api/comparison/score/index.md) for other measures.

We call the `alias` method to give our comparison the name `"roosevelt_biden"`. This is the name that we can use to refer to the comparison when working with the `Catalog`.

!!! note
    `wl.comp` does not do anything by itself: it returns a lazily evaluated plan for how to
    make the comparison. Attaching it to the `Catalog` is how the comparison is actually executed.


## Inspecting a comparison

The method `with_comparisons` stores the comparison in our `Catalog`. If you are familiar with [Polars], this is similar to how `df.with_columns()` attaches expressions (columns) to a dataframe. 

We can access our comparison using the alias that we gave it, `"roosevelt_biden"`.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:display_config"
```

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:get_comparison"
```

The comparison's dataframe `df` contains details about the token-by-token differences.

```python exec="true" source="material-block" result="text" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:inspect"
```

The column `Joe Biden__wl_normed` is $p_\tau^{(\text{Biden})}$, the normalized frequency of the token in Biden's speeches. Similarly, the column `Franklin D. Roosevelt__wl_normed` is $p_\tau^{(\text{Roosevelt})}$. The `comparison_score` is the difference $p_\tau^{(\text{Biden})} - p_\tau^{(\text{Roosevelt})}$.

The `attribution` is determined by whether the score is positive or negative. When it is positive, the token was used more in Biden's speeches, and so it is "attributed" to his corpus (the `comparison`). When the score is negative, the token was used more in Roosevelt's speeches, and so it is attributed to the `reference` corpus.


## Making multiple comparisons

A `Catalog` can contain multiple comparisons, differentiated by their aliases. We can construct them by simply calling the method `with_comparisons` again, which accepts any number of comparisons.

```python exec="true" source="above" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:multiple_comparisons"
```

Now, if we look at the catalog's comparisons, we see that the new ones are stored alongside the original comparison.

```python exec="true" source="material-block" result="text" session="scoring_comparisons"
--8<-- "snippets/cookbooks/scoring_comparisons.py:comparison_keys"
```


  [Polars]: https://pola.rs/