# Excluding Tokens

Sometimes we do not want to include all tokens in our comparison. For example, we may want to remove exceedingly common tokens ("stop words") or exceedingly rare ones if we think that they may not contribute meaningfully to the difference between our corpora.

!!! note
    If you exclude tokens from a comparison, WordLevel automatically renormalizes all normalized frequencies, ensuring that all scores are correctly calculated over the narrower vocabulary.


## Example data

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs. 

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:setup"
```

We will start by comparing them with the rank-turbulence divergence. We can look at the most distinguishing tokens (the ones with the highest scores) before we exclude any. Observe how most of the top tokens are function words and contractions. While these are genuine differences, we may want to exclude them because they are not particularly insightful.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:display_config"
```

```python exec="true" source="material-block" result="text" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:top_tokens"
```


## Excluding specific tokens

If we know which specific tokens we want to drop—like a pre-defined list of stop words—we can exclude them from the comparison.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:tokens"
```

Because we excluded several of the previous top tokens, we can see that the new top tokens change, even though the two comparisons are otherwise identical.

```python exec="true" source="material-block" result="text" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:tokens_effect"
```


## Excluding by frequency

We can also exclude tokens by how often they appear. The `by_frequency` method takes exactly one of three bounds: `less_than`, `greater_than`, or `between`. Below, we exclude all tokens that appear 10 or fewer times.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:by_frequency"
```

We can also use the normalized frequency, or proportion, to exclude tokens.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:by_proportion"
```

!!! note
    By default, a token is only excluded if it meets the criteria for _both_ corpora. For example, `exclude.by_frequency(less_than=10)` excludes tokens that appear 10 or fewer times in both the reference and comparison corpora. If it appeared 11 times in one corpus and 3 times in the other, then it would not be excluded.

    To apply the exclusion criteria against only one corpus, or either corpus, see the [`corpus`](../api/comparison/exclude/by_frequency.md) parameter for details.

!!! warning
    A token missing from one corpus has a frequency and proportion of zero there, so specifying `corpus="either"` and the `less_than` parameter will remove every token that the two corpora do not share, in addition to tokens below the bound.

## Excluding by rank

The exact frequency of a token may vary in corpora of different sizes. At times, then, we may want to exclude tokens by their rank. That is, instead of saying "Keep all tokens that were used at least 20 times," we may want to say "Keep the 1,000 most frequently used tokens" or "Remove the 10 most frequently used tokens." 

The `by_rank` method takes exactly one of three bounds: `within_rank`, `beyond_rank`, or `between`. Rank 1 indicates the most frequently used token in a corpus. In the example below, we use the `within_rank` parameter to exclude the ten most frequently used tokens in each corpus.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:within_rank"
```

Instead of excluding the most common tokens, we can keep them by excluding all other tokens with the `beyond_rank` parameter. In the example below, we exclude any tokens that are not used more frequently than rank 1,000 in both corpora. Note, this means that if a token is ranked at 998 in one corpus and 1,002 in the other, then it is kept because it is within rank 1,000 in at least one of them.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:beyond_rank"
```

Within a corpus, tokens may tie for how frequently they are used. This is particularly likely to happen in the tail of a frequency distribution, where many tokens may have been used exactly one time, for example. Tied tokens form a "block" where all tokens in that block receive the same rank, and the `tie_method` parameter determines what rank is given to them: `min` gives the block the first of the ranks it spans, `max` the last, and `average` the mean of them. See the [documentation](../api/types/rank_ties.md) for more details.

!!! note
    In the example above demonstrating how to use `within_rank`, we chain two exclusions to remove the top ten tokens for each corpus individually. This is equivalent to doing a single exclusion and setting `corpus="either"`. 

## Excluding by lexicon score

When a comparison is scored by a [lexicon](scoring_with_lexicons.md), tokens can be excluded by the score that the lexicon assigns them.

For example, say that we score tokens with the [labMT](../api/lexicons/off-the-shelf/labMT.md) sentiment lexicon, which assigns scores on a scale from 1 to 9. To focus on tokens with "strong" sentiment, we can exclude those with "neutral" sentiment by filtering out tokens with scores between 4 and 6 (inclusive).

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:by_lexicon_score"
```

## Narrowing to a shared vocabulary

Some measures, like the Kullback-Leibler divergence, are only well defined if both corpora have the same vocabulary. We can exclude tokens that are not shared between the two corpora.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:unshared_tokens"
```

!!! warning
    A token that only appears in one corpus but not the other may strongly indicate how the corpora are different. Excluding it removes that signal. It is preferable to use measures that are robust to different vocabularies, instead of excluding tokens to create a common vocabulary.

## Chaining multiple exclusions

Exclusions can be chained together, each one narrowing the vocabulary further. Below, we drop our list of stop words _and_ tokens that appear more than 40 or more times in both corpora.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:chained_exclusions"
```

The tokens "government" and "president" were top distinguishing tokens [when we only excluded stop words](#excluding-specific-tokens), but they are removed by the additional frequency exclusion.

```python exec="true" source="material-block" result="text" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:chained_exclusions_effect"
```
