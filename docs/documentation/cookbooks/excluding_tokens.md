# Excluding Tokens

Sometimes we do not want to include all tokens in our comparison. For example, we may want to remove exceedingly common tokens ("stop words") or exceedingly rare ones if we think that they may not contribute meaningfully to the difference between our corpora.

!!! note
    If you exclude tokens from a comparison, WordLevel automatically renormalizes all frequencies, ensuring that all scores are correctly calculated over the narrower vocabulary.


## Example dataset

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs. 

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:setup"
```

We will start by comparing them with the rank divergence. We can look at the most distinguishing tokens (the ones with the highest scores) before we exclude any. 

Observe how most of the top tokens are function words and contractions. While these are genuine differences, we may want to exclude them because they are not particularly insightful.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:display_config"
```

```python exec="true" source="material-block" result="text" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:top_tokens"
```


## Excluding specific tokens

If we know which specific tokens we want to drop—like a pre-defined list of stop words—we can exclude them specifically from the comparison.

All exclusion methods are called from the [`Comparison.exclude`](../api/comparison/exclude/index.md) namespace.


```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:tokens"
```

Because we excluded several of the previous top tokens, we can see that the new top tokens are different even though our comparison is the same in every other way.


```python exec="true" source="material-block" result="text" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:tokens_effect"
```


## Excluding by frequency

We can also exclude tokens by how often they do or do not appear. The `by_frequency` method can take one of three parameters: `less_than`, `greater_than`, or `between`. Below, we exclude all tokens that appear less than 10 times.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:by_frequency"
```

We can also use the normalized frequency, or proportion, to exclude tokens.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:by_proportion"
```

!!! note
    By default, a token is only excluded if it meets the criteria for _both_ corpora. For example, `exclude.by_frequency(less_than=10)` excludes tokens that appear less than 10 times in both. If it appeared 11 times in one corpus but not the other, then it would not be excluded.

    If you want the exclusion criteria to apply according to a specific corpus or either, see the [`corpus` parameter](../api/comparison/exclude/by_frequency.md) for details.

!!! warning
    A token missing from one corpus has a frequency and proportion of zero there, so specifying `corpus="either"` and the `less_than` parameter will remove every token that the two corpora do not share.

## Excluding by lexicon score

When a comparison is scored by a [lexicon](scoring_with_lexicons.md), tokens can be excluded by the score that the lexicon assigns them.

For example, say we score words with the [labMT](../api/lexicons/labMT.md) sentiment lexicon, which assigns scores on a scale from 1 to 9. To focus on tokens with "strong" sentiment, we can exclude those with "neutral" sentiment by filtering out tokens with scores between 4 and 6.

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

Exclusions can be chained together, each one narrowing the vocabulary further. Below, we drop our list of stop words _and_ tokens that appear more than 40 times in both corpora.

```python exec="true" source="above" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:chained_exclusions"
```

For our example, these exclusions capture different things. For example, "government" and "president" were top tokens [once the stop words alone were excluded](#excluding-specific-tokens), but they are removed by the additional frequency exclusion that we applied.

```python exec="true" source="material-block" result="text" session="excluding_tokens"
--8<-- "snippets/cookbooks/excluding_tokens.py:chained_exclusions_effect"
```
