# Excluding Tokens

Excluding narrows a comparison to the vocabulary you are interested in. Tokens are removed from both
corpora before they are scored, so the frequencies the scoring measure works with are normalized over
whatever vocabulary remains.

We will pick up where [Scoring Comparisons](score_comparisons.md) left off, with the rank divergence
between Roosevelt and Biden's speeches.

```python exec="true" source="above" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:setup"
```

Here are the 10 most distinguishing words, before excluding anything.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:display_config"
```

```python exec="true" source="material-block" result="text" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:top_tokens"
```

Most of them are function words and contractions. They genuinely distinguish the two presidents, but
they may not be what you are asking about.

## Excluding specific tokens

Pass the tokens to drop. Exclusions are part of the comparison, so they chain before the scoring
method.

```python exec="true" source="above" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:tokens"
```

The same measure now surfaces a different vocabulary.

```python exec="true" source="material-block" result="text" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:tokens_effect"
```

!!! warning
    Excluding tokens changes the normalized frequencies of every token that remains, so scores are
    not comparable between a filtered and an unfiltered comparison.

## Excluding by frequency

Give a range of raw counts to drop. Exactly one of `less_than`, `greater_than`, and `between` sets
the range; call the method again to exclude by a second, separate criterion.

```python exec="true" source="above" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:by_frequency"
```

By default a token is only excluded if it falls in the range in **both** corpora. Pass
`corpus="either"`, `"reference"`, or `"comparison"` to change that.

## Excluding by proportion

A raw count means different things in corpora of different sizes. `by_proportion` uses each token's
share of its own corpus instead, so the same bounds mean the same thing on either side.

```python exec="true" source="above" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:by_proportion"
```

Proportions are taken over each corpus as the catalog gave it, before anything the comparison
excludes, so the bounds mean the same thing no matter what else is excluded.

!!! warning
    A token missing from one corpus has a frequency and proportion of zero there, so `corpus="either"`
    paired with `less_than` removes every token the two corpora do not share.

## Excluding by lexicon score

When a comparison is scored by a [lexicon](use_lexicons.md), tokens can be excluded by the score the
lexicon gives them. Dropping the middle of the scale keeps only the words that carry a strong
sentiment either way.

```python exec="true" source="above" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:by_lexicon_score"
```

## Keeping only shared vocabulary

`unshared_tokens` drops every token that does not appear in both corpora. This is what makes the
Kullback-Leibler divergence well defined, since it is infinite when the comparison corpus uses a word
the reference never does.

```python exec="true" source="above" session="exclude_tokens"
--8<-- "snippets/how_to/exclude_tokens.py:unshared_tokens"
```

!!! warning
    A word that only one corpus uses can be a stark difference between them. Prefer a measure that
    handles those absences over discarding them, where you can.

See the [exclusion documentation](../api/comparison/exclude/index.md) for the full set of parameters.
