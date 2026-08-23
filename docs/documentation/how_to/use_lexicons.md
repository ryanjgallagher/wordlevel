# Using Lexicons

A **lexicon** assigns a score to each token, so a corpus can be summarized by the average score of
the words in it. Comparing two corpora then asks how that average differs, and which words drive the
difference.

```python exec="true" source="above" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:setup"
```

## Loading a pre-defined lexicon

WordLevel publishes several lexicons. Here is [labMT], which rates how happy or unhappy words feel.

```python exec="true" source="above" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:load_labmt"
```

Lexicons are downloaded the first time they are used and cached afterwards, so later loads do not
need a network connection.

## Inspecting a lexicon

A lexicon holds its scores in a dataframe, along with the `center` of its scale.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:display_config"
```

```python exec="true" source="material-block" result="text" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:inspect_lexicon"
```

## Scoring a comparison

Pass the lexicon to `score.lexicon`. The `reference_score` sets the neutral point that each word is
measured against, and determines which side of the comparison a word falls on, so set it
deliberately. Here we use the lexicon's own center.

```python exec="true" source="above" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:score_with_lexicon"
```

## Using your own scores

A lexicon does not have to be one of the pre-defined ones. Any mapping from tokens to scores works,
as does a Polars dataframe with a token column and a score column. Set the `reference_score` to
whatever counts as neutral on your scale.

```python exec="true" source="above" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:custom_lexicon"
```

## A different lexicon per corpus

Sometimes each corpus should be scored by its own lexicon, such as when word meanings drift between
two periods and each period has its own scores. Pass `lexicon_reference` and `lexicon_comparison`
instead of a single `lexicon`.

Each lexicon scores only the corpus it belongs to. Below, `"before"` is the reference corpus, so its
words are scored by `scores_before`, and `"after"` is the comparison corpus, so its words are scored
by `scores_after`. The same word can therefore carry a different score in each corpus, which is the
point of scoring them separately.

```python exec="true" source="above" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:per_corpus_lexicons"
```

## Borrowed and dropped tokens

A token can be scored in one lexicon but missing from the other. Above, `"happy"` is only in
`scores_before` and `"sad"` is only in `scores_after`. WordLevel keeps those tokens and borrows the
score from the other corpus, so they still contribute to the comparison.

A token missing from *both* lexicons is a different case: there is no score to borrow, so it is
dropped from the comparison entirely. That is why `"the"` does not appear below, even though it is
the most frequent token in both corpora.

```python exec="true" source="material-block" result="text" session="use_lexicons"
--8<-- "snippets/how_to/use_lexicons.py:scored_tokens"
```

!!! warning
    Dropping unscored tokens changes what the comparison is measuring: it is a comparison over the
    lexicon's vocabulary, not over the full vocabulary of the corpora. How much of your text a
    lexicon covers is worth checking before drawing conclusions from it.

See the [lexicon documentation](../api/lexicons/index.md) for every lexicon WordLevel publishes, and
the [scoring documentation](../api/comparison/score/lexicon.md) for the full set of parameters.


  [labMT]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026752
