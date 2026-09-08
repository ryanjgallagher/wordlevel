# Scoring with Lexicons

A lexicon assigns scores to tokens. For example, sentiment dictionaries are some of the most popular lexicons: words are given a score based on how "positive" or "negative" they are. 

When tokens are given scores, we can score an entire corpus by taking the weighted average $\sum \phi_\tau \, p_\tau$, where $\phi_\tau$ is the score given to a token $\tau$ by the lexicon, and $p_\tau$ is how relatively often it appears in the corpus.


## Example data

We will use two small corpora and put them into a `Catalog`. See the [cookbook](constructing_catalogs.md) for more about working with catalogs.

```python exec="true" source="above" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:setup"
```

## Loading a lexicon

WordLevel provides [several pre-made lexicons](../api/lexicons/index.md), constructed by academic researchers. They can be loaded using the `wl.lex` operator.

Below, we load the [labMT](../api/lexicons/labMT.md) sentiment lexicon, which rates how positive ("happy") or negative ("sad") a word is on a scale from 1 to 9.

```python exec="true" source="above" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:load_labmt"
```

!!! note
    The first time that you load a lexicon it is downloaded from [HuggingFace](https://huggingface.co/datasets/ryanjgallag/wordlevel-lexicons). The lexicon is cached locally, so subsequent calls do not download it repeatedly.


## Inspecting a lexicon

The lexicon's mapping between tokens and scores is stored as a dataframe `df`. 

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:display_config"
```

```python exec="true" source="material-block" result="text" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:inspect_lexicon"
```

The lexicon also records the `center` of its scale. For labMT, this is the score at the middle of the scale indicating that a word is "neutral"—neither positive nor negative.

```python exec="true" source="material-block" result="text" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:inspect_center"
```


## Scoring a comparison

We can compare our corpora by taking the difference in their weighted averages. That is, we can compute 

$$
\sum \phi_\tau \, p^{(C)}_\tau - \phi_\tau \, p^{(R)}_\tau
$$

where $p_\tau^{(C)}$ and $p_\tau^{(R)}$ are relative frequencies of a token $\tau$ in the comparison corpus $C$ and reference corpus $R$ respectively.

To do this, we can pass our lexicon directly to the comparison's `score.lexicon` method. See the [cookbook](scoring_comparisons.md) for more about scoring comparisons in general.

```python exec="true" source="above" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:score_with_lexicon"
```


## Scoring with a custom lexicon

You can also use a custom lexicon to make comparisons. The `score.lexicon` method accepts a dictionary or dataframe mapping of tokens to scores.

```python exec="true" source="above" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:custom_lexicon"
```


## Dropped tokens

Lexicons can only score the tokens that appear in them. All other tokens are dropped from the calculation. For example, "the" is the most common token in both of our corpora, but there is no score for it in our custom lexicon, so it is dropped from the comparison.

```python exec="true" source="material-block" result="text" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:dropped_tokens"
```


## Scoring with multiple lexicons

At times, we may have different lexicons for each corpora. For example, tokens may be associated with different sentiments at different periods of time (e.g. "catfish" or "amazon" in 1920 vs. 2020) or social contexts (e.g. "trump" in left- and right-leaning news outlets).

We can still compute the difference in the weighted averages of the corpora, but now the scores depend on the reference and comparison corpora.

$$
\sum \phi^{(C)}_\tau \, p^{(C)}_\tau - \phi^{(R)}_\tau \, p^{(R)}_\tau
$$

For our example data, we create two custom lexicons: one associated with our `before` corpus and one associated with our `after` corpus. We then pass these to the `lexicon_reference` and `lexicon_comparison` parameters when scoring the comparison.

```python exec="true" source="above" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:per_corpus_lexicons"
```


## Borrowing scores

When we have different lexicons per corpora, some tokens may be scored in both lexicons, but others may only appear in one or the other. For example, the words "doomscrolling" and "deepfake" did not exist until the 2010s, so they cannot have a sentiment score in a lexicon specific to the 1920s. Other words like "apple" and "cloud" are used more often today than in the past (because of their associations with technology), and so they may not appear in a lexicon of frequently used words in the 1920s.

In these cases, we can allow a token to "borrow" a score across lexicons. Effectively, this allows us to just compare the change in the relative frequency of the token, even if we cannot measure the change in its score across the corpora.

In our example lexicons, "happy" is only defined in the `before ` lexicon, and "sad" is only defined in the `after` lexicon. We can see that those tokens "borrow" their scores across lexicons because we set `impute="borrow"` when creating our comparison.


```python exec="true" source="material-block" result="text" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:scored_tokens"
```

## Setting reference scores

Many lexicons—particularly sentiment lexicons—are built on binary scales from a score that is most "negative" to a score that is most "positive." This means a each token is relatively negative or positive: "sad" has negative sentiment, while "happy" has a positive sentiment. If we distinguish between these tokens, it can change how we interpret our comparison. For example, a corpus can have a higher sentiment than another because it either used more positive words or it used less negative words.

To encode this, we can pass a _reference score_ to the comparison. Each score is compared to this anchor, determining whether it is relatively positive or negative.

```python exec="true" source="above" session="scoring_with_lexicons"
--8<-- "snippets/cookbooks/scoring_with_lexicons.py:reference_score"
```

!!! warning
    Since the reference score can change the _per token_ scores (though not the overall difference), they should only be used if you understand how to interpret [word shift graphs](../concepts/word_shift_graphs.md).


