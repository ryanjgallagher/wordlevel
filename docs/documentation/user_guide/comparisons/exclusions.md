# Exclusions

We will pick up where [Scoring Comparisons](scoring.md) left off, with the rank divergence between
Roosevelt and Biden's speeches.

```python exec="true" source="above" session="exclusions"
--8<-- "snippets/user_guide/comparisons/exclusions.py:setup"
```

## Excluding Tokens

Now, let's look at the top 10 most distinguishing words according to the rank divergence.

<!-- Runs in the session but renders nothing: omitting `source` hides the block itself. -->
```python exec="true" session="exclusions"
--8<-- "snippets/user_guide/comparisons/exclusions.py:display_config"
```

```python exec="true" source="material-block" result="text" session="exclusions"
--8<-- "snippets/user_guide/comparisons/exclusions.py:top_tokens"
```


## Excluding by Frequency

Implement rank, so that you can filter "stop words" easily



## Excluding by Score

Sentiment