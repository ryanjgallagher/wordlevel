# Normalizing Scores

Normalizing rescales every token's contribution by a single constant, putting the scores on a
common scale. For how to choose between the two methods and how to read the results, see
[Normalization](../concepts/normalization.md).

```python exec="true" source="above" session="normalize_scores"
--8<-- "snippets/how_to/normalize_scores.py:setup"
```

## By Total Difference

Normalize by $\sum | \delta |$, so that each score is a token's share of the _total_ difference
between the corpora.

```python exec="true" source="above" session="normalize_scores" hl_lines="4"
--8<-- "snippets/how_to/normalize_scores.py:normalize_sum_abs"
```

## By Net Difference

Normalize by $| \sum \delta |$, so that each score is sized relative to the _net_ difference
between the corpora.

```python exec="true" source="above" session="normalize_scores" hl_lines="4"
--8<-- "snippets/how_to/normalize_scores.py:normalize_abs_sum"
```

!!! warning
    The net difference of normalized frequencies is always zero. This means that only `"sum_abs"` 
    can be used as a normalization method when scoring by `proportion`. Using `"abs_sum"` will
    raise an error.

!!! warning
    Normalizing scores causes them to be eagerly evaluated because the normalization needs to verify
    that the normalization constant is nonzero. 

The normalization method used is recorded on the comparison itself; see
[Tracking Reproducibility](reproducibility.md).
