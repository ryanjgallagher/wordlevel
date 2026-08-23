# Normalizing Scores

It can sometimes be difficult to interpret the individual token-level scores. To make them more
interpretable, we can normalize them in different ways. 

```python exec="true" source="above" session="normalize_scores"
--8<-- "snippets/how_to/normalize_scores.py:setup"
```

## By Total Difference

One way is to normalize by the sum of the absolute scores—if each score is represented by $\delta$, 
then we normalize by $\sum | \delta |$. This is the _total_ difference between the corpora, 
regardless of the score signs. We can interpret each normalized score as the share of that total 
difference accounted for by an individual token. For example, a score of 0.05 means that a token 
makes up 5% of the total difference between the corpora. A score of -0.05 also means that a token 
makes up 5% of the total difference, but contributes in the negative direction. The absolute values 
of the normalized scores sum to 1.

```python exec="true" source="above" session="normalize_scores" hl_lines="4"
--8<-- "snippets/how_to/normalize_scores.py:normalize_sum_abs"
```

## By Net Difference

We can also normalize by the magnitude of the overall score, $| \sum \delta |$. This is the _net_
difference between the corpora. We can interpret each normalized score as how large a token's
contribution is relative to the overall, net difference—it is the ratio between them. For example,
a score of 0.05 means that a token's contribution is 5% the size of the net difference. A score of
-1.05 means that a token's contribution is 105% the size of the net difference, and it contributes
in the negative direction. Even though individual normalized scores can exceed 1 in magnitude,
they still sum to $\pm 1$ because opposing contributions can cancel one another out.

```python exec="true" source="above" session="normalize_scores" hl_lines="4"
--8<-- "snippets/how_to/normalize_scores.py:normalize_abs_sum"
```

!!! note
    For scoring measures that produce token-level scores that are always non-negative, like the 
    [rank divergence](../api/comparison/score/rank_divergence.md) and 
    [Jensen-Shannon divergence](../api/comparison/score/jensen_shannon_divergence.md), the
    different forms of normalization produce the same result. For other measures, where tokens' 
    scores can be positive or negative, the normalizations can produce different results.

!!! warning
    The net difference of normalized frequencies is always zero. This means that only `"sum_abs"` 
    can be used as a normalization method when scoring by `proportion`. Using `"abs_sum"` will
    raise an error.

!!! warning
    Normalizing scores causes them to be eagerly evaluated because the normalization needs to verify
    that the normalization constant is nonzero. 

The normalization method used is recorded on the comparison itself; see
[Tracking Reproducibility](reproducibility.md).
