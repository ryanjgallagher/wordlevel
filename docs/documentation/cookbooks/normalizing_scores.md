# Normalizing Scores

We can put comparison scores on a common scale by normalizing them. This allows us to easily interpret the scores relative to one another, rather than in terms of their raw units.

In a comparison, each token $\tau$ makes a contribution $\delta_\tau$ to the difference. The overall difference between the two corpora is $\sum \delta_\tau$.


## Example data

We will use an example dataset of US presidential speeches from Franklin D. Roosevelt and Joe Biden. Each document is a speech, and each corpus contains all of the speeches by that particular president. See the [cookbook](constructing_catalogs.md) for more about working with catalogs.

```python exec="true" source="above" session="normalizing_scores"
--8<-- "snippets/cookbooks/normalizing_scores.py:setup"
```


## Normalizing by total difference

The first way that we can normalize scores is by the _total_ difference, which is equivalent to dividing all of the scores by $\sum | \delta_\tau |$. This allows us to interpret each token's score as a percent of the total difference between the two corpora.


```python exec="true" source="above" session="normalizing_scores"
--8<-- "snippets/cookbooks/normalizing_scores.py:normalize_sum_abs"
```

## Normalizing by net difference

The second way that we can normalize is by the _net_ difference, which is equivalent to dividing all of the scores by $\left| \sum \delta_\tau \right|$. This allows us to interpret each token's score as its size relative to that net difference.


```python exec="true" source="above" session="normalizing_scores" 
--8<-- "snippets/cookbooks/normalizing_scores.py:normalize_abs_sum"
```

!!! warning
    Normalizing scores causes them to be eagerly evaluated because the normalization needs to verify
    that the normalization constant is nonzero. 
