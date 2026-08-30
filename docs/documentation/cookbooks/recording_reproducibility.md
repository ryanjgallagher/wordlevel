# Recording Steps for Reproducibility

Every comparison tracks the information needed to reproduce it: the corpora it was built from, the
scoring method and its parameters, any tokens that were excluded, and how the scores were
normalized. 

```python exec="true" source="above" session="recording_reproducibility"
--8<-- "snippets/cookbooks/recording_reproducibility.py:setup"
```

## Inspecting a comparison

The comparison records the reference corpus, the comparison corpus, the scoring method, the method's
parameters, and more.

```python exec="true" source="material-block" result="text" session="recording_reproducibility"
--8<-- "snippets/cookbooks/recording_reproducibility.py:comparison_metadata"
```

## Normalization

If the scores were [normalized](normalizing_scores.md), the method and the constant that was divided
out are recorded under the attributes `normalized`, `normalized_by`, and `normalization_constant`.

```python exec="true" source="material-block" result="text" session="recording_reproducibility"
--8<-- "snippets/cookbooks/recording_reproducibility.py:normalization_metadata"
```

See the [documentation](../api/comparison/comparison_result.md) for all of the tracked metadata.
