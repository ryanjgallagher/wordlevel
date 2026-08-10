# Comparisons

A **comparison** quantifies the word-level differences between two corpora.

## Example Dataset

To work with comparisons, we will use an example dataset of presidential speeches. Each document is 
a speech by a US president, and each corpus is the set of all speeches by a particular president. 
We will use the speeches by Presidents Joe Biden and Donald Trump.

```python
import wordlevel as wl

speeches = wl.Dataset(
    "presidential_speeches", 
    corpora=["Joe Biden", "Donald Trump"]
)
```

## Scoring




## Exclusions