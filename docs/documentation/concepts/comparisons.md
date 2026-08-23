# Comparisons

## What is a comparison?

A **comparison** quantifies the token-level differences between two corpora. It is a scientific plan
for how to compare the corpora, including how to measure their differences, which tokens to include,
and more. If you are familiar with [Polars], this is similar to how
[expressions](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) are plans for how
to transform and analyze columns in a dataframe.

The core of every comparison is a _scoring method_ for quantifying the token-level differences. 
WordLevel implements a [full suite of measures](../api/comparison/score/index.md), including 
basic frequency differences, entropy-based metrics like the Kullback-Leibler and Jensen-Shannon 
divergences, and other advanced measures.

## Reference and comparison corpora

Each comparison has two corpora: a **_reference_** corpus and a **_comparison_** corpus. The 
reference corpus can be considered the baseline from which the comparison corpus deviates. It is
important to be mindful of which corpus is which, since it can affect the interpretation of the
results. 

## Why direction matters

For example, say we have two corpora: the speeches of President Franklin D. Roosevelt and
the speeches of President Joe Biden. Suppose Roosevelt's speeches are the reference corpus and 
Biden's are the comparison corpus, and we want to simply measure the difference in how often 
different words appear. Since Roosevelt is our reference, this means we are calculating 
$f^{(\text{Biden})}_\tau - f^{(\text{Roosevelt})}_\tau$, where $f^{(C)}_\tau$ is the frequency of 
token $\tau$ in corpus $C$. If the difference is positive, then it means that the word was used more 
by Biden. Similarly, if it is negative, then it means it was used less by Biden (more by Roosevelt).

## Symmetric measures

Some scoring methods, like the 
[Jensen-Shannon divergence](../api/comparison/score/jensen_shannon_divergence.md) or 
[rank turbulence divergence](../api/comparison/score/rank_divergence.md), are symmetric by 
default, meaning that the token-level difference is the same regardless of which corpus is the 
reference and which is the comparison. However, it can still be important to track which corpus is 
which, particularly for interpreting shift plots. See the 
[scoring documentation](../api/comparison/score/index.md) for details on how to interpret each 
measure.

<!-- TODO: add documentation link to shift plots above -->


  [Polars]: https://pola.rs/
