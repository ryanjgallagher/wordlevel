# Comparisons

## What is a comparison?

A **comparison** quantifies the token-level differences between two corpora. It is a scientific plan
for how to compare the corpora, including how to measure their differences, which tokens to include,
and more. If you are familiar with [Polars], this is similar to how
[expressions](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) are plans for how
to transform and analyze columns in a dataframe.

The core of every comparison is a _scoring method_ for quantifying the token-level differences. 
WordLevel implements a [full suite of measures](measures/index.md), including 
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
[Jensen-Shannon divergence](measures/divergences.md#jensen-shannon-divergence) or 
[rank turbulence divergence](measures/divergences.md#rank-turbulence-divergence), are symmetric by 
default, meaning that the token-level difference is the same regardless of which corpus is the 
reference and which is the comparison. However, it can still be important to track which corpus is 
which, particularly for interpreting a [word shift graph](word_shift_graphs.md). See
[Measures](measures/index.md) for details on how to interpret each measure.

## How tokens are attributed

Every comparison splits its tokens into two groups. What those groups mean depends on the plot.

A [word shift graph](word_shift_graphs.md) reads the same way whatever the measure. It breaks each
token's contribution into its parts, and every bar says how the comparison corpus differs from the
reference: whether the token was used more or less, and whether it scored higher or lower.

A [bar graph](../how_to/plots.md) does not decompose anything, so its two sides mean one of two
things. Under some measures each token is attributed to a **corpus**, the one it is most
characteristic of. Under others each token is attributed by whether it **reinforces or offsets** the
overall difference: a contribution with the same sign as the overall difference reinforces it, making
its magnitude larger, while a contribution with the opposite sign offsets it, pulling the difference
back toward zero.

A comparison records which of the two it uses under its `attributed_by` attribute, which is also what
decides the [configuration](../how_to/customize_plots.md) a bar graph takes.

  [Polars]: https://pola.rs/
