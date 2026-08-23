# Measures

Every comparison is built on a scoring measure, which decides what "different" means. Each measure
assigns every token a **contribution** $\delta_\tau$, saying how much that token adds to the overall
difference between the corpora.

Throughout, $R$ and $C$ are the reference and comparison corpora, and $p_\tau^{(R)}$ and
$p_\tau^{(C)}$ are the normalized frequency of a token $\tau$ in each.

## Weighted averages

Most of WordLevel's measures are **weighted averages**. Each token is given a score $\phi_\tau$, and
the corpus-level score is the average of those scores weighted by how often each token appears:

$$
\Phi = \sum_\tau p_\tau \phi_\tau
$$

Comparing two corpora then means taking the difference between their averages,
$\Phi^{(C)} - \Phi^{(R)}$. What separates one weighted-average measure from another is only what
$\phi_\tau$ means: a sentiment score, a surprisal, a comparison against a mixture. Each measure's
page below says what its $\phi_\tau$ is.

Two measures are not weighted averages. [Raw frequency](frequencies.md) differences work on counts
rather than normalized frequencies, and [rank-turbulence divergence](divergences.md#rank-turbulence-divergence) compares
rankings rather than averaging a score. They still assign contributions, but the decomposition below
does not apply, and they cannot be drawn as a
[word shift graph](../word_shift_graphs.md).

## How a contribution decomposes

For a weighted average, a token's contribution splits into two interpretable parts. When a token's
score is the same in both corpora, only the first part is possible: the token contributes because it
is used more or less often. When the score itself differs between the corpora, the second part
appears as well.

$$
\delta_\tau =
\underbrace{
\left(
    p_\tau^{(C)} - p_\tau^{(R)}
\right)
\left[
    \frac{1}{2}
    \left(
        \phi_\tau^{(C)} + \phi_\tau^{(R)}
    \right)
    - \Phi^{(\text{ref})}
\right]
}_{\text{used more or less}}
+
\underbrace{
\frac{1}{2}
\left(
    p_\tau^{(C)} + p_\tau^{(R)}
\right)
\left(
    \phi_\tau^{(C)} - \phi_\tau^{(R)}
\right)
}_{\text{scored higher or lower}}
$$

The first term is the token's change in frequency, weighted by how far its score sits from the
reference score. The second is its change in score, weighted by how common it is. A
[word shift graph](../word_shift_graphs.md) draws these two terms as the two stacked bars.

## Reference scores

$\Phi^{(\text{ref})}$ is the baseline each token's score is measured against. Its sign is what makes
a token count as relatively positive or relatively negative, so it decides which side of a comparison
the token falls on.

When a measure has a natural neutral point, such as the centre of a lexicon's scale, that is usually
the right choice. Whatever the measure, set it deliberately and report it.

## The order parameter

Several measures take an order parameter $\alpha$ that sets which part of the vocabulary the measure
attends to. When $\alpha < 1$, more weight is given to rare tokens; when $\alpha > 1$, more weight is
given to common tokens.

Because it decides whether a result is driven by a corpus's common words or its rare ones, $\alpha$
is worth choosing deliberately rather than left at its default.

## The measures

- [Frequencies](frequencies.md)
- [Lexicons](lexicons.md)
- [Entropies](entropies.md)
- [Divergences](divergences.md)
