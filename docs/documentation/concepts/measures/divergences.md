# Divergences

A divergence measures how much one corpus's distribution of words departs from another's.

## Kullback-Leibler divergence

The Kullback-Leibler divergence is asymmetric: it measures how much more information is needed to
encode the comparison corpus when using the reference as the "codebook." Each token is scored by its
surprisal under the *other* corpus's distribution, so a token's contribution is:

$$
\delta_\tau = p_\tau^{(C)} \log_b \frac{p_\tau^{(C)}}{p_\tau^{(R)}}
$$

The total divergence is always non-negative, but individual contributions can be either sign. A
positive contribution identifies a token that appears relatively more in the comparison corpus; a
negative contribution, more in the reference. Because the total is non-negative, positive
contributions reinforce the divergence and negative contributions offset it.

!!! warning
    The Kullback-Leibler divergence is only well-defined when every word in the comparison corpus
    also appears in the reference. A word the reference never uses has $p_\tau^{(R)} = 0$, and the
    divergence goes to infinity. Either
    [exclude the unshared tokens](../../how_to/exclude_tokens.md) or use a symmetric measure.

## Jensen-Shannon divergence

The Jensen-Shannon divergence is symmetric and bounded. It is built by comparing each corpus against
a synthetic **mixture** corpus $M$ representing the combination of the two, and averaging the
results. Each token is scored by how far its own corpus sits from that mixture:

$$
\phi_\tau^{(R)} = w_R \log_b \frac{p_\tau^{(M)}}{p_\tau^{(R)}}
\qquad
\phi_\tau^{(C)} = w_C \log_b \frac{p_\tau^{(C)}}{p_\tau^{(M)}}
$$

where $w_R$ and $w_C$ are the weights given to each corpus in the mixture.

Because the mixture contains every word either corpus uses, $p_\tau^{(M)}$ is never zero where a
token occurs. That is what makes the Jensen-Shannon divergence well-defined even when the two
vocabularies do not match, and why it is the usual answer when the Kullback-Leibler divergence
diverges.

It also takes an [order parameter](index.md#the-order-parameter) $\alpha$, which replaces the
logarithms above with the generalized form used by the [Tsallis entropy](entropies.md).

## Rank-turbulence divergence

Rank-turbulence divergence compares corpora by their **rankings** rather than their frequencies. Each
corpus's vocabulary is ordered by relative frequency, and the measure asks how far each token moves
between the two orderings.

It is symmetric and bounded between 0 and 1, where 0 indicates that the corpora are identical and 1
that they are maximally divergent.

Unlike the two divergences above, rank-turbulence divergence is **not a weighted average**. It has no
per-token score $\phi_\tau$, so [the decomposition](index.md#how-a-contribution-decomposes) does not
apply and it cannot be drawn as a [word shift graph](../word_shift_graphs.md). Use a
[bar graph](../../how_to/plots.md) instead.

Working on ranks rather than frequencies has two consequences worth knowing. The corpora do not need
to be comparable in size, because a ranking is already relative to its own corpus. And the result is
not dominated by the handful of words that are most frequent in both, since a word that is common in
both corpora occupies a similar rank in each and so contributes little.

It takes an [order parameter](index.md#the-order-parameter) $\alpha$ that sets whether differences
among rare or common words matter more.
