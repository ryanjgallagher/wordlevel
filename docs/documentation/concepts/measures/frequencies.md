# Frequencies

The simplest question to ask of two corpora is how often a word appears in each.

## Raw frequency

Comparing raw frequencies takes the difference in counts directly:

$$
\delta_\tau = f_\tau^{(C)} - f_\tau^{(R)}
$$

Positive contributions identify tokens that occur more often in the comparison; negative
contributions identify tokens that occur more often in the reference.

Raw frequency is **not a weighted average**. It works on counts rather than normalized frequencies,
so it has no $\phi_\tau$ and cannot be decomposed or drawn as a word shift graph. It also depends on
how much text each corpus contains: a corpus twice the size will tend to show larger counts
throughout, whether or not it uses a word differently.

## Proportion

Comparing proportions removes that dependence by asking how often a word appears *relative to* the
rest of its corpus:

$$
\delta_\tau = p_\tau^{(C)} - p_\tau^{(R)}
$$

Positive contributions identify tokens that occur relatively more often in the comparison.

Proportion **is** a weighted average, in the simplest possible way: every token is scored
$\phi_\tau = 1$. The corpus-level score $\Phi = \sum_\tau p_\tau \cdot 1$ is the same for both
corpora, so the second term of [the decomposition](index.md#how-a-contribution-decomposes) vanishes
and every contribution comes from a token being used more or less often.
