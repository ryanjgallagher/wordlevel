# Entropies

Entropy measures textual diversity: how spread out a corpus's vocabulary is, rather than what its
words mean.

## Shannon entropy

Shannon entropy scores each token by its **surprisal**, how unexpected the token is:

$$
\phi_\tau = \log_b \frac{1}{p_\tau}
$$

The less frequent a token, the higher its surprisal. The corpus-level score
$\Phi = \sum_\tau p_\tau \log_b \frac{1}{p_\tau}$ is the average surprisal of the corpus, which is
its Shannon entropy.

Because $\phi_\tau$ depends on the token's frequency in its own corpus, a word's score genuinely
differs between the two corpora, and both terms of
[the decomposition](index.md#how-a-contribution-decomposes) are in play.

## Tsallis entropy

Tsallis entropy generalizes this with an [order parameter](index.md#the-order-parameter) $\alpha$,
replacing the surprisal with:

$$
\phi_\tau = \frac{1}{\alpha - 1}\left(1 - p_\tau^{\alpha - 1}\right)
$$

When $\alpha = 1$ this reduces to the Shannon surprisal, and the Tsallis entropy reduces to the
Shannon entropy. Lowering $\alpha$ pushes weight toward rare words; raising it pushes weight toward
common ones.
