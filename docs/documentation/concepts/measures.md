# Measures

Every comparison is built on a scoring measure, which decides what "different" means. This page
describes what each family of measures asks. For how to apply them, see
[Scoring Comparisons](../how_to/score_comparisons.md); for the formulas and parameters, see the
[scoring documentation](../api/comparison/score/index.md).

## Frequencies and proportions

The simplest question is how often a word appears. Each token's contribution is the difference in
its raw frequency between the corpora. Positive contributions identify tokens that occur more often
in the comparison; negative contributions identify tokens that occur more often in the reference.

Raw counts depend on how much text each corpus contains, so comparing proportions instead asks how
often a word appears *relative to* the rest of its corpus. Each token's contribution is then the
difference in its normalized frequency within each corpus, and positive contributions identify
tokens that occur relatively more often in the comparison.

## Lexicon-weighted averages

A lexicon associates a score with each token. The overall score for each corpus individually is the
lexicon-weighted average: if $p_\tau$ is the normalized frequency of a token $\tau$ in a corpus, and
$\phi_\tau$ is the score assigned to it by the lexicon, then the corpus-level score is
$\Phi = \sum_\tau p_\tau \phi_\tau$. Comparing the corpora takes the difference in weighted averages
between them, $\Phi^{(C)} - \Phi^{(R)}$.

## Entropy and surprisal

Shannon entropy is a measure of textual diversity and "surprise." If $p_\tau$ is the normalized
frequency of a token $\tau$ in a corpus, then its Shannon entropy is
$\sum_\tau p_\tau \log \frac{1}{p_\tau}$. The factor $\log \frac{1}{p_\tau}$ is known as the
"surprisal" of a token: the less frequent a token, the higher its surprisal.

Tsallis entropy generalizes this, parameterized by an order $\alpha$. When $\alpha = 1$, the Tsallis
entropy reduces to the Shannon entropy.

## The order parameter

Several measures take an order parameter $\alpha$ that sets which part of the vocabulary the measure
attends to. When $\alpha < 1$, more weight is given to rare tokens; when $\alpha > 1$, more weight is
given to common tokens.

Because it decides whether a result is driven by a corpus's common words or its rare ones, $\alpha$
is worth choosing deliberately rather than left at its default.

## Divergences

The Kullback-Leibler divergence is an asymmetric measure of how two corpora differ. From an
information-theoretic perspective, it measures how much more information is needed to encode the
comparison corpus when using the reference as the "codebook."

The total divergence is always non-negative, but individual per-token contributions can be either
sign. A positive contribution identifies a token that appears relatively more in the comparison
corpus; a negative contribution, more in the reference. Because the total is non-negative, positive
contributions reinforce the divergence and negative contributions offset it.

!!! warning
    The Kullback-Leibler divergence is only well-defined when all of the words in the comparison
    also appear in the reference. Otherwise it diverges to infinity. Either
    [exclude the unshared tokens](../how_to/exclude_tokens.md) or use a symmetric measure.

The Jensen-Shannon divergence is a symmetric, bounded measure of how two corpora differ. It is
computed by averaging the Kullback-Leibler divergence of each corpus against a synthetic mixture
corpus representing the combination of both corpora, which is what makes it well-defined even when
the two vocabularies do not match.

## Rank-turbulence divergence

Rank-turbulence divergence compares corpora by comparing the ranks of tokens in each corpus when
ordered by relative frequency. The measure is symmetric and bounded between 0 and 1, where 0
indicates that the corpora are identical, and 1 indicates that they are maximally divergent.

Because it works on ranks rather than frequencies, it does not require the corpora to be of
comparable size, and it is not dominated by the handful of words that are most frequent in both.
