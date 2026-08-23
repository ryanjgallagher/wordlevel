# Lexicons

A [lexicon](../../how_to/use_lexicons.md) assigns a score to each token, so a corpus can be
summarized by the average score of the words in it.

$$
\phi_\tau = \text{the score the lexicon gives } \tau
$$

The corpus-level score $\Phi = \sum_\tau p_\tau \phi_\tau$ is that average weighted by how often each
word is used, and comparing two corpora takes the difference between their averages.

This is the most direct weighted average, because $\phi_\tau$ is supplied rather than derived: if the
lexicon rates words for sentiment, the comparison is a difference in average sentiment.

A lexicon can be supplied for each corpus separately, in which case $\phi_\tau^{(R)}$ and
$\phi_\tau^{(C)}$ genuinely differ and the second term of
[the decomposition](index.md#how-a-contribution-decomposes) becomes meaningful: a word can
contribute because it was used more often, or because it means something different in the two
corpora, or both.

The natural reference score is the neutral point of the lexicon's own scale, so that words above it
read as positive and words below it read as negative.
