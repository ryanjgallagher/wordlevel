# Reading a Word Shift Graph

A word shift graph explains a difference between two corpora by breaking it into the contribution of
each word. It answers not just *which* words differ, but *how* they add up to the overall difference.

Word shift graphs can only be used with scores derived from weighted averages. For comparing corpora
using measures that are not weighted averages, use a [bar graph](../how_to/plots.md) instead.

## How to read one

All bars are interpreted in terms of how the comparison corpus differs from the reference corpus.

For each token, there are two stacked bars. One bar shows whether the token is relatively positive or
negative and whether it was used more or less in the comparison versus the reference. The other bar
shows whether the token's score is higher or lower in the comparison versus the reference. Tokens are
ranked by the magnitude of both stacked bars together.

## Where the bars come from

Let $\tau$ be a token, and let $p_\tau^{(\mathcal{C})}$ and $\phi_\tau^{(\mathcal{C})}$ be the
normalized frequency and score of $\tau$ in corpus $\mathcal{C}$ respectively. Then the contribution
$\delta_\tau$ of the token is:

$$
\delta_\tau
    =
    \overbrace{
        \biggl(
            p_\tau^{(C)} - p_\tau^{(R)}
        \biggr)
    }^{
        \uparrow / \downarrow
    }
    \overbrace{
        \biggl[
            \frac{1}{2}
            \left(
                \phi_\tau^{(C)} + \phi_\tau^{(R)} - \Phi^{(\text{ref})}
            \right)
        \biggr]
    }^{
        + / -
    }
    +
    \underbrace{
        \frac{1}{2}
        \biggl(
            p_\tau^{(C)} + p_\tau^{(R)}
        \biggr)
        \biggl(
            \phi_\tau^{(C)} - \phi_\tau^{(R)}
        \biggr)
    }_{
        \bigtriangleup / \bigtriangledown
    }
$$

where $R$ and $C$ are the reference and comparison corpora respectively, and $\Phi^{(\text{ref})}$ is
the user-provided reference score.

The stacked bars represent the two parts of the summand. The first bar is whether a word is
relatively positive $(+ / -)$ and appears more or less in the comparison corpus
$(\uparrow / \downarrow)$. The second bar is whether the score is more or less positive in the
comparison corpus $(\bigtriangleup / \bigtriangledown)$. Note, the bars may counteract one another if
the two summand terms are different signs: this is shown visually with faded bars in the word shift
plot when that is the case.

## Reference scores

The reference score $\Phi^{(\text{ref})}$ is the baseline that each word's score is measured against.
It appears in the first term above, where the two corpora's scores for a word are averaged and the
reference score is subtracted. Its sign is what makes a word count as relatively positive $(+)$ or
relatively negative $(-)$.

This makes the reference score a defining part of the comparison rather than a display setting. Move
it and words change which side of the shift they fall on, and the graph tells a different story about
the same data.

For a lexicon, the natural choice is the neutral point of the lexicon's own scale, so that words
above it read as positive and words below it read as negative. Whatever the measure, choose the
reference score deliberately and report it alongside the result.

## Further reading

> Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J., Danforth, C. M., &
Dodds, P. S. (2021). [Generalized word shift graphs: a method for visualizing and explaining pairwise
comparisons between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
EPJ Data Science, 10(1), 4.
