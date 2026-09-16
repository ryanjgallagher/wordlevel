# Word Shift Graphs

## Weighted averages

Many measures for comparing corpora can be represented as a difference in weighted averages. Let $p_\tau$ be the relative frequency of a token $\tau$ and let $\phi_\tau$ be the score assigned to that token. Then we can compute the difference:

$$
\Phi 
    = \sum_\tau p_\tau^{(C)} \phi_\tau - \sum_\tau p_\tau^{(R)} \phi_\tau
    = \sum_\tau \phi_\tau \left(p_\tau^{(C)} - p_\tau^{(R)} \right)
$$

where $C$ and $R$ indicate the comparison and reference corpora respectively. 

Observe how the overall difference is a linear sum over the tokens. This means that we can measure how each token contributes individually to the difference. We denote this as $\delta_\tau$. Further, its sign is determined entirely by whether $p_\tau^{(C)}$ or $p_\tau^{(R)}$ is larger—if it's positive, then we know that the token was used more in the comparison corpus, and vice versa. 


## Reference scores

There are times where we may be able to interpret each token's score qualitatively. For example, suppose that we have a sentiment lexicon that assigns a score $\phi_\tau$ to each token on a scale from 1 to 9, where 1 is the most "negative" a token can be, and 9 is the most "positive" it can be. The middle of this scale is 5. Therefore, if a token has a score below 5, then it is "relatively negative," and if it has a score above 5, then it is "relatively positive."

We can encode the qualitative notion of a score being "relatively" positive or negative by introducing a _reference score_, $\phi^{(\text{ref})}$. It is mathematically equivalent to rewrite our difference in weighted averages $\Phi$ as follows.

$$
\Phi = 
    \sum_\tau
    \underbrace{\left(p_\tau^{(C)} - p_\tau^{(R)} \right)}_{\uparrow / \downarrow}
    \underbrace{\left(\phi - \phi^{(\text{ref})} \right)}_{+ / -}
$$

The contribution of each token $\delta_\tau$ now depends on two things: whether it was used more or less in the comparison or reference corpus $\left(\uparrow / \downarrow \right)$, and whether its score is relatively positive or negative $\left( + / - \right)$. This means that there are four ways that a token can contribute, based on the signs of these components:

- $\left(+ \uparrow \right)$ A relatively positive token is used more.
- $\left(+ \downarrow \right)$ A relatively positive token is used less.
- $\left(- \uparrow \right)$ A relatively negative token is used more.
- $\left(- \downarrow \right)$ A relatively negative token  is used less.


## Basic word shifts

We rank the contributions by magnitude, and partition by these different qualitative interpretations. This is what is plotted in a _word shift graph_.

```python exec="true" source="above" session="word_shift_graphs"
--8<-- "snippets/concepts/word_shift_graphs.py:shift"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="word_shift_graphs"
--8<-- "snippets/concepts/word_shift_graphs.py:render"
```

In the word shift above, we are comparing the presidential speeches of Biden (the comparison corpus) to Roosevelt (the reference corpus). Overall, Biden's speeches have a higher sentiment than Roosevelt's speeches. Two types of contributes directly contribute to this:

- $\left(+ \uparrow \right)$ Relatively positive tokens are used more ("you", "america", "we", etc.).
- $\left(- \downarrow \right)$ Relatively negative tokens are used less ("war", "fighting", etc.).

Some tokens offset the higher sentiment, meaning that the difference would have been even more positive without them.

- $\left(+ \downarrow \right)$ Relatively positive tokens are used less ("great", "peace", "will", etc.).
- $\left(- \uparrow \right)$ Relatively negative tokens are used more ("violence", "tax", "cancer", etc.).

## Corpus-specific scores

Scores may be associated with a specific corpus. For example, we may have sentiment lexicons that are constructed for the context of each corpus—such as lexicons that encode how positive or negative words were in different years. Further though, when scores can depend on a specific corpus, they can be a function of the frequency itself. The Shannon entropy of a corpus, for example, measures the contribution of each token as $p_\tau \log 1 / p_\tau$.

We can represent corpus-specific scores by extending the word shift framework. The contributions $\delta_\tau$ of each token $\tau$ can be expressed as

$$
\delta_\tau =
    \underbrace{
        \left(
            p_\tau^{(C)} - p_\tau^{(R)}
        \right)
    }_{\uparrow / \downarrow}
    \underbrace{
        \left[
            \frac{1}{2}
            \left(
                \phi_\tau^{(C)} + \phi_\tau^{(R)}
            \right)
            - \Phi^{(\text{ref})}
        \right]
    }_{+ / -}
    +
    \underbrace{
        \frac{1}{2}
        \left(
            p_\tau^{(C)} + p_\tau^{(R)}
        \right)
        \left(
            \phi_\tau^{(C)} - \phi_\tau^{(R)}
        \right)
    }_{\bigtriangleup / \bigtriangledown}
$$

Like before, we can see that a token's contribution depends on whether it was used more or less $\left(\uparrow / \downarrow \right)$, and whether its score is relatively positive or negative $\left( + / -\right)$. However, it also now depends on the difference in the scores themselves $\left( \bigtriangleup / \bigtriangledown \right)$.

- $\left(\bigtriangleup \right)$ A score is higher within the comparison corpus.
- $\left(\bigtriangledown \right)$ A score is higher within the reference corpus.


## Detailed word shifts

Note how the new term in our word shift decomposition is additive. We represent this in the word shift plot as a stacked bar. 

```python exec="true" source="above" session="word_shift_graphs"
--8<-- "snippets/concepts/word_shift_graphs.py:detailed_shift"
```

<!-- Renders the chart: omitting `source` hides the plumbing that converts it to an SVG. -->
```python exec="true" html="true" session="word_shift_graphs"
--8<-- "snippets/concepts/word_shift_graphs.py:render"
```

This word shift graph tells a different story than before: when we use decade-specific sentiment lexicons, Biden's speeches are more _negative_ than Roosevelt's speeches. A large part of this is driven by specific tokens being associated with more negative sentiment.

- $\left( \bigtriangledown \right)$ Tokens have more negative scores ("nation", "americans", "country", etc.). These contribute to the lower sentiment.
- $\left( \bigtriangleup \right)$ Tokens have more positive scores ("together", "thank", "president", etc.). These offset the sentiment difference being even lower.

The two components of each contribution can counteract each other (one can be positive, the other can be negative). We display this by fading the bars by the counteracted component. For example, "nation" contributes negatively because its score is lower, but that is offset partly by the fact that it is a relatively negative word that was used less.


## Further reading

> Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J., Danforth, C. M., &
Dodds, P. S. (2021). [Generalized word shift graphs: a method for visualizing and explaining pairwise
comparisons between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
EPJ Data Science, 10(1), 4.
