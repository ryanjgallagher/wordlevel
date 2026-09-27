# Normalization

Each scoring measure produces a per-token contribution $\delta_\tau$. Normalization divides every contribution by a single constant so that scores become interpretable and comparable across measures. Two normalization strategies are available.


## By total difference

One way to normalize scores is by the sum of their absolute scores. 

This is equivalent to normalizing by $\sum | \delta_\tau |$. This is the _total_ difference between the corpora, regardless of the score signs. We can interpret each normalized score as the share of that total difference accounted for by an individual token. For example, a score of 0.05 means that a token makes up 5% of the total difference between the corpora. A score of -0.05 also means that a token makes up 5% of the total difference, but contributes in the negative direction. The absolute values of the normalized scores sum to 1.

When using [`Comparison.normalize`](../api/comparison/manipulate/normalize.md), this is specified by setting `by="total_diff"`. 

## By net difference

Another way to normalize scores is by the magnitude of the overall score. 

This is equivalent to normalizing by $| \sum \delta_\tau |$. This is the _net_ difference between the corpora. We can interpret each normalized score as how large a token's contribution is relative to the overall, net difference—it is the ratio between them. For example, a score of 0.05 means that a token's contribution is 5% the size of the net difference. A score of -1.05 means that a token's contribution is 105% the size of the net difference, and it contributes negatively to the difference. Even though individual normalized scores can exceed 1 in magnitude, they still sum to $\pm 1$ because opposing contributions can cancel one another out.

When using [`Comparison.normalize`](../api/comparison/manipulate/normalize.md), this is specified by setting `by="net_diff"`. 
