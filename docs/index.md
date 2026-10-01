---
title: wordlevel
hide:
  - navigation
  - toc
---

![wordlevel](assets/images/full_logo.svg#only-light)
![wordlevel](assets/images/full_logo_dark_mode.svg#only-dark)


WordLevel is an open-source library for word-level natural language processing and text analysis.
It allows you to measure how two bodies of text differ token by token, and turn that measurement 
into a publication-ready [word shift graph](https://wordlevel.dev/documentation/concepts/word_shift_graphs/):
a chart that shows not only _which_ words distinguish the texts, but also _how_ they do so.

## Install

WordLevel runs on Python 3.10 and above, and installs with any Python package manager. See the
[install guide](documentation/install.md) for details.

```bash
pip install wordlevel
```


## Compare texts

The code below compares the speeches of Presidents Franklin D. Roosevelt and Joe Biden by
sentiment, scoring each president with a lexicon from each era.

```python
--8<-- "snippets/index.py:example"
```

<figure class="cover-figure" markdown>
![Word shift graph comparing the sentiment of Joe Biden's speeches against Franklin D. Roosevelt's](assets/images/cover_word_shift_graph.svg#only-light)
![Word shift graph comparing the sentiment of Joe Biden's speeches against Franklin D. Roosevelt's](assets/images/cover_word_shift_graph_dark.svg#only-dark)
</figure>

Biden's speeches score lower in sentiment overall. From the word shift graph, we can see that 
most of the gap comes from words like "nation," "families," and "americans" carrying lower sentiment 
in the 2000s lexicon than the 1940s one (▽), only partly offset by Biden using relatively positive 
words such as "together" and "well" more often (+↑).


## Where to go next

<div class="grid cards" markdown>

-   __Quick Start__

    ---

    Install the package, make your first comparison, and build a word shift graph.

    [:octicons-arrow-right-24: Get started](documentation/install.md)

-   __Cookbooks__

    ---

    Task-by-task recipes for constructing catalogs, scoring, and plotting.

    [:octicons-arrow-right-24: Browse the cookbooks](documentation/cookbooks/constructing_catalogs.md)

-   __Concepts__

    ---

    The ideas behind corpora, comparisons, and word shift graphs.

    [:octicons-arrow-right-24: Read the concepts](documentation/concepts/corpora_and_catalogs.md)

-   __API Reference__

    ---

    Every public class, method, and configuration, generated from the source.

    [:octicons-arrow-right-24: Look up the API](documentation/api/index.md)

</div>


## Citation

If you use WordLevel in your research, please cite the following paper:

> Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J., Danforth, C. M., &
> Dodds, P. S. (2021). [Generalized word shift graphs: a method for visualizing and explaining
> pairwise comparisons between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
> _EPJ Data Science_, 10(1), 4.