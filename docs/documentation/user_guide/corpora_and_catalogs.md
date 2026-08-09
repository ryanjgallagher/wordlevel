# Corpora & Catalogs

## Corpus

### What is a corpus?

A **corpus** is a collection of documents, where each document contains an ordered sequence of
tokens. A **token** is the smallest unit of text that we care about. Often, tokens are individual
words separated by white space. For example, we can break the sentence "WordLevel is a Python
package" into the tokens `("WordLevel", "is", "a", "Python", "package")`. In other cases, "tokens"
may refer to *n*-grams (*n* consecutive words) or subword pieces. We refer to the full set of unique
tokens as the **vocabulary.**

If we have multiple collections of documents across more than one corpus, then we have 
**corpora.** For example, we may say that all of the social media posts made on June 1st are one 
corpus (where each post is a document), and all of the posts made on June 2nd are another corpus. 
The June 1st corpus and June 2nd corpus together are our corpora. 

!!! warning
    A corpus is meant to be a relatively _large_ collection of text containing at least several
    thousand tokens. While what constitutes a "document" or "corpus" may differ depending on the 
    context, a single short document with only tens or hundreds of tokens is _not_ a corpus. 
    WordLevel's methods are only scientifically suited for corpora that have large, stable frequency 
    distributions across the vocabulary, and they are not recommended for comparing short documents individually.

### Representing a corpus

WordLevel represents a corpus as a set of tokens and their frequencies. The simplest representation 
is the [bag-of-words] model, which breaks documents down into tokens and their frequencies, ignoring 
their original order. This can be constructed using a dictionary mapping:

```python
corpus = {
    "the": 100,  # The token "the" appears 100 times across all the documents in the corpus
    "and": 88,
    "sad": 24,
    "happy": 35,
    # And so on...
}
```

Similarly, we can also represent a corpus as a set of rows and columns, where there is one row
per token, one column indicating its name, and one column indicating its frequency.

| token | frequency |
|-------|-----------|
| the   | 100       |
| and   | 88        |
| sad   | 24        |
| happy | 35        |

In Python, we can represent these rows and columns as a dataframe. For example, using the package
[Polars], our corpus is encoded as a dataframe.

```python
import polars as pl

corpus = pl.DataFrame({
    "token": ["the", "and", "sad", "happy"],
    "frequency": [100, 88, 24, 35]
})
```

!!! tip
    Currently, WordLevel does not have any built-in functionality for tokenizing documents and
    counting token frequencies. For tools that handle tokenization, consider using other packages 
    like [NLTK], [spaCy], or [scikit-learn].

## Catalog

### What is a catalog?

A **catalog** is WordLevel's interface for working with multiple corpora. It is the workbench for
transforming, manipulating, and comparing corpora in the same way that a dataframe is a workbench
for manipulating columns. 

### Creating a catalog

As described [above](#representing-a-corpus), one way to represent a corpus is as a [bag-of-words]
dictionary mapping from tokens to their frequencies. Say that we have two corpora, one about animals 
in general, and one about cats specifically. Their vocabularies will have some overlaps and some
differences.

```python
--8<-- "snippets/user_guide/catalog.py:setup"
```

If these are our corpora, we can create a `Catalog` in several ways, depending on which is the most
convenient. 

#### From multiple dictionaries

We can create a `Catalog` by passing both corpora (represented as token-to-frequency dictionaries) 
and giving them names. Note, the order of the dictionaries and corpus names needs to be the same. 

```python
--8<-- "snippets/user_guide/catalog.py:from_multiple_dicts"
```

#### From a single dictionary

We can also pass a single dictionary which names each token-to-frequency corpus explicitly:

```python
--8<-- "snippets/user_guide/catalog.py:from_single_dict"
```

#### From a dataframe
If we already have our corpora formatted as a dataframe, we can create a `Catalog` directly
from it. First, let's create a dataframe from our example data.

```python
--8<-- "snippets/user_guide/catalog.py:build_dataframe"
```

To create a `Catalog`, we need to provide the column indicating the names of the tokens, the 
`token_col`, and the names of the columns indicating frequencies of tokens, the `corpora`. Note,
with a list of dictionaries, we use the `corpora` parameter to name the corpora—here, we use it to 
select them.

```python
--8<-- "snippets/user_guide/catalog.py:from_dataframe"
```

!!! note
    WordLevel can create a `Catalog` from any [Narwhals]-supported dataframe, including Pandas,
    DuckDB, PySpark, and others.

    Polars is the internal engine for WordLevel. This means that even if you _provide_ a 
    non-Polars dataframe, any dataframe you _receive_ back from WordLevel will be a Polars 
    dataframe. See the [Polars] documentation to learn more about how to manipulate such dataframes.

    If you pass a lazy non-Polars dataframe, it will be eagerly collected and materialized in memory
    when the `Catalog` is constructed.


See the documentation of `Catalog` [constructors](../api/catalog/construct/index.md) for more 
details.


### Working with catalogs

All of the above constructors create the same `Catalog` object. Regardless of which one we used, the 
corpora are now represented as a [Polars] dataframe under the hood. We can inspect that dataframe 
directly through the `df` attribute.

```python
--8<-- "snippets/user_guide/catalog.py:inspect"
```

The output shows that there is one row per token, and there is one column for each corpus: one for
the animals corpus and one for the cats corpus. Observe two things:

- There are two new columns with the suffix `__wl_normed`. These columns are automatically created 
by the `Catalog` and indicate the _normalized_ frequency of each token in each corpus: the 
proportion between 0 and 1 derived by normalizing a token's frequency by the total sum of 
frequencies within a corpus.
- If a token does not appear in a corpus, it has a frequency of zero in that corpus. For example, 
"duck" is not in the cats corpus, but we see that in its row, it has a frequency of zero for that
corpus, even if we did not explicitly provide that information in the dictionaries. In other words,
the `Catalog` considers the union of vocabularies across all of the corpora, filling in zero
frequencies as needed.


  [Polars]: https://pola.rs/
  [NLTK]: https://www.nltk.org/
  [spaCy]: https://spacy.io/ 
  [scikit-learn]: https://scikit-learn.org/
  [Narwhals]: https://narwhals-dev.github.io/narwhals/
  [bag-of-words]: https://en.wikipedia.org/wiki/Bag-of-words_model