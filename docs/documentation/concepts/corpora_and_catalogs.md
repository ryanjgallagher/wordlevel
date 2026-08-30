# Corpora & Catalogs

## What is a corpus?

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

## How WordLevel represents a corpus

WordLevel represents a corpus as a set of tokens and their frequencies. The simplest representation 
is the [bag-of-words] model, which breaks documents down into tokens and their frequencies, ignoring 
their original order.

For the concrete ways to get your data into that form, see
[Constructing Catalogs](../cookbooks/constructing_catalogs.md).

## What is a catalog?

A **catalog** is WordLevel's interface for working with multiple corpora. It is the workbench for
transforming, manipulating, and comparing corpora in the same way that a dataframe is a workbench
for manipulating columns. 


  [bag-of-words]: https://en.wikipedia.org/wiki/Bag-of-words_model
