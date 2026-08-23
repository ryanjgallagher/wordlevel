# Exclusions

## Why exclude tokens

Excluding tokens is a way of narrowing a comparison to the vocabulary that is of interest, such as
removing stop words that carry little meaning of their own. Tokens are excluded from both corpora,
before they are scored, so the frequencies the scoring measure works with are normalized over the
remaining vocabulary.

## Excluding changes what remains

!!! warning
    Excluding tokens changes the normalized frequencies of every token that remains.

Because the remaining frequencies are renormalized, scores from a filtered comparison are not
directly comparable with scores from an unfiltered one, even for the tokens both have in common.

## Excluding commutes with scoring

Scoring a comparison after excluding gives the same result as excluding after scoring. Excluding more
than once removes the union of the tokens, and a token that is not in the vocabulary is ignored.

## Unshared tokens

A token that only one corpus uses can be a stark difference between them. Prefer using a measure that
can handle these absences, rather than excluding them entirely.

## Counts and shares

A count means different things in corpora of different sizes, so excluding by a share of each corpus
is often the better choice when the corpora are not the same size.

Proportions are taken over each corpus as the catalog gave it, before anything the comparison
excludes, so the bounds mean the same thing no matter what else the comparison excludes or how it is
scored.

!!! warning
    A token missing from one of the corpora has a frequency and a proportion of zero, so
    `corpus="either"` paired with `less_than` removes every token that the two corpora do not share.

For how to apply each of these, see [Excluding Tokens](../how_to/exclude_tokens.md).
