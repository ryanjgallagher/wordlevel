from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TypeAlias, Sequence, Mapping

import polars as pl

from wordlevel.compare.comparisons.comparison import ComparisonResult


@dataclass
class ComparatorData:
    df: pl.LazyFrame
    token_col: str
    freq_cols: Collection[str]
    normalization_suffix: str
    comparisons: dict[str, ComparisonResult]


ListOfFreqs: TypeAlias = Sequence[Mapping[str, float]]
"""
A sequence of word-to-frequencey mappings, one per corpus. Each mapping is `{word: frequency}` for a single corpus

Example:
    ```python
    list_of_freqs = [
        {"hi": 1, "hey": 2, "hello": 3},
        {"hi": 2, "hey"; 4, "bye": 6}
    ]
    ```
"""


DictOfFreqs: TypeAlias = Mapping[str, Mapping[str, float]]
"""
A mapping of corpus names to word-to-frequency mappings. The mapping is `{corpus: {word: frequency}}`

Example:
    ```python
    dict_of_freqs = {
        "corpus1": {"hi": 1, "hey": 2, "hello": 3},
        "corpus2": {"hi": 2, "hey"; 4, "bye": 6}
    }
    ```
"""
