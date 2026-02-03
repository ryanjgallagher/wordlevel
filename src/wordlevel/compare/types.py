from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import polars as pl

from wordlevel.compare.comparisons.comparison import ComparisonResult


@dataclass
class ComparatorData:
    df: pl.LazyFrame
    label_col: str
    freq_cols: Collection[str]
    normalization_suffix: str
    comparisons: dict[str, ComparisonResult]
