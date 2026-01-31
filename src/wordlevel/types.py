from typing import TypeVar

import polars as pl


PolarsFrame = TypeVar("PolarsFrame", bound=pl.DataFrame | pl.LazyFrame)
