from __future__ import annotations

import narwhals as nw
import polars as pl
from narwhals.typing import IntoFrame
from typing_extensions import TypeIs


def is_dataframe(data: object) -> TypeIs[IntoFrame]:
    """
    Checks whether an object is a dataframe that can be converted to a Polars dataframe.

    Recognizes any dataframe from a backend supported by Narwhals, as well as any object that
    produces an Arrow C stream.

    Args:
        data: Any object

    Returns:
        Whether the object is a convertible dataframe
    """
    if isinstance(nw.from_native(data, pass_through=True), (nw.DataFrame, nw.LazyFrame)):
        return True

    # Producers of an Arrow C stream that Narwhals has no dedicated backend for
    return hasattr(data, "__arrow_c_stream__")


def to_polars(df: IntoFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Converts a dataframe from any backend supported by Narwhals to a Polars dataframe.

    Polars dataframes are used as given, so a LazyFrame stays lazy. Dataframes from other lazy
    backends, like DuckDB and PySpark, are materialized in memory.

    Args:
        df: A dataframe from any backend supported by Narwhals

    Returns:
        The dataframe as a Polars DataFrame or LazyFrame

    Raises:
        TypeError: If `df` is not a dataframe that can be converted to Polars.
    """
    df_nw = nw.from_native(df, pass_through=True)

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df
    elif isinstance(df_nw, (nw.DataFrame, nw.LazyFrame)):
        # Narwhals has no lazy-to-lazy bridge into Polars, so lazy backends are materialized
        return df_nw.lazy().collect(backend="polars").to_polars()
    elif hasattr(df, "__arrow_c_stream__"):
        # Producers of an Arrow C stream that Narwhals also supports
        return nw.from_arrow(df, backend=pl).to_polars()
    else:
        raise TypeError(f"Input type not supported: {type(df)}")
