from __future__ import annotations
from polars_standard.polars_standard import PolarsDataFrame, PolarsColumn

import polars as pl
from typing import Any, Sequence, TYPE_CHECKING, TypeVar, Generic

if TYPE_CHECKING:
    from dataframe_api import (
        DataFrame,
        DTypeT,
        IntDType,
        Bool,
        Column,
        Int64,
        Float64,
        DType,
        Scalar,
        GroupBy,
    )
else:
    DTypeT = TypeVar("DTypeT")

    class DataFrame(Generic[DTypeT]):
        ...

    class IntDType:
        ...

    class Bool:
        ...

    class Column(Generic[DTypeT]):
        ...

    class Int64:
        ...

    class Float64:
        ...

    class DType:
        ...

    class Scalar:
        ...

    class GroupBy:
        ...


def _map_standard_to_polars_dtypes(dtype: DType) -> pl.DataType:
    if isinstance(dtype, Int64):
        return pl.Int64()
    if isinstance(dtype, Float64):
        return pl.Float64()
    if isinstance(dtype, Bool):
        return pl.Boolean()
    raise AssertionError(f"Unknown dtype: {dtype}")


def concat(dataframes: Sequence[PolarsDataFrame]) -> PolarsDataFrame:
    dfs = []
    for _df in dataframes:
        dfs.append(_df.dataframe)
    return PolarsDataFrame(pl.concat(dfs))


def dataframe_from_dict(data: dict[str, PolarsColumn[Any]]) -> PolarsDataFrame:
    return PolarsDataFrame(
        pl.DataFrame({label: column.column for label, column in data.items()})
    )


def column_from_sequence(
    sequence: Sequence[DTypeT], dtype: DType
) -> PolarsColumn[DTypeT]:
    return PolarsColumn(pl.Series(sequence, dtype=_map_standard_to_polars_dtypes(dtype)))


def convert_to_standard_compliant_dataframe(df: pl.DataFrame) -> PolarsDataFrame:
    return PolarsDataFrame(df)
