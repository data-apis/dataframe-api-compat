from __future__ import annotations

from typing import Any
from typing import Literal
from typing import TYPE_CHECKING
from typing import TypeVar

import polars as pl

from dataframe_api_compat.polars_standard.polars_standard import LATEST_API_VERSION
from dataframe_api_compat.polars_standard.polars_standard import null
from dataframe_api_compat.polars_standard.polars_standard import PolarsColumn
from dataframe_api_compat.polars_standard.polars_standard import PolarsDataFrame
from dataframe_api_compat.polars_standard.polars_standard import PolarsGroupBy
from dataframe_api_compat.polars_standard.polars_standard import PolarsPermissiveColumn
from dataframe_api_compat.polars_standard.polars_standard import PolarsPermissiveFrame

if TYPE_CHECKING:
    from collections.abc import Sequence
    from dataframe_api._types import DType

col = PolarsColumn
Column = col
PermissiveColumn = PolarsPermissiveColumn
DataFrame = PolarsDataFrame
PermissiveFrame = PolarsPermissiveFrame
GroupBy = PolarsGroupBy

PolarsType = TypeVar("PolarsType", pl.DataFrame, pl.LazyFrame)


class Int64:
    ...


class Int32:
    ...


class Int16:
    ...


class Int8:
    ...


class UInt64:
    ...


class UInt32:
    ...


class UInt16:
    ...


class UInt8:
    ...


class Float64:
    ...


class Float32:
    ...


class Bool:
    ...


class String:
    ...


class Date:
    ...


class Datetime:
    def __init__(self, time_unit, time_zone=None):
        self.time_unit = time_unit
        self.time_zone = time_zone


class Duration:
    def __init__(self, time_unit):
        self.time_unit = time_unit


def map_polars_dtype_to_standard_dtype(dtype: Any) -> DType:
    if dtype == pl.Int64:
        return Int64()
    if dtype == pl.Int32:
        return Int32()
    if dtype == pl.Int16:
        return Int16()
    if dtype == pl.Int8:
        return Int8()
    if dtype == pl.UInt64:
        return UInt64()
    if dtype == pl.UInt32:
        return UInt32()
    if dtype == pl.UInt16:
        return UInt16()
    if dtype == pl.UInt8:
        return UInt8()
    if dtype == pl.Float64:
        return Float64()
    if dtype == pl.Float32:
        return Float32()
    if dtype == pl.Boolean:
        return Bool()
    if dtype == pl.Utf8:
        return String()
    if dtype == pl.Date:
        return Date()
    if isinstance(dtype, pl.Datetime):
        return Datetime(dtype.time_unit, dtype.time_zone)
    if isinstance(dtype, pl.Duration):
        return Duration(dtype.time_unit)
    raise AssertionError(f"Got invalid dtype: {dtype}")


def is_null(value: Any) -> bool:
    return value is null


def _map_standard_to_polars_dtypes(dtype: Any) -> pl.DataType:
    if isinstance(dtype, Int64):
        return pl.Int64()
    if isinstance(dtype, Int32):
        return pl.Int32()
    if isinstance(dtype, Int16):
        return pl.Int16()
    if isinstance(dtype, Int8):
        return pl.Int8()
    if isinstance(dtype, UInt64):
        return pl.UInt64()
    if isinstance(dtype, UInt32):
        return pl.UInt32()
    if isinstance(dtype, UInt16):
        return pl.UInt16()
    if isinstance(dtype, UInt8):
        return pl.UInt8()
    if isinstance(dtype, Float64):
        return pl.Float64()
    if isinstance(dtype, Float32):
        return pl.Float32()
    if isinstance(dtype, Bool):
        return pl.Boolean()
    if isinstance(dtype, String):
        return pl.Utf8()
    if isinstance(dtype, Datetime):
        return pl.Datetime(dtype.time_unit, dtype.time_zone)
    if isinstance(dtype, Duration):  # pragma: no cover
        # pending fix in polars itself
        return pl.Duration(dtype.time_unit)
    raise AssertionError(f"Unknown dtype: {dtype}")


def concat(dataframes: Sequence[PolarsDataFrame]) -> PolarsDataFrame:
    dfs = []
    api_versions = set()
    for _df in dataframes:
        dfs.append(_df.dataframe)
        api_versions.add(_df._api_version)
    if len(api_versions) > 1:  # pragma: no cover
        raise ValueError(f"Multiple api versions found: {api_versions}")
    return PolarsDataFrame(pl.concat(dfs), api_version=api_versions.pop())


def dataframe_from_dict(
    data: dict[str, PolarsPermissiveColumn[Any]], *, api_version: str | None = None
) -> PolarsDataFrame:
    for _, col in data.items():
        if not isinstance(col, PolarsPermissiveColumn):  # pragma: no cover
            raise TypeError(f"Expected PolarsPermissiveColumn, got {type(col)}")
        if isinstance(col.column, pl.Expr):
            raise NotImplementedError(
                "dataframe_from_dict not supported for lazy columns"
            )
    return PolarsDataFrame(
        pl.DataFrame(
            {label: column.column.rename(label) for label, column in data.items()}  # type: ignore[union-attr]
        ).lazy(),
        api_version=api_version or LATEST_API_VERSION,
    )


def column_from_1d_array(
    data: Any, *, dtype: Any, name: str, api_version: str | None = None
) -> PolarsPermissiveColumn[Any]:  # pragma: no cover
    ser = pl.Series(values=data, dtype=_map_standard_to_polars_dtypes(dtype), name=name)
    return PolarsPermissiveColumn(ser, api_version=api_version or LATEST_API_VERSION)


def column_from_sequence(
    sequence: Sequence[Any],
    *,
    dtype: Any,
    name: str | None = None,
    api_version: str | None = None,
) -> PolarsPermissiveColumn[Any]:
    return PolarsPermissiveColumn(
        pl.Series(
            values=sequence, dtype=_map_standard_to_polars_dtypes(dtype), name=name
        ),
        api_version=api_version or LATEST_API_VERSION,
    )


def dataframe_from_2d_array(
    data: Any,
    *,
    names: Sequence[str],
    dtypes: dict[str, Any],
    api_version: str | None = None,
) -> PolarsDataFrame:  # pragma: no cover
    df = pl.DataFrame(
        data,
        schema={
            key: _map_standard_to_polars_dtypes(value) for key, value in dtypes.items()
        },
    ).lazy()
    return PolarsDataFrame(df, api_version=api_version or LATEST_API_VERSION)


def convert_to_standard_compliant_dataframe(
    df: pl.LazyFrame, api_version: str | None = None
) -> PolarsDataFrame:
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    return PolarsDataFrame(df_lazy, api_version=api_version or LATEST_API_VERSION)


def convert_to_standard_compliant_column(
    ser: pl.Series, api_version: str | None = None
) -> PolarsPermissiveColumn[Any]:  # pragma: no cover  (todo: is this even needed?)
    return PolarsPermissiveColumn(ser, api_version=api_version or LATEST_API_VERSION)


def is_dtype(dtype: Any, kind: str | tuple[str, ...]) -> bool:
    if isinstance(kind, str):
        kind = (kind,)
    dtypes: set[Any] = set()
    for _kind in kind:
        if _kind == "bool":
            dtypes.add(Bool)
        if _kind == "signed integer" or _kind == "integral" or _kind == "numeric":
            dtypes |= {Int64, Int32, Int16, Int8}
        if _kind == "unsigned integer" or _kind == "integral" or _kind == "numeric":
            dtypes |= {UInt64, UInt32, UInt16, UInt8}
        if _kind == "floating" or _kind == "numeric":
            dtypes |= {Float64, Float32}
        if _kind == "string":
            dtypes.add(String)
    return isinstance(dtype, tuple(dtypes))


def any_rowwise(*columns: str, skip_nulls: bool = True):
    return PolarsColumn(pl.any_horizontal(list(columns) or "*").alias("any"))


def all_rowwise(*columns: str, skip_nulls: bool = True):
    return PolarsColumn(pl.all_horizontal(list(columns) or "*").alias("all"))


def sorted_indices(
    keys: str | list[str] | None = None,
    *,
    ascending: Sequence[bool] | bool = True,
    nulls_position: Literal["first", "last"] = "last",
) -> Column:
    return PolarsColumn(pl.arg_sort_by(keys or "*", descending=not ascending))


def unique_indices(
    keys: str | list[str] | None = None, *, skip_nulls: bool = True
) -> Column:
    raise NotImplementedError("namespace.unique_indices not implemented for polars yet")
