from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar

import polars as pl

from dataframe_api_compat.polars_standard.polars_standard import LATEST_API_VERSION
from dataframe_api_compat.polars_standard.polars_standard import null
from dataframe_api_compat.polars_standard.polars_standard import PolarsColumn
from dataframe_api_compat.polars_standard.polars_standard import PolarsDataFrame
from dataframe_api_compat.polars_standard.polars_standard import PolarsGroupBy

if TYPE_CHECKING:
    from collections.abc import Sequence

Column = PolarsColumn
DataFrame = PolarsDataFrame
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


DTYPE_MAP = {
    pl.Int64(): Int64(),
    pl.Int32(): Int32(),
    pl.Int16(): Int16(),
    pl.Int8(): Int8(),
    pl.UInt64(): UInt64(),
    pl.UInt32(): UInt32(),
    pl.UInt16(): UInt16(),
    pl.UInt8(): UInt8(),
    pl.Float64(): Float64(),
    pl.Float32(): Float32(),
    pl.Boolean(): Bool(),
    pl.Utf8(): String(),
}


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
    data: dict[str, PolarsColumn[Any]], *, api_version: str | None = None
) -> PolarsDataFrame:
    for _, col in data.items():
        if not isinstance(col, PolarsColumn):  # pragma: no cover
            raise TypeError(f"Expected PolarsColumn, got {type(col)}")
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
) -> PolarsColumn[Any]:  # pragma: no cover
    ser = pl.Series(values=data, dtype=_map_standard_to_polars_dtypes(dtype), name=name)
    return PolarsColumn(
        ser, dtype=ser.dtype, id_=None, api_version=api_version or LATEST_API_VERSION
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


def column_from_sequence(
    sequence: Sequence[Any],
    *,
    dtype: Any,
    name: str | None = None,
    api_version: str | None = None,
) -> PolarsColumn[Any]:
    return PolarsColumn(
        pl.Series(
            values=sequence, dtype=_map_standard_to_polars_dtypes(dtype), name=name
        ),
        dtype=_map_standard_to_polars_dtypes(dtype),
        id_=None,
        api_version=api_version or LATEST_API_VERSION,
    )


def convert_to_standard_compliant_dataframe(
    df: pl.DataFrame | pl.LazyFrame, api_version: str | None = None
) -> PolarsDataFrame:
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    return PolarsDataFrame(df_lazy, api_version=api_version or LATEST_API_VERSION)


def convert_to_standard_compliant_column(
    ser: pl.Series, api_version: str | None = None
) -> PolarsColumn[Any]:  # pragma: no cover  (todo: is this even needed?)
    return PolarsColumn(
        ser, dtype=ser.dtype, id_=None, api_version=api_version or LATEST_API_VERSION
    )


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
