from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import polars as pl

from dataframe_api_compat.polars_standard.polars_standard import null  # noqa: F401
from dataframe_api_compat.polars_standard.polars_standard import PolarsColumn
from dataframe_api_compat.polars_standard.polars_standard import PolarsDataFrame
from dataframe_api_compat.polars_standard.polars_standard import PolarsGroupBy

if TYPE_CHECKING:
    from collections.abc import Sequence

Column = PolarsColumn
DataFrame = PolarsDataFrame
GroupBy = PolarsGroupBy


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


DTYPE_MAP = {
    pl.Int64: Int64(),
    pl.Int32: Int32(),
    pl.Int16: Int16(),
    pl.Int8: Int8(),
    pl.UInt64: UInt64(),
    pl.UInt32: UInt32(),
    pl.UInt16: UInt16(),
    pl.UInt8: UInt8(),
    pl.Float64: Float64(),
    pl.Float32: Float32(),
    pl.Boolean: Bool(),
}


def is_null(value: Any) -> bool:
    return value is None


def _map_standard_to_polars_dtypes(dtype: Any) -> pl.DataType:
    if isinstance(dtype, Int64):
        return pl.Int64()
    if isinstance(dtype, Int32):
        return pl.Int32()
    if isinstance(dtype, Float64):
        return pl.Float64()
    if isinstance(dtype, Float32):
        return pl.Float32()
    if isinstance(dtype, Bool):
        return pl.Boolean()
    raise AssertionError(f"Unknown dtype: {dtype}")


def concat(dataframes: Sequence[PolarsDataFrame]) -> PolarsDataFrame:
    dfs = []
    for _df in dataframes:
        dfs.append(_df.dataframe)
    return PolarsDataFrame(pl.concat(dfs))


def dataframe_from_dict(data: dict[str, PolarsColumn[Any]]) -> PolarsDataFrame:
    for _, col in data.items():
        if not isinstance(col, PolarsColumn):  # pragma: no cover
            raise TypeError(f"Expected PolarsColumn, got {type(col)}")
    return PolarsDataFrame(
        pl.DataFrame(
            {label: column.column.rename(label) for label, column in data.items()}
        )
    )


def column_from_1d_array(
    data: Any, *, dtype: Any, name: str
) -> PolarsColumn[Any]:  # pragma: no cover
    ser = pl.Series(values=data, dtype=_map_standard_to_polars_dtypes(dtype), name=name)
    return PolarsColumn(ser)


def dataframe_from_2d_array(
    data: Any, *, names: Sequence[str], dtypes: dict[str, Any]
) -> PolarsDataFrame:  # pragma: no cover
    df = pl.DataFrame(
        data,
        schema={
            key: _map_standard_to_polars_dtypes(value) for key, value in dtypes.items()
        },
    )
    return PolarsDataFrame(df)


def column_from_sequence(
    sequence: Sequence[Any], *, dtype: Any, name: str | None = None
) -> PolarsColumn[Any]:
    return PolarsColumn(
        pl.Series(values=sequence, dtype=_map_standard_to_polars_dtypes(dtype), name=name)
    )


def convert_to_standard_compliant_dataframe(
    df: pl.DataFrame, api_version: str | None = None
) -> PolarsDataFrame:
    if api_version is None:
        api_version = "2023.08"
    if api_version != "2023.08":  # pragma: no cover
        raise ValueError(
            f"Unknown api_version: {api_version}. Expected: '2023.08', or None"
        )
    return PolarsDataFrame(df)


def convert_to_standard_compliant_column(
    ser: pl.Series, api_version: str | None = None
) -> PolarsColumn[Any]:
    if api_version is None:
        api_version = "2023.08"
    if api_version != "2023.08":  # pragma: no cover
        raise ValueError(
            f"Unknown api_version: {api_version}. Expected: '2023.08', or None"
        )
    return PolarsColumn(ser)
