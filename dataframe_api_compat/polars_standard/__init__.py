from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import polars as pl

if TYPE_CHECKING:
    from dataframe_api.dtypes import Bool as BoolT
    from dataframe_api.dtypes import Date as DateT
    from dataframe_api.dtypes import Datetime as DatetimeT
    from dataframe_api.dtypes import Duration as DurationT
    from dataframe_api.dtypes import Float32 as Float32T
    from dataframe_api.dtypes import Float64 as Float64T
    from dataframe_api.dtypes import Int8 as Int8T
    from dataframe_api.dtypes import Int16 as Int16T
    from dataframe_api.dtypes import Int32 as Int32T
    from dataframe_api.dtypes import Int64 as Int64T
    from dataframe_api.dtypes import String as StringT
    from dataframe_api.dtypes import UInt8 as UInt8T
    from dataframe_api.dtypes import UInt16 as UInt16T
    from dataframe_api.dtypes import UInt32 as UInt32T
    from dataframe_api.dtypes import UInt64 as UInt64T
else:
    BoolT = object
    DateT = object
    DatetimeT = object
    DurationT = object
    Float32T = object
    Float64T = object
    Int16T = object
    Int32T = object
    Int64T = object
    Int8T = object
    StringT = object
    UInt16T = object
    UInt32T = object
    UInt64T = object
    UInt8T = object

from dataframe_api_compat.polars_standard.polars_standard import LATEST_API_VERSION
from dataframe_api_compat.polars_standard.polars_standard import PolarsColumn
from dataframe_api_compat.polars_standard.polars_standard import PolarsDataFrame
from dataframe_api_compat.polars_standard.polars_standard import PolarsGroupBy
from dataframe_api_compat.polars_standard.polars_standard import null

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api.typing import DType

Column = PolarsColumn
DataFrame = PolarsDataFrame
GroupBy = PolarsGroupBy


class Int64(Int64T):
    ...


class Int32(Int32T):
    ...


class Int16(Int16T):
    ...


class Int8(Int8T):
    ...


class UInt64(UInt64T):
    ...


class UInt32(UInt32T):
    ...


class UInt16(UInt16T):
    ...


class UInt8(UInt8T):
    ...


class Float64(Float64T):
    ...


class Float32(Float32T):
    ...


class Bool(BoolT):
    ...


class String(StringT):
    ...


class Date(DateT):
    ...


class Datetime(DatetimeT):
    def __init__(
        self,
        time_unit: Literal["ms", "us"],
        time_zone: str | None = None,
    ) -> None:
        self.time_unit = time_unit
        self.time_zone = time_zone


class Duration(DurationT):
    def __init__(self, time_unit: Literal["ms", "us"]) -> None:
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
        time_unit = cast(Literal["ms", "us"], dtype.time_unit)
        return Datetime(time_unit, dtype.time_zone)
    if isinstance(dtype, pl.Duration):
        time_unit = cast(Literal["ms", "us"], dtype.time_unit)
        return Duration(time_unit)
    msg = f"Got invalid dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


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
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def concat(dataframes: Sequence[PolarsDataFrame]) -> PolarsDataFrame:
    dfs: list[pl.DataFrame | pl.LazyFrame] = []
    api_versions: set[str] = set()
    for df in dataframes:
        dfs.append(df.dataframe)
        api_versions.add(df.api_version)
    if len(api_versions) > 1:  # pragma: no cover
        msg = f"Multiple api versions found: {api_versions}"
        raise ValueError(msg)
    return PolarsDataFrame(
        pl.concat(dfs),  # type: ignore[type-var]
        api_version=api_versions.pop(),
    )


def dataframe_from_columns(*columns: PolarsColumn) -> PolarsDataFrame:
    data = {}
    api_version: set[str] = set()
    for col in columns:
        col.df.validate_is_collected("dataframe_from_columns")
        df = cast(pl.DataFrame, col.df.dataframe)
        data[col.name] = df.select(col.expr).get_column(col.name)
        api_version.add(col.api_version)
    if len(api_version) > 1:  # pragma: no cover
        msg = f"found multiple api versions: {api_version}"
        raise ValueError(msg)
    return PolarsDataFrame(pl.DataFrame(data).lazy(), api_version=list(api_version)[0])


def column_from_1d_array(
    data: Any,
    *,
    dtype: Any,
    name: str,
    api_version: str | None = None,
) -> PolarsColumn:  # pragma: no cover
    ser = pl.Series(values=data, dtype=_map_standard_to_polars_dtypes(dtype), name=name)
    # TODO propagate api version
    df = cast(
        PolarsDataFrame,
        (
            ser.to_frame()
            .__dataframe_consortium_standard__(api_version=api_version)
            .collect()
        ),
    )
    return df.col(name)


def column_from_sequence(
    sequence: Sequence[Any],
    *,
    dtype: Any,
    name: str,
) -> PolarsColumn:
    ser = pl.Series(
        values=sequence,
        dtype=_map_standard_to_polars_dtypes(dtype),
        name=name,
    )
    # TODO propagate api version
    df = cast(
        PolarsDataFrame,
        (
            ser.to_frame()
            .__dataframe_consortium_standard__(api_version=LATEST_API_VERSION)
            .collect()
        ),
    )
    return df.col(name)


def dataframe_from_2d_array(
    data: Any,
    *,
    schema: dict[str, Any],
) -> PolarsDataFrame:  # pragma: no cover
    df = pl.DataFrame(
        data,
        schema={
            key: _map_standard_to_polars_dtypes(value) for key, value in schema.items()
        },
    ).lazy()
    return PolarsDataFrame(df, api_version=LATEST_API_VERSION)


def date(year: int, month: int, day: int) -> Any:
    return pl.date(year, month, day)


def convert_to_standard_compliant_column(
    ser: pl.Series,
    api_version: str | None = None,
) -> PolarsColumn:
    df = cast(
        PolarsDataFrame,
        (
            ser.to_frame()
            .__dataframe_consortium_standard__(api_version=api_version)
            .collect()
        ),
    )
    return df.col(ser.name)


def convert_to_standard_compliant_dataframe(
    df: pl.LazyFrame,
    api_version: str | None = None,
) -> PolarsDataFrame:
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    return PolarsDataFrame(df_lazy, api_version=api_version or LATEST_API_VERSION)


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
