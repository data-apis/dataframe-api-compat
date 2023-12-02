from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import polars as pl

from dataframe_api_compat.polars_standard.column_object import Column
from dataframe_api_compat.polars_standard.dataframe_object import DataFrame
from dataframe_api_compat.polars_standard.scalar_object import Scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api.typing import Column as ColumnT
    from dataframe_api.typing import DataFrame as DataFrameT
    from dataframe_api.typing import Namespace as NamespaceT
    from dataframe_api.typing import Scalar as ScalarT

    BoolT = NamespaceT.Bool
    DateT = NamespaceT.Date
    DatetimeT = NamespaceT.Datetime
    DurationT = NamespaceT.Duration
    Float32T = NamespaceT.Float32
    Float64T = NamespaceT.Float64
    Int8T = NamespaceT.Int8
    Int16T = NamespaceT.Int16
    Int32T = NamespaceT.Int32
    Int64T = NamespaceT.Int64
    StringT = NamespaceT.String
    UInt8T = NamespaceT.UInt8
    UInt16T = NamespaceT.UInt16
    UInt32T = NamespaceT.UInt32
    UInt64T = NamespaceT.UInt64
    NullTypeT = NamespaceT.NullType

    from dataframe_api.groupby_object import Aggregation as AggregationT
    from dataframe_api.typing import DType
    from dataframe_api.typing import Namespace as NamespaceT
else:
    NamespaceT = object
    BoolT = object
    DateT = object
    DatetimeT = object
    DurationT = object
    Float32T = object
    Float64T = object
    Int8T = object
    Int16T = object
    Int32T = object
    Int64T = object
    StringT = object
    UInt8T = object
    UInt16T = object
    UInt32T = object
    UInt64T = object
    AggregationT = object
    NullTypeT = object
    NamespaceT = object


SUPPORTED_VERSIONS = frozenset({"2023.11-beta"})


class Namespace(NamespaceT):
    def __init__(self, *, api_version: str) -> None:
        self.__dataframe_api_version__ = api_version
        self.api_version = api_version

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

    class NullType(NullTypeT):
        ...

    null = NullType()

    def is_null(self, value: Any) -> bool:
        return value is self.null

    def dataframe_from_columns(
        self,
        *columns: Column,  # type: ignore[override]
    ) -> DataFrame:
        data = {}
        api_version: set[str] = set()
        for col in columns:
            ser = col._materialise()
            data[ser.name] = ser
            api_version.add(col._api_version)
        if len(api_version) > 1:  # pragma: no cover
            msg = f"found multiple api versions: {api_version}"
            raise ValueError(msg)
        return DataFrame(
            pl.DataFrame(data).lazy(),
            api_version=list(api_version)[0],
        )

    def column_from_1d_array(  # type: ignore[override]
        self,
        array: Any,
        *,
        name: str = "",
    ) -> Column:
        ser = pl.Series(values=array, name=name)
        return Column(ser, api_version=self.api_version, df=None, is_persisted=True)

    def column_from_sequence(
        self,
        sequence: Sequence[Any],
        *,
        dtype: DType,
        name: str = "",
    ) -> Column:
        ser = pl.Series(
            values=sequence,
            dtype=_map_standard_to_polars_dtypes(dtype),
            name=name,
        )
        return Column(ser, api_version=self.api_version, df=None, is_persisted=True)

    def dataframe_from_2d_array(
        self,
        data: Any,
        *,
        names: Sequence[str],
    ) -> DataFrame:
        df = pl.DataFrame(
            data,
            schema=names,
        ).lazy()
        return DataFrame(df, api_version=self.api_version)

    def date(self, year: int, month: int, day: int) -> Scalar:
        return Scalar(
            pl.date(year, month, day),
            api_version=self.api_version,
            df=None,
            is_persisted=True,
        )

    class Aggregation(AggregationT):
        def __init__(self, column_name: str, output_name: str, aggregation: str) -> None:
            self.column_name = column_name
            self.output_name = output_name
            self.aggregation = aggregation

        def rename(self, name: str | ScalarT) -> AggregationT:
            return self.__class__(self.column_name, name, self.aggregation)  # type: ignore[arg-type]

        @classmethod
        def any(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "any")

        @classmethod
        def all(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "all")

        @classmethod
        def min(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "min")

        @classmethod
        def max(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "max")

        @classmethod
        def sum(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "sum")

        @classmethod
        def prod(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "product")

        @classmethod
        def median(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "median")

        @classmethod
        def mean(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "mean")

        @classmethod
        def std(
            cls: AggregationT,
            column: str,
            *,
            correction: float | ScalarT | NullTypeT = 1,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "std")

        @classmethod
        def var(
            cls: AggregationT,
            column: str,
            *,
            correction: float | ScalarT | NullTypeT = 1,
            skip_nulls: bool | ScalarT = True,
        ) -> AggregationT:
            return Namespace.Aggregation(column, column, "var")

        @classmethod
        def size(
            cls: AggregationT,
        ) -> AggregationT:
            return Namespace.Aggregation("__placeholder__", "size", "count")

    def concat(
        self,
        dataframes: Sequence[DataFrameT],
    ) -> DataFrame:
        dataframes = cast("Sequence[DataFrame]", dataframes)
        dfs: list[pl.LazyFrame | pl.DataFrame] = []
        api_versions: set[str] = set()
        for df in dataframes:
            dfs.append(df.dataframe)
            api_versions.add(df._api_version)
        if len(api_versions) > 1:  # pragma: no cover
            msg = f"Multiple api versions found: {api_versions}"
            raise ValueError(msg)
        # todo raise if not all share persistedness
        return DataFrame(
            pl.concat(dfs),  # type: ignore[type-var]
            api_version=api_versions.pop(),
        )

    def is_dtype(self, dtype: Any, kind: str | tuple[str, ...]) -> bool:
        if isinstance(kind, str):
            kind = (kind,)
        dtypes: set[Any] = set()
        for _kind in kind:
            if _kind == "bool":
                dtypes.add(self.Bool)
            if _kind == "signed integer" or _kind == "integral" or _kind == "numeric":
                dtypes |= {self.Int64, self.Int32, self.Int16, self.Int8}
            if _kind == "unsigned integer" or _kind == "integral" or _kind == "numeric":
                dtypes |= {self.UInt64, self.UInt32, self.UInt16, self.UInt8}
            if _kind == "floating" or _kind == "numeric":
                dtypes |= {self.Float64, self.Float32}
            if _kind == "string":
                dtypes.add(self.String)
        return isinstance(dtype, tuple(dtypes))

    # Horizontal reductions

    def all_horizontal(self, *columns: ColumnT, skip_nulls: bool = True) -> ColumnT:
        return reduce(lambda x, y: x & y, columns)

    def any_horizontal(self, *columns: ColumnT, skip_nulls: bool = True) -> ColumnT:
        return reduce(lambda x, y: x | y, columns)

    def sorted_indices(
        self,
        *columns: ColumnT,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        raise NotImplementedError

    def unique_indices(self, *columns: ColumnT, skip_nulls: bool = True) -> Column:
        raise NotImplementedError


def map_polars_dtype_to_standard_dtype(dtype: Any) -> DType:
    if dtype == pl.Int64:
        return Namespace.Int64()
    if dtype == pl.Int32:
        return Namespace.Int32()
    if dtype == pl.Int16:
        return Namespace.Int16()
    if dtype == pl.Int8:
        return Namespace.Int8()
    if dtype == pl.UInt64:
        return Namespace.UInt64()
    if dtype == pl.UInt32:
        return Namespace.UInt32()
    if dtype == pl.UInt16:
        return Namespace.UInt16()
    if dtype == pl.UInt8:
        return Namespace.UInt8()
    if dtype == pl.Float64:
        return Namespace.Float64()
    if dtype == pl.Float32:
        return Namespace.Float32()
    if dtype == pl.Boolean:
        return Namespace.Bool()
    if dtype == pl.Utf8:
        return Namespace.String()
    if dtype == pl.Date:  # pragma: no cover  # not supported yet?
        return Namespace.Date()
    if isinstance(dtype, pl.Datetime):
        time_unit = cast(Literal["ms", "us"], dtype.time_unit)
        return Namespace.Datetime(time_unit, dtype.time_zone)
    if isinstance(dtype, pl.Duration):
        time_unit = cast(Literal["ms", "us"], dtype.time_unit)
        return Namespace.Duration(time_unit)
    msg = f"Got invalid dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def _map_standard_to_polars_dtypes(dtype: Any) -> pl.DataType:
    if isinstance(dtype, Namespace.Int64):
        return pl.Int64()
    if isinstance(dtype, Namespace.Int32):
        return pl.Int32()
    if isinstance(dtype, Namespace.Int16):
        return pl.Int16()
    if isinstance(dtype, Namespace.Int8):
        return pl.Int8()
    if isinstance(dtype, Namespace.UInt64):
        return pl.UInt64()
    if isinstance(dtype, Namespace.UInt32):
        return pl.UInt32()
    if isinstance(dtype, Namespace.UInt16):
        return pl.UInt16()
    if isinstance(dtype, Namespace.UInt8):
        return pl.UInt8()
    if isinstance(dtype, Namespace.Float64):
        return pl.Float64()
    if isinstance(dtype, Namespace.Float32):
        return pl.Float32()
    if isinstance(dtype, Namespace.Bool):
        return pl.Boolean()
    if isinstance(dtype, Namespace.String):
        return pl.Utf8()
    if isinstance(dtype, Namespace.Datetime):
        return pl.Datetime(dtype.time_unit, dtype.time_zone)
    if isinstance(dtype, Namespace.Duration):
        return pl.Duration(dtype.time_unit)
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def convert_to_standard_compliant_column(
    ser: pl.Series,
    api_version: str | None = None,
) -> Column:
    return Column(
        ser,
        api_version=api_version or "2023.11-beta",
        df=None,
        is_persisted=True,
    )


def convert_to_standard_compliant_dataframe(
    df: pl.LazyFrame | pl.DataFrame,
    api_version: str | None = None,
) -> DataFrame:
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    return DataFrame(df_lazy, api_version=api_version or "2023.11-beta")
