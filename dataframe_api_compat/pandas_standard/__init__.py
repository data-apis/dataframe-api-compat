from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import pandas as pd

from dataframe_api_compat.pandas_standard.pandas_standard import LATEST_API_VERSION
from dataframe_api_compat.pandas_standard.pandas_standard import PandasColumn
from dataframe_api_compat.pandas_standard.pandas_standard import PandasDataFrame
from dataframe_api_compat.pandas_standard.pandas_standard import PandasGroupBy

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    from dataframe_api.groupby_object import Aggregation as AggregationT
    from dataframe_api.typing import DType
    from dataframe_api.typing import Namespace
else:
    Namespace = object
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


Column = PandasColumn
DataFrame = PandasDataFrame
GroupBy = PandasGroupBy


def map_pandas_dtype_to_standard_dtype(dtype: Any) -> DType:
    if dtype == "int64":
        return PandasNamespace.Int64()
    if dtype == "Int64":
        return PandasNamespace.Int64()
    if dtype == "int32":
        return PandasNamespace.Int32()
    if dtype == "Int32":
        return PandasNamespace.Int32()
    if dtype == "int16":
        return PandasNamespace.Int16()
    if dtype == "Int16":
        return PandasNamespace.Int16()
    if dtype == "int8":
        return PandasNamespace.Int8()
    if dtype == "Int8":
        return PandasNamespace.Int8()
    if dtype == "uint64":
        return PandasNamespace.UInt64()
    if dtype == "UInt64":
        return PandasNamespace.UInt64()
    if dtype == "uint32":
        return PandasNamespace.UInt32()
    if dtype == "UInt32":
        return PandasNamespace.UInt32()
    if dtype == "uint16":
        return PandasNamespace.UInt16()
    if dtype == "UInt16":
        return PandasNamespace.UInt16()
    if dtype == "uint8":
        return PandasNamespace.UInt8()
    if dtype == "UInt8":
        return PandasNamespace.UInt8()
    if dtype == "float64":
        return PandasNamespace.Float64()
    if dtype == "Float64":
        return PandasNamespace.Float64()
    if dtype == "float32":
        return PandasNamespace.Float32()
    if dtype == "Float32":
        return PandasNamespace.Float32()
    if dtype == "bool":
        return PandasNamespace.Bool()
    if dtype == "object":
        return PandasNamespace.String()
    if dtype == "string":
        return PandasNamespace.String()
    if dtype == "datetime64[s]":
        return PandasNamespace.Date()
    if dtype.startswith("datetime64["):
        match = re.search(r"datetime64\[(\w{1,2})", dtype)
        assert match is not None
        time_unit = cast(Literal["ms", "us"], match.group(1))
        return PandasNamespace.Datetime(time_unit)
    if dtype.startswith("timedelta64["):
        match = re.search(r"timedelta64\[(\w{1,2})", dtype)
        assert match is not None
        time_unit = cast(Literal["ms", "us"], match.group(1))
        return PandasNamespace.Duration(time_unit)
    msg = f"Unsupported dtype! {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def map_standard_dtype_to_pandas_dtype(dtype: DType) -> Any:
    if isinstance(dtype, PandasNamespace.Int64):
        return "int64"
    if isinstance(dtype, PandasNamespace.Int32):
        return "int32"
    if isinstance(dtype, PandasNamespace.Int16):
        return "int16"
    if isinstance(dtype, PandasNamespace.Int8):
        return "int8"
    if isinstance(dtype, PandasNamespace.UInt64):
        return "uint64"
    if isinstance(dtype, PandasNamespace.UInt32):
        return "uint32"
    if isinstance(dtype, PandasNamespace.UInt16):
        return "uint16"
    if isinstance(dtype, PandasNamespace.UInt8):
        return "uint8"
    if isinstance(dtype, PandasNamespace.Float64):
        return "float64"
    if isinstance(dtype, PandasNamespace.Float32):
        return "float32"
    if isinstance(dtype, PandasNamespace.Bool):
        return "bool"
    if isinstance(dtype, PandasNamespace.String):
        return "object"
    if isinstance(dtype, PandasNamespace.Datetime):
        if dtype.time_zone is not None:  # pragma: no cover (todo)
            return f"datetime64[{dtype.time_unit}, {dtype.time_zone}]"
        return f"datetime64[{dtype.time_unit}]"
    if isinstance(dtype, PandasNamespace.Duration):
        return f"timedelta64[{dtype.time_unit}]"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def convert_to_standard_compliant_column(
    ser: pd.Series[Any],
    api_version: str | None = None,
) -> PandasColumn:
    if api_version is None:  # pragma: no cover
        api_version = LATEST_API_VERSION
    if ser.name is not None and not isinstance(ser.name, str):
        msg = f"Expected column with string name, got: {ser.name}"
        raise ValueError(msg)
    if ser.name is None:
        ser = ser.rename("")
    df = cast(PandasDataFrame, ser.to_frame().__dataframe_consortium_standard__())
    df = df.collect()
    name = cast(str, ser.name)
    return PandasColumn(df.col(name).column, api_version=api_version, df=df)


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame,
    api_version: str | None = None,
) -> PandasDataFrame:
    if api_version is None:
        api_version = LATEST_API_VERSION
    return PandasDataFrame(df, api_version=api_version)


class PandasNamespace(Namespace):
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
            # TODO validate time zone
            self.time_zone = time_zone

    class Duration(DurationT):
        def __init__(self, time_unit: Literal["ms", "us"]) -> None:
            self.time_unit = time_unit

    class Null:
        ...

    null = Null

    def dataframe_from_columns(
        self,
        *columns: PandasColumn,  # type: ignore[override]
    ) -> PandasDataFrame:
        data = {}
        api_versions: set[str] = set()
        for col in columns:
            col.df.validate_is_collected("dataframe_from_columns")
            data[col.name] = col.column
            api_versions.add(col.api_version)
        return PandasDataFrame(pd.DataFrame(data), api_version=list(api_versions)[0])

    def column_from_1d_array(
        self,
        data: Any,
        *,
        dtype: Any,
        name: str | None = None,
    ) -> PandasColumn:
        ser = pd.Series(data, dtype=map_standard_dtype_to_pandas_dtype(dtype), name=name)
        df = ser.to_frame().__dataframe_consortium_standard__().collect()
        return PandasColumn(df.col(name).column, api_version=self.api_version, df=df)

    def column_from_sequence(
        self,
        sequence: Sequence[Any],
        *,
        dtype: Any,
        name: str = "",
    ) -> PandasColumn:
        ser = pd.Series(
            sequence,
            dtype=map_standard_dtype_to_pandas_dtype(dtype),
            name=name,
        )
        df = ser.to_frame().__dataframe_consortium_standard__().collect()
        return PandasColumn(df.col(name).column, api_version=LATEST_API_VERSION, df=df)

    def concat(
        self,
        dataframes: Sequence[PandasDataFrame],  # type: ignore[override]
    ) -> PandasDataFrame:
        dtypes = dataframes[0].dataframe.dtypes
        dfs: list[pd.DataFrame] = []
        api_versions: set[str] = set()
        for df in dataframes:
            try:
                pd.testing.assert_series_equal(
                    df.dataframe.dtypes,
                    dtypes,
                )
            except AssertionError as exc:
                msg = "Expected matching columns"
                raise ValueError(msg) from exc
            else:
                dfs.append(df.dataframe)
            api_versions.add(df.api_version)
        if len(api_versions) > 1:  # pragma: no cover
            msg = f"Multiple api versions found: {api_versions}"
            raise ValueError(msg)
        return PandasDataFrame(
            pd.concat(
                dfs,
                axis=0,
                ignore_index=True,
            ),
            api_version=api_versions.pop(),
        )

    # typing needs fixing upstream
    def dataframe_from_2d_array(  # type: ignore[override]
        self,
        data: Any,
        *,
        schema: dict[str, Any],
    ) -> PandasDataFrame:  # pragma: no cover
        df = pd.DataFrame(data, columns=list(schema)).astype(
            {
                key: map_standard_dtype_to_pandas_dtype(value)
                for key, value in schema.items()
            },
        )
        return PandasDataFrame(df, api_version=self.api_version)

    def is_null(self, value: Any) -> bool:
        return value is self.null

    def is_dtype(self, dtype: Any, kind: str | tuple[str, ...]) -> bool:
        if isinstance(kind, str):
            kind = (kind,)
        dtypes: set[Any] = set()
        for _kind in kind:
            if _kind == "bool":
                dtypes.add(PandasNamespace.Bool)
            if _kind == "signed integer" or _kind == "integral" or _kind == "numeric":
                dtypes |= {
                    PandasNamespace.Int64,
                    PandasNamespace.Int32,
                    PandasNamespace.Int16,
                    PandasNamespace.Int8,
                }
            if _kind == "unsigned integer" or _kind == "integral" or _kind == "numeric":
                dtypes |= {
                    PandasNamespace.UInt64,
                    PandasNamespace.UInt32,
                    PandasNamespace.UInt16,
                    PandasNamespace.UInt8,
                }
            if _kind == "floating" or _kind == "numeric":
                dtypes |= {PandasNamespace.Float64, PandasNamespace.Float32}
            if _kind == "string":
                dtypes.add(PandasNamespace.String)
        return isinstance(dtype, tuple(dtypes))

    def date(self, year: int, month: int, day: int) -> Any:
        import datetime as dt  # temporary: make own class

        return pd.Timestamp(dt.date(year, month, day))

    class Aggregation(AggregationT):  # pragma: no cover
        def __init__(self, column_name: str, output_name: str, aggregation: str) -> None:
            self.column_name = column_name
            self.output_name = output_name
            self.aggregation = aggregation

        def rename(self, name: str) -> AggregationT:
            return self.__class__(self.column_name, name, self.aggregation)

        @classmethod
        def any(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "any")

        @classmethod
        def all(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "all")

        @classmethod
        def min(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "min")

        @classmethod
        def max(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "max")

        @classmethod
        def sum(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "sum")

        @classmethod
        def prod(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "prod")

        @classmethod
        def median(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "median")

        @classmethod
        def mean(
            cls: AggregationT,
            column: str,
            *,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "mean")

        @classmethod
        def std(
            cls: AggregationT,
            column: str,
            *,
            correction: int | float = 1,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "std")

        @classmethod
        def var(
            cls: AggregationT,
            column: str,
            *,
            correction: int | float = 1,
            skip_nulls: bool = True,
        ) -> AggregationT:
            return PandasNamespace.Aggregation(column, column, "var")

        @classmethod
        def size(
            cls: AggregationT,
        ) -> AggregationT:
            return PandasNamespace.Aggregation("placeholder", "size", "size")
