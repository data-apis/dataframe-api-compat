from __future__ import annotations

import datetime as dt
import re
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import cast

import pandas as pd

from dataframe_api_compat.pandas_standard.column_object import Column
from dataframe_api_compat.pandas_standard.dataframe_object import DataFrame
from dataframe_api_compat.pandas_standard.scalar_object import Scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api.groupby_object import Aggregation as AggregationT
    from dataframe_api.typing import Column as ColumnT
    from dataframe_api.typing import DataFrame as DataFrameT
    from dataframe_api.typing import DType
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

SUPPORTED_VERSIONS = frozenset({"2023.11-beta"})


def map_pandas_dtype_to_standard_dtype(dtype: Any) -> DType:
    if dtype == "int64":
        return Namespace.Int64()
    if dtype == "Int64":
        return Namespace.Int64()
    if dtype == "int32":
        return Namespace.Int32()
    if dtype == "Int32":
        return Namespace.Int32()
    if dtype == "int16":
        return Namespace.Int16()
    if dtype == "Int16":
        return Namespace.Int16()
    if dtype == "int8":
        return Namespace.Int8()
    if dtype == "Int8":
        return Namespace.Int8()
    if dtype == "uint64":
        return Namespace.UInt64()
    if dtype == "UInt64":
        return Namespace.UInt64()
    if dtype == "uint32":
        return Namespace.UInt32()
    if dtype == "UInt32":
        return Namespace.UInt32()
    if dtype == "uint16":
        return Namespace.UInt16()
    if dtype == "UInt16":
        return Namespace.UInt16()
    if dtype == "uint8":
        return Namespace.UInt8()
    if dtype == "UInt8":
        return Namespace.UInt8()
    if dtype == "float64":
        return Namespace.Float64()
    if dtype == "Float64":
        return Namespace.Float64()
    if dtype == "float32":
        return Namespace.Float32()
    if dtype == "Float32":
        return Namespace.Float32()
    if dtype == "bool":
        return Namespace.Bool()
    if dtype == "object":
        return Namespace.String()
    if dtype == "string":
        return Namespace.String()
    if dtype.startswith("datetime64["):
        match = re.search(r"datetime64\[(\w{1,2})", dtype)
        assert match is not None
        time_unit = cast(Literal["ms", "us"], match.group(1))
        return Namespace.Datetime(time_unit)
    if dtype.startswith("timedelta64["):
        match = re.search(r"timedelta64\[(\w{1,2})", dtype)
        assert match is not None
        time_unit = cast(Literal["ms", "us"], match.group(1))
        return Namespace.Duration(time_unit)
    msg = f"Unsupported dtype! {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def map_standard_dtype_to_pandas_dtype(dtype: DType) -> Any:
    if isinstance(dtype, Namespace.Int64):
        return "int64"
    if isinstance(dtype, Namespace.Int32):
        return "int32"
    if isinstance(dtype, Namespace.Int16):
        return "int16"
    if isinstance(dtype, Namespace.Int8):
        return "int8"
    if isinstance(dtype, Namespace.UInt64):
        return "uint64"
    if isinstance(dtype, Namespace.UInt32):
        return "uint32"
    if isinstance(dtype, Namespace.UInt16):
        return "uint16"
    if isinstance(dtype, Namespace.UInt8):
        return "uint8"
    if isinstance(dtype, Namespace.Float64):
        return "float64"
    if isinstance(dtype, Namespace.Float32):
        return "float32"
    if isinstance(dtype, Namespace.Bool):
        return "bool"
    if isinstance(dtype, Namespace.String):
        return "object"
    if isinstance(dtype, Namespace.Datetime):
        if dtype.time_zone is not None:  # pragma: no cover (todo)
            return f"datetime64[{dtype.time_unit}, {dtype.time_zone}]"
        return f"datetime64[{dtype.time_unit}]"
    if isinstance(dtype, Namespace.Duration):
        return f"timedelta64[{dtype.time_unit}]"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def convert_to_standard_compliant_column(
    ser: pd.Series[Any],
    api_version: str | None = None,
) -> Column:
    if ser.name is not None and not isinstance(ser.name, str):
        msg = f"Expected column with string name, got: {ser.name}"
        raise ValueError(msg)
    if ser.name is None:
        ser = ser.rename("")
    return Column(
        ser,
        api_version=api_version or "2023.11-beta",
        df=None,
        is_persisted=True,
    )


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame,
    api_version: str | None = None,
) -> DataFrame:
    return DataFrame(df, api_version=api_version or "2023.11-beta")


class Namespace(NamespaceT):
    def __init__(self, *, api_version: str) -> None:
        self.__dataframe_api_version__ = api_version
        self._api_version = api_version

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

    class NullType(NullTypeT):
        ...

    null = NullType()

    def dataframe_from_columns(
        self,
        *columns: ColumnT,
    ) -> DataFrame:
        data = {}
        api_versions: set[str] = set()
        for col in columns:
            ser = col._materialise()  # type: ignore[attr-defined]
            data[ser.name] = ser
            api_versions.add(col._api_version)  # type: ignore[attr-defined]
        return DataFrame(pd.DataFrame(data), api_version=list(api_versions)[0])

    def column_from_1d_array(  # type: ignore[override]
        self,
        data: Any,
        *,
        name: str | None = None,
    ) -> Column:
        ser = pd.Series(data, name=name)
        return Column(ser, api_version=self._api_version, df=None, is_persisted=True)

    def column_from_sequence(
        self,
        sequence: Sequence[Any],
        *,
        dtype: DType | None = None,
        name: str = "",
    ) -> Column:
        if dtype is not None:
            ser = pd.Series(
                sequence,
                dtype=map_standard_dtype_to_pandas_dtype(dtype),
                name=name,
            )
        else:
            ser = pd.Series(sequence, name=name)
        return Column(ser, api_version=self._api_version, df=None, is_persisted=True)

    def concat(
        self,
        dataframes: Sequence[DataFrameT],
    ) -> DataFrame:
        dataframes = cast("Sequence[DataFrame]", dataframes)
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
            dfs.append(df.dataframe)
            api_versions.add(df._api_version)
        if len(api_versions) > 1:  # pragma: no cover
            msg = f"Multiple api versions found: {api_versions}"
            raise ValueError(msg)
        return DataFrame(
            pd.concat(
                dfs,
                axis=0,
                ignore_index=True,
            ),
            api_version=api_versions.pop(),
        )

    def dataframe_from_2d_array(
        self,
        data: Any,
        *,
        names: Sequence[str],
    ) -> DataFrame:
        df = pd.DataFrame(data, columns=list(names))
        return DataFrame(df, api_version=self._api_version)

    def is_null(self, value: Any) -> bool:
        return value is self.null

    def is_dtype(self, dtype: DType, kind: str | tuple[str, ...]) -> bool:
        if isinstance(kind, str):
            kind = (kind,)
        dtypes: set[Any] = set()
        for _kind in kind:
            if _kind == "bool":
                dtypes.add(Namespace.Bool)
            if _kind == "signed integer" or _kind == "integral" or _kind == "numeric":
                dtypes |= {
                    Namespace.Int64,
                    Namespace.Int32,
                    Namespace.Int16,
                    Namespace.Int8,
                }
            if _kind == "unsigned integer" or _kind == "integral" or _kind == "numeric":
                dtypes |= {
                    Namespace.UInt64,
                    Namespace.UInt32,
                    Namespace.UInt16,
                    Namespace.UInt8,
                }
            if _kind == "floating" or _kind == "numeric":
                dtypes |= {Namespace.Float64, Namespace.Float32}
            if _kind == "string":
                dtypes.add(Namespace.String)
        return isinstance(dtype, tuple(dtypes))

    def date(self, year: int, month: int, day: int) -> Scalar:
        return Scalar(
            pd.Timestamp(dt.date(year, month, day)),
            api_version=self._api_version,
        )

    # --- horizontal reductions

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

    def unique_indices(
        self,
        *columns: ColumnT,
        skip_nulls: bool | ScalarT = True,
    ) -> Column:
        raise NotImplementedError

    class Aggregation(AggregationT):
        def __init__(self, column_name: str, output_name: str, aggregation: str) -> None:
            self.column_name = column_name
            self.output_name = output_name
            self.aggregation = aggregation

        def __repr__(self) -> str:  # pragma: no cover
            return f"{self.__class__.__name__}({self.column_name!r}, {self.output_name!r}, {self.aggregation!r})"

        def _replace(self, **kwargs: str) -> AggregationT:
            return self.__class__(
                column_name=kwargs.get("column_name", self.column_name),
                output_name=kwargs.get("output_name", self.output_name),
                aggregation=kwargs.get("aggregation", self.aggregation),
            )

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
            return Namespace.Aggregation(column, column, "prod")

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
            return Namespace.Aggregation("__placeholder__", "size", "size")

    def col(self, column_name: str) -> Expr:
        return Expr.from_column_name(column_name)


class Expr:
    def __init__(self, call: Callable[[DataFrame], Column]) -> None:
        self.call = call

    @classmethod
    def from_column_name(cls: type[Expr], column_name: str) -> Expr:
        def call(df: DataFrame) -> Column:
            return Column(
                df.dataframe.loc[:, column_name],
                api_version=df._api_version,
            )

        return cls(call)

    def __getattribute__(self, attr: str) -> Any:
        if attr == "call":
            return super().__getattribute__("call")

        def func(*args: Any, **kwargs: Any) -> Expr:
            def call(df: DataFrame) -> Column:
                return getattr(self.call(df), attr)(  # type: ignore[no-any-return]
                    *[(arg.call(df) if isinstance(arg, Expr) else arg) for arg in args],
                    **{
                        arg_name: (
                            arg_value if isinstance(arg_value, Expr) else arg_value
                        )
                        for arg_name, arg_value in kwargs.items()
                    },
                )

            return Expr(call=call)

        return func

    def __eq__(self, other: Expr | Any) -> Expr:  # type: ignore[override]
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__eq__(other.call(df)))
        return Expr(lambda df: self.call(df).__eq__(other))

    def __ne__(self, other: Expr | Any) -> Expr:  # type: ignore[override]
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__ne__(other.call(df)))
        return Expr(lambda df: self.call(df).__ne__(other))

    def __ge__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__ge__(other.call(df)))
        return Expr(lambda df: self.call(df).__ge__(other))

    def __gt__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__gt__(other.call(df)))
        return Expr(lambda df: self.call(df).__gt__(other))

    def __le__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__le__(other.call(df)))
        return Expr(lambda df: self.call(df).__le__(other))

    def __lt__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__lt__(other.call(df)))
        return Expr(lambda df: self.call(df).__lt__(other))

    def __and__(self, other: Expr | bool | Scalar) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__and__(other.call(df)))
        return Expr(lambda df: self.call(df).__and__(other))

    def __rand__(self, other: Column | Any) -> Column:
        return self.__and__(other)

    def __or__(self, other: Expr | bool | Scalar) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__or__(other.call(df)))
        return Expr(lambda df: self.call(df).__or__(other))

    def __ror__(self, other: Column | Any) -> Column:
        return self.__or__(other)

    def __add__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__add__(other.call(df)))
        return Expr(lambda df: self.call(df).__add__(other))

    def __radd__(self, other: Column | Any) -> Column:
        return self.__add__(other)

    def __sub__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__sub__(other.call(df)))
        return Expr(lambda df: self.call(df).__sub__(other))

    def __rsub__(self, other: Column | Any) -> Column:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__mul__(other.call(df)))
        return Expr(lambda df: self.call(df).__mul__(other))

    def __rmul__(self, other: Column | Any) -> Column:
        return self.__mul__(other)

    def __truediv__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__truediv__(other.call(df)))
        return Expr(lambda df: self.call(df).__truediv__(other))

    def __rtruediv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __floordiv__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__floordiv__(other.call(df)))
        return Expr(lambda df: self.call(df).__floordiv__(other))

    def __rfloordiv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __pow__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__pow__(other.call(df)))
        return Expr(lambda df: self.call(df).__pow__(other))

    def __rpow__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Expr | Any) -> Expr:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__mod__(other.call(df)))
        return Expr(lambda df: self.call(df).__mod__(other))

    def __rmod__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __divmod__(self, other: Expr | Any) -> tuple[Expr, Column]:
        if isinstance(other, Expr):
            return Expr(lambda df: self.call(df).__divmod__(other.call(df)))
        return Expr(lambda df: self.call(df).__divmod__(other))

    # Unary

    def __invert__(self: Column) -> Column:
        return Expr(lambda df: self.call(df).__invert__())
