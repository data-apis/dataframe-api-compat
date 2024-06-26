from __future__ import annotations

import datetime as dt
import re
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import modin.pandas as pd

from dataframe_api_compat.modin_standard.column_object import Column
from dataframe_api_compat.modin_standard.dataframe_object import DataFrame
from dataframe_api_compat.modin_standard.scalar_object import Scalar

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
    # this is copied from the pandas implementation;
    # TODO: need similar tests for modin (tests with extension dtypes)
    if dtype == "int64":
        return Namespace.Int64()
    if dtype == "Int64":  # pragma: no cover
        return Namespace.Int64()
    if dtype == "int32":
        return Namespace.Int32()
    if dtype == "Int32":  # pragma: no cover
        return Namespace.Int32()
    if dtype == "int16":
        return Namespace.Int16()
    if dtype == "Int16":  # pragma: no cover
        return Namespace.Int16()
    if dtype == "int8":
        return Namespace.Int8()
    if dtype == "Int8":  # pragma: no cover
        return Namespace.Int8()
    if dtype == "uint64":
        return Namespace.UInt64()
    if dtype == "UInt64":  # pragma: no cover
        return Namespace.UInt64()
    if dtype == "uint32":
        return Namespace.UInt32()
    if dtype == "UInt32":  # pragma: no cover
        return Namespace.UInt32()
    if dtype == "uint16":
        return Namespace.UInt16()
    if dtype == "UInt16":  # pragma: no cover
        return Namespace.UInt16()
    if dtype == "uint8":
        return Namespace.UInt8()
    if dtype == "UInt8":  # pragma: no cover
        return Namespace.UInt8()
    if dtype == "float64":
        return Namespace.Float64()
    if dtype == "Float64":  # pragma: no cover
        return Namespace.Float64()
    if dtype == "float32":
        return Namespace.Float32()
    if dtype == "Float32":  # pragma: no cover
        return Namespace.Float32()
    if dtype in ("bool", "boolean"):
        # Also for `pandas.core.arrays.boolean.BooleanDtype`
        return Namespace.Bool()
    if dtype == "object":
        return Namespace.String()
    if dtype == "string":  # pragma: no cover
        return Namespace.String()
    if hasattr(dtype, "name"):
        # For types like `numpy.dtypes.DateTime64DType`
        dtype = dtype.name
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
            # TODO: implement `testing` module in modin
            # For example: `pd.testing.assert_series_equal`
            if not df.dataframe.dtypes.equals(dtypes):
                msg = "Expected matching columns"
                raise ValueError(msg)
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
            df=None,
            is_persisted=True,
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
