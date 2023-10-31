from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype

import dataframe_api_compat.pandas_standard
from dataframe_api_compat.pandas_standard.dataframe_object import PandasDataFrame

NUMPY_MAPPING = {
    "Int64": "int64",
    "Int32": "int32",
    "Int16": "int16",
    "Int8": "int8",
    "UInt64": "uint64",
    "UInt32": "uint32",
    "UInt16": "uint16",
    "UInt8": "uint8",
    "boolean": "bool",
}


class Null:
    ...


null = Null()

_ARRAY_API_DTYPES = frozenset(
    (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ),
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api import Aggregation
    from dataframe_api import Column
    from dataframe_api import DataFrame
    from dataframe_api import GroupBy
    from dataframe_api.typing import DType
else:
    Column = object
    DataFrame = object
    GroupBy = object
    Namespace = object
    Aggregation = object


class Scalar:
    def __init__(self, value: Any, api_version: str, df: PandasDataFrame) -> None:
        self.value = value
        self._api_version = api_version
        self.df = df

    def _validate_other(self, other: Any) -> Any:
        if isinstance(other, (PandasColumn, PandasDataFrame)):
            return NotImplemented
        if isinstance(other, Scalar):
            if id(self.df) != id(other.df):
                msg = "cannot compare columns/scalars from different dataframes"
                raise ValueError(
                    msg,
                )
            return other.value
        return other

    def _from_scalar(self, scalar: Scalar) -> Scalar:
        return Scalar(scalar, df=self.df, api_version=self._api_version)

    def force_materialise(self) -> Any:
        # Just for testing/debugging (for compatibility with PolarsScalar)
        return self.value

    def __lt__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__lt__(other))

    def __le__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__le__(other))

    def __eq__(self, other: Any) -> Scalar:  # type: ignore[override]
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__eq__(other))

    def __ne__(self, other: Any) -> Scalar:  # type: ignore[override]
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__ne__(other))

    def __gt__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__gt__(other))

    def __ge__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__ge__(other))

    def __add__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__add__(other))

    def __radd__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__radd__(other))

    def __sub__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__sub__(other))

    def __rsub__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rsub__(other))

    def __mul__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__mul__(other))

    def __rmul__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rmul__(other))

    def __mod__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__mod__(other))

    def __rmod__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rmod__(other))

    def __pow__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__pow__(other))

    def __rpow__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rpow__(other))

    def __floordiv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__floordiv__(other))

    def __rfloordiv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rfloordiv__(other))

    def __truediv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__truediv__(other))

    def __rtruediv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rtruediv__(other))

    def __neg__(self) -> Scalar:
        self.df.validate_is_collected("Scalar.__neg__")
        return self.value.__neg__()  # type: ignore[no-any-return]

    def __pos__(self) -> Scalar:
        self.df.validate_is_collected("Scalar.__pos__")
        return self.value.__pos__()  # type: ignore[no-any-return]

    def __abs__(self) -> bool:
        self.df.validate_is_collected("Scalar.__abs__")
        return self.value.__abs__()  # type: ignore[no-any-return]

    def __bool__(self) -> bool:
        self.df.validate_is_collected("Scalar.__bool__")
        return self.value.__bool__()  # type: ignore[no-any-return]

    def __int__(self) -> int:
        self.df.validate_is_collected("Scalar.__int__")
        return self.value.__int__()  # type: ignore[no-any-return]

    def __float__(self) -> float:
        self.df.validate_is_collected("Scalar.__float__")
        return self.value.__float__()  # type: ignore[no-any-return]

    def __repr__(self) -> str:  # pragma: no cover
        return self.value.__repr__()  # type: ignore[no-any-return]


class PandasColumn(Column):
    def __init__(
        self,
        series: pd.Series[Any],
        *,
        df: PandasDataFrame,
        api_version: str,  # TODO: propagate
    ) -> None:
        """Parameters
        ----------
        df
            DataFrame this column originates from.
        """
        self._name = series.name or ""
        self._column = series
        self.api_version = api_version
        self.df = df

    def __repr__(self) -> str:  # pragma: no cover
        return self.column.__repr__()  # type: ignore[no-any-return]

    def __iter__(self) -> NoReturn:
        msg = ""
        raise NotImplementedError(msg)

    def _from_series(self, series: pd.Series) -> PandasColumn:
        return PandasColumn(
            series.reset_index(drop=True),
            api_version=self.api_version,
            df=self.df,
        )

    def _validate_comparand(self, other: Column | Any) -> Column | Any:
        if isinstance(other, Scalar):
            if id(self.df) != id(other.df):
                msg = "cannot compare columns/scalars from different dataframes"
                raise ValueError(
                    msg,
                )
            return other.value
        if isinstance(other, PandasColumn):
            if id(self.df) != id(other.df):
                msg = "cannot compare columns from different dataframes"
                raise ValueError(msg)
            return other.column
        return other

    # In the standard
    def __column_namespace__(
        self,
    ) -> dataframe_api_compat.pandas_standard.PandasNamespace:
        return dataframe_api_compat.pandas_standard.PandasNamespace(
            api_version=self.api_version,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def column(self) -> pd.Series[Any]:
        return self._column

    @property
    def dtype(self) -> DType:
        return dataframe_api_compat.pandas_standard.map_pandas_dtype_to_standard_dtype(
            self._column.dtype,
        )

    def get_rows(self, indices: Column) -> PandasColumn:
        return self._from_series(self.column.iloc[indices.column])

    def filter(self, mask: Column) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.loc[mask.column])

    def get_value(self, row_number: int) -> Any:
        self.df.validate_is_collected("Column.get_value")
        return self.column.iloc[row_number]

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> PandasColumn:
        return self._from_series(self.column.iloc[start:stop:step])

    # Binary comparisons

    def __eq__(self, other: PandasColumn | Any) -> PandasColumn:  # type: ignore[override]
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser == other).rename(ser.name)

    def __ne__(self, other: Column | Any) -> PandasColumn:  # type: ignore[override]
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser != other).rename(ser.name)

    def __ge__(self, other: Column | Any) -> PandasColumn:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser >= other).rename(ser.name)

    def __gt__(self, other: Column | Any) -> PandasColumn:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser > other).rename(ser.name)

    def __le__(self, other: Column | Any) -> PandasColumn:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser <= other).rename(ser.name)

    def __lt__(self, other: Column | Any) -> PandasColumn:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser < other).rename(ser.name)

    def __and__(self, other: Column | bool) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser & other).rename(ser.name)

    def __rand__(self, other: Column | Any) -> PandasColumn:
        return self.__and__(other)

    def __or__(self, other: Column | bool) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser | other).rename(ser.name)

    def __ror__(self, other: Column | Any) -> PandasColumn:
        return self.__or__(other)

    def __add__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser + other).rename(ser.name)

    def __radd__(self, other: Column | Any) -> PandasColumn:
        return self.__add__(other)

    def __sub__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser - other).rename(ser.name)

    def __rsub__(self, other: Column | Any) -> PandasColumn:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser * other).rename(ser.name)

    def __rmul__(self, other: Column | Any) -> PandasColumn:
        return self.__mul__(other)

    def __truediv__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser / other).rename(ser.name)

    def __rtruediv__(self, other: Column | Any) -> PandasColumn:
        raise NotImplementedError

    def __floordiv__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser // other).rename(ser.name)

    def __rfloordiv__(self, other: Column | Any) -> PandasColumn:
        raise NotImplementedError

    def __pow__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser**other).rename(ser.name)

    def __rpow__(self, other: Column | Any) -> PandasColumn:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser % other).rename(ser.name)

    def __rmod__(self, other: Column | Any) -> PandasColumn:  # pragma: no cover
        raise NotImplementedError

    def __divmod__(self, other: Column | Any) -> tuple[PandasColumn, PandasColumn]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self: PandasColumn) -> PandasColumn:
        ser = self.column
        return self._from_series(~ser)

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> Scalar:
        ser = self.column
        return Scalar(ser.any(), api_version=self.api_version, df=self.df)

    def all(self, *, skip_nulls: bool = True) -> Scalar:
        ser = self.column
        return Scalar(ser.all(), api_version=self.api_version, df=self.df)

    def min(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.min(), api_version=self.api_version, df=self.df)

    def max(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.max(), api_version=self.api_version, df=self.df)

    def sum(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.sum(), api_version=self.api_version, df=self.df)

    def prod(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.prod(), api_version=self.api_version, df=self.df)

    def median(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.median(), api_version=self.api_version, df=self.df)

    def mean(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.mean(), api_version=self.api_version, df=self.df)

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.std(ddof=correction), api_version=self.api_version, df=self.df)

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        ser = self.column
        return Scalar(ser.var(ddof=correction), api_version=self.api_version, df=self.df)

    # Transformations

    def is_null(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.isna())

    def is_nan(self) -> PandasColumn:
        ser = self.column
        if is_extension_array_dtype(ser.dtype):
            return self._from_series(np.isnan(ser).replace(pd.NA, False).astype(bool))
        return self._from_series(ser.isna())

    def sort(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasColumn:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().rename(self.name))
        return self._from_series(ser.sort_values().rename(self.name)[::-1])

    def sorted_indices(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasColumn:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().index.to_series(name=self.name))
        return self._from_series(ser.sort_values().index.to_series(name=self.name)[::-1])

    def is_in(self, values: Column) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.isin(values.column))

    def unique_indices(
        self,
        *,
        skip_nulls: bool = True,
    ) -> PandasColumn:  # pragma: no cover
        msg = "not yet supported"
        raise NotImplementedError(msg)

    def fill_nan(self, value: float | pd.NAType) -> PandasColumn:
        ser = self.column.copy()
        ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
        return self._from_series(ser)

    def fill_null(
        self,
        value: Any,
    ) -> PandasColumn:
        ser = self.column.copy()
        if is_extension_array_dtype(ser.dtype):
            # crazy hack to preserve nan...
            num = pd.Series(
                np.where(np.isnan(ser).fillna(False), 0, ser.fillna(value)),
                dtype=ser.dtype,
            )
            other = pd.Series(
                np.where(np.isnan(ser).fillna(False), 0, 1),
                dtype=ser.dtype,
            )
            ser = num / other
        else:
            ser = ser.fillna(value)
        return self._from_series(ser.rename(self.name))

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.cumprod())

    def cumulative_max(self, *, skip_nulls: bool = True) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.cummax())

    def cumulative_min(self, *, skip_nulls: bool = True) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.cummin())

    def rename(self, name: str) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.rename(name))

    def to_array(self) -> Any:
        self.df.validate_is_collected("Column.to_array")
        return self.column.to_numpy(
            dtype=NUMPY_MAPPING.get(self.column.dtype.name, self.column.dtype.name),
        )

    def __len__(self) -> int:
        self.df.validate_is_collected("Column.__len__")
        return len(self.column)

    def year(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.year)

    def month(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.month)

    def day(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.day)

    def hour(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.hour)

    def minute(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.minute)

    def second(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.second)

    def microsecond(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.microsecond)

    def nanosecond(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.microsecond * 1000 + ser.dt.nanosecond)

    def iso_weekday(self) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.dt.weekday + 1)

    def floor(self, frequency: str) -> PandasColumn:
        frequency = (
            frequency.replace("day", "D")
            .replace("hour", "H")
            .replace("minute", "T")
            .replace("second", "S")
            .replace("millisecond", "ms")
            .replace("microsecond", "us")
            .replace("nanosecond", "ns")
        )
        ser = self.column
        return self._from_series(ser.dt.floor(frequency))

    def unix_timestamp(self) -> PandasColumn:
        ser = self.column
        if ser.dt.tz is None:
            return self._from_series(
                pd.Series(
                    np.floor(
                        ((ser - datetime(1970, 1, 1)).dt.total_seconds()).astype(
                            "float64",
                        ),
                    ),
                    name=ser.name,
                ),
            )
        else:  # pragma: no cover (todo: tz-awareness)
            return self._from_series(
                pd.Series(
                    np.floor(
                        (
                            (
                                ser.dt.tz_convert("UTC").dt.tz_localize(None)
                                - datetime(1970, 1, 1)
                            ).dt.total_seconds()
                        ).astype("float64"),
                    ),
                    name=ser.name,
                ),
            )


class PandasGroupBy(GroupBy):
    def __init__(self, df: pd.DataFrame, keys: Sequence[str], api_version: str) -> None:
        self.df = df
        self.grouped = df.groupby(list(keys), sort=False, as_index=False)
        self.keys = list(keys)
        self._api_version = api_version

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self.df.columns.difference(result.columns)
        if len(failed_columns) > 0:  # pragma: no cover
            msg = "Groupby operation could not be performed on columns "
            f"{failed_columns}. Please drop them before calling group_by."
            raise AssertionError(
                msg,
            )

    def size(self) -> PandasDataFrame:
        return PandasDataFrame(self.grouped.size(), api_version=self._api_version)

    def _validate_booleanness(self) -> None:
        if not (
            (self.df.drop(columns=self.keys).dtypes == "bool")
            | (self.df.drop(columns=self.keys).dtypes == "boolean")
        ).all():
            msg = "'function' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def any(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        self._validate_booleanness()
        result = self.grouped.any()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def all(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        self._validate_booleanness()
        result = self.grouped.all()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def min(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = self.grouped.min()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def max(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = self.grouped.max()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def sum(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = self.grouped.sum()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def prod(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = self.grouped.prod()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def median(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = self.grouped.median()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def mean(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = self.grouped.mean()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PandasDataFrame:
        result = self.grouped.std()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PandasDataFrame:
        result = self.grouped.var()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def aggregate(  # type: ignore[override]
        self,
        *aggregations: dataframe_api_compat.pandas_standard.PandasNamespace.Aggregation,
    ) -> PandasDataFrame:  # pragma: no cover
        output_names = [aggregation.output_name for aggregation in aggregations]

        include_size = False
        size_output_name = None
        column_aggregations: list[
            dataframe_api_compat.pandas_standard.PandasNamespace.Aggregation
        ] = []
        for aggregation in aggregations:
            if aggregation.aggregation == "size":
                include_size = True
                size_output_name = aggregation.output_name
            else:
                column_aggregations.append(aggregation)

        agg = {
            aggregation.column_name: aggregation.aggregation
            for aggregation in column_aggregations
        }
        if agg:
            aggregated = self.grouped.agg(agg).rename(
                {
                    aggregation.column_name: aggregation.output_name
                    for aggregation in column_aggregations
                },
                axis=1,
            )

        if include_size:
            size = self.grouped.size().drop(self.keys, axis=1)
            assert len(size.columns) == 1
            size = size.rename(columns={size.columns[0]: size_output_name})

        if agg and include_size:
            df = pd.concat([aggregated, size], axis=1)
        elif agg:
            df = aggregated
        elif include_size:
            df = size
        else:
            msg = "No aggregations specified"
            raise ValueError(msg)
        return PandasDataFrame(
            df.loc[:, output_names],
            api_version=self._api_version,
            is_collected=False,
        )
