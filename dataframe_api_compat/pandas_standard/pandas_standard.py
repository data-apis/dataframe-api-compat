from __future__ import annotations

import collections
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype

import dataframe_api_compat.pandas_standard

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
    from collections.abc import Mapping
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

    def __bool__(self) -> bool:
        self.df.validate_is_collected("Scalar.__bool__")
        return self.value.__bool__()  # type: ignore[no-any-return]

    def __int__(self) -> int:
        self.df.validate_is_collected("Scalar.__int__")
        return self.value.__int__()  # type: ignore[no-any-return]

    def __float__(self) -> float:
        self.df.validate_is_collected("Scalar.__float__")
        return self.value.__float__()  # type: ignore[no-any-return]


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


LATEST_API_VERSION = "2023.09-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.08-beta"))


class PandasDataFrame(DataFrame):
    # Not part of the Standard

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        api_version: str,
        is_collected: bool = False,
    ) -> None:
        self._is_collected = is_collected
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe.reset_index(drop=True)
        if api_version not in SUPPORTED_VERSIONS:  # pragma: no cover
            msg = f"Unsupported API version, expected one of: {SUPPORTED_VERSIONS}. "
            f"Got: {api_version}Try updating dataframe-api-compat?"
            raise AssertionError(
                msg,
            )
        self.api_version = api_version

    def validate_is_collected(self, method: str) -> pd.DataFrame:
        if not self._is_collected:
            msg = f"Method {method} requires you to call `.collect` first.\n\nNote: `.collect` forces materialisation in lazy libraries and so should be called as late as possible in your pipeline, and only once per dataframe."
            raise ValueError(
                msg,
            )
        return self.dataframe

    def __repr__(self) -> str:  # pragma: no cover
        return self.dataframe.__repr__()  # type: ignore[no-any-return]

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                msg = f"Expected unique column names, got {col} {count} time(s)"
                raise ValueError(
                    msg,
                )

    def _validate_booleanness(self) -> None:
        if not (
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def _validate_column(self, column: Column) -> None:
        if id(self) != id(column.df):  # type: ignore[attr-defined]
            msg = "cannot compare columns from different dataframes"
            raise ValueError(msg)

    # In the Standard

    def col(self, name: str) -> PandasColumn:
        return PandasColumn(
            self.dataframe.loc[:, name],
            df=self,
            api_version=self.api_version,
        )

    def shape(self) -> tuple[int, int]:
        df = self.validate_is_collected("Column.shape")
        return df.shape  # type: ignore[no-any-return]

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.pandas_standard.map_pandas_dtype_to_standard_dtype(
                dtype.name,
            )
            for column_name, dtype in self.dataframe.dtypes.items()
        }

    def __dataframe_namespace__(
        self,
    ) -> dataframe_api_compat.pandas_standard.PandasNamespace:
        return dataframe_api_compat.pandas_standard.PandasNamespace(
            api_version=self.api_version,
        )

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[start:stop:step],
            api_version=self.api_version,
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def group_by(self, *keys: str) -> PandasGroupBy:
        for key in keys:
            if key not in self.column_names:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        return PandasGroupBy(self.dataframe, keys, api_version=self.api_version)

    def select(self, *columns: str) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.loc[:, list(columns)],
            api_version=self.api_version,
        )

    def get_rows(self, indices: Column) -> PandasDataFrame:
        self._validate_column(indices)
        return PandasDataFrame(
            self.dataframe.iloc[indices.column, :],
            api_version=self.api_version,
        )

    def filter(self, mask: Column) -> PandasDataFrame:
        self._validate_column(mask)
        df = self.dataframe
        df = df.loc[mask.column]
        return PandasDataFrame(df, api_version=self.api_version)

    def assign(self, *columns: Column) -> PandasDataFrame:
        df = self.dataframe.copy()  # TODO: remove defensive copy with CoW?
        for column in columns:
            self._validate_column(column)
            df[column.name] = column.column
        return PandasDataFrame(df, api_version=self.api_version)

    def drop_columns(self, *labels: str) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.drop(list(labels), axis=1),
            api_version=self.api_version,
        )

    def rename_columns(self, mapping: Mapping[str, str]) -> PandasDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            msg = f"Expected Mapping, got: {type(mapping)}"
            raise TypeError(msg)
        return PandasDataFrame(
            self.dataframe.rename(columns=mapping),
            api_version=self.api_version,
        )

    def get_column_names(self) -> list[str]:  # pragma: no cover
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasDataFrame:
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return PandasDataFrame(
            df.sort_values(list(keys), ascending=ascending),
            api_version=self.api_version,
        )

    # Binary operations

    def __eq__(self, other: Any) -> PandasDataFrame:  # type: ignore[override]
        return PandasDataFrame(self.dataframe.__eq__(other), api_version=self.api_version)

    def __ne__(self, other: Any) -> PandasDataFrame:  # type: ignore[override]
        return PandasDataFrame(self.dataframe.__ne__(other), api_version=self.api_version)

    def __ge__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe.__ge__(other), api_version=self.api_version)

    def __gt__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe.__gt__(other), api_version=self.api_version)

    def __le__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe.__le__(other), api_version=self.api_version)

    def __lt__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe.__lt__(other), api_version=self.api_version)

    def __and__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__and__(other),
            api_version=self.api_version,
        )

    def __rand__(self, other: Column | Any) -> PandasDataFrame:
        return self.__and__(other)

    def __or__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe.__or__(other), api_version=self.api_version)

    def __ror__(self, other: Column | Any) -> PandasDataFrame:
        return self.__or__(other)

    def __add__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__add__(other),
            api_version=self.api_version,
        )

    def __radd__(self, other: Column | Any) -> PandasDataFrame:
        return self.__add__(other)

    def __sub__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__sub__(other),
            api_version=self.api_version,
        )

    def __rsub__(self, other: Column | Any) -> PandasDataFrame:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__mul__(other),
            api_version=self.api_version,
        )

    def __rmul__(self, other: Column | Any) -> PandasDataFrame:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__truediv__(other),
            api_version=self.api_version,
        )

    def __rtruediv__(self, other: Column | Any) -> PandasDataFrame:  # pragma: no cover
        raise NotImplementedError

    def __floordiv__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__floordiv__(other),
            api_version=self.api_version,
        )

    def __rfloordiv__(self, other: Column | Any) -> PandasDataFrame:  # pragma: no cover
        raise NotImplementedError

    def __pow__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__pow__(other),
            api_version=self.api_version,
        )

    def __rpow__(self, other: Column | Any) -> PandasDataFrame:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__mod__(other),
            api_version=self.api_version,
        )

    def __rmod__(self, other: Column | Any) -> PandasDataFrame:  # pragma: no cover
        raise NotImplementedError

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PandasDataFrame, PandasDataFrame]:
        quotient, remainder = self.dataframe.__divmod__(other)
        return PandasDataFrame(quotient, api_version=self.api_version), PandasDataFrame(
            remainder,
            api_version=self.api_version,
        )

    # Unary

    def __invert__(self) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(self.dataframe.__invert__(), api_version=self.api_version)

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(
            self.dataframe.any().to_frame().T,
            api_version=self.api_version,
        )

    def all(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(
            self.dataframe.all().to_frame().T,
            api_version=self.api_version,
        )

    def min(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.min().to_frame().T,
            api_version=self.api_version,
        )

    def max(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.max().to_frame().T,
            api_version=self.api_version,
        )

    def sum(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.sum().to_frame().T,
            api_version=self.api_version,
        )

    def prod(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.prod().to_frame().T,
            api_version=self.api_version,
        )

    def median(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.median().to_frame().T,
            api_version=self.api_version,
        )

    def mean(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.mean().to_frame().T,
            api_version=self.api_version,
        )

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.std().to_frame().T,
            api_version=self.api_version,
        )

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.var().to_frame().T,
            api_version=self.api_version,
        )

    # Horizontal reductions

    def all_rowwise(self, *, skip_nulls: bool = True) -> PandasColumn:
        df = self.dataframe
        return PandasColumn(
            df.all(axis=1),
            api_version=self.api_version,
            df=self,
        )

    def any_rowwise(self, *, skip_nulls: bool = True) -> PandasColumn:
        df = self.dataframe
        return PandasColumn(
            df.any(axis=1),
            api_version=self.api_version,
            df=self,
        )

    def sorted_indices(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasColumn:  # pragma: no cover
        raise NotImplementedError

    def unique_indices(
        self,
        *keys: str,
        skip_nulls: bool = True,
    ) -> PandasColumn:  # pragma: no cover
        raise NotImplementedError

    # Transformations

    def is_null(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result: list[pd.Series] = []
        for column in self.dataframe.columns:
            result.append(self.dataframe[column].isna())
        return PandasDataFrame(pd.concat(result, axis=1), api_version=self.api_version)

    def is_nan(self) -> PandasDataFrame:
        result: list[pd.Series] = []
        for column in self.dataframe.columns:
            if is_extension_array_dtype(self.dataframe[column].dtype):
                result.append(
                    np.isnan(self.dataframe[column]).replace(pd.NA, False).astype(bool),
                )
            else:
                result.append(self.dataframe[column].isna())
        return PandasDataFrame(pd.concat(result, axis=1), api_version=self.api_version)

    def fill_nan(self, value: float | pd.NAType) -> PandasDataFrame:
        new_cols = {}
        df = self.dataframe
        for col in df.columns:
            ser = df[col].copy()
            if is_extension_array_dtype(ser.dtype):
                if value is dataframe_api_compat.pandas_standard.PandasNamespace.null:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = pd.NA
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
            else:
                if value is dataframe_api_compat.pandas_standard.PandasNamespace.null:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = np.nan
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
            new_cols[col] = ser
        df = pd.DataFrame(new_cols)
        return PandasDataFrame(df, api_version=self.api_version)

    def fill_null(
        self,
        value: Any,
        *,
        column_names: list[str] | None = None,
    ) -> PandasDataFrame:
        if column_names is None:
            column_names = self.dataframe.columns.tolist()
        assert isinstance(column_names, list)  # help type checkers
        df = self.dataframe.copy()
        for column in column_names:
            col = df[column]
            if is_extension_array_dtype(col.dtype):
                # crazy hack to preserve nan...
                num = pd.Series(
                    np.where(np.isnan(col).fillna(False), 0, col.fillna(value)),
                    dtype=col.dtype,
                )
                other = pd.Series(
                    np.where(np.isnan(col).fillna(False), 0, 1),
                    dtype=col.dtype,
                )
                col = num / other
            else:
                col = col.fillna(value)
            df[column] = col
        return PandasDataFrame(df, api_version=self.api_version)

    # Other

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> PandasDataFrame:
        if how not in ["left", "inner", "outer"]:
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
            raise ValueError(msg)
        assert isinstance(other, PandasDataFrame)
        return PandasDataFrame(
            self.dataframe.merge(
                other.dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
            ),
            api_version=self.api_version,
        )

    def collect(self) -> PandasDataFrame:
        if self._is_collected:
            msg = "Dataframe is already collected"
            raise ValueError(msg)
        return PandasDataFrame(
            self.dataframe,
            api_version=self.api_version,
            is_collected=True,
        )

    def to_array(self, dtype: DType) -> Any:
        self.validate_is_collected("Column.to_array")
        return self.dataframe.to_numpy(dtype)
