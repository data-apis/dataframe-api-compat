from __future__ import annotations

import collections
from datetime import datetime
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import Literal
from typing import NoReturn
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype

import dataframe_api_compat.pandas_standard

DType = TypeVar("DType")

NUMPY_MAPPING = {
    "Int64": "int64",
    "Int32": "int32",
    "Int16": "int16",
    "Int8": "int8",
    "UInt64": "uint64",
    "UInt32": "uint32",
    "UInt16": "uint16",
    "UInt8": "uint8",
    "Bool": "bool",
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
    )
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dataframe_api import (
        Bool,
        PermissiveColumn,
        Column,
        DataFrame,
        PermissiveFrame,
        GroupBy,
    )

    ExtraCall = tuple[
        Callable[[pd.Series, pd.Series | None], pd.Series], pd.Series, pd.Series
    ]

else:

    class DataFrame:
        ...

    class PermissiveFrame:
        ...

    class PermissiveColumn(Generic[DType]):
        ...

    class Column:
        ...

    class GroupBy:
        ...

    class Bool:
        ...


class PandasColumn(Column):
    def __init__(
        self,
        series,
        *,
        df: PandasDataFrame,
        api_version: str | None = None,  # todo: propagate
    ) -> None:
        """
        Parameters
        ----------
        root_names
            Columns from DataFrame to consider as inputs to expression.
            If `None`, all input columns are considered.
        output_name
            Name of resulting column.
        base_call
            Call to be applied to DataFrame. Should return a Series.
        extra_calls
            Extra calls to chain to output of `base_call`. Must take Series
            and output Series.
        """
        self._name = series.name or ""
        self._column = series
        self._api_version = api_version
        self._df = df
        # TODO: keep track of output name

    def __repr__(self):
        return self.column.__repr__()

    def _from_series(self, series):
        return PandasColumn(
            series.reset_index(drop=True), api_version=self._api_version, df=self._df
        )

    def _validate_comparand(self, other: Column | Any) -> Column | Any:
        if isinstance(other, PandasColumn):
            if id(self._df) != id(other._df):
                raise ValueError("cannot compare columns from different dataframes")
            if len(other.column) == 1 and len(self.column) > 1:
                # Let pandas take care of broadcasting
                return other.column[0]
            return other.column
        return other

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def name(self) -> str:
        return self._name

    @property
    def column(self):
        return self._column

    def get_rows(self, indices: Column | PermissiveColumn[Any]) -> PandasColumn:
        return self._from_series(self.column.iloc[indices.column])

    def filter(self, mask: Column) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.loc[mask.column])

    def get_value(self, row: int) -> Any:
        if not self._df._is_collected:
            self._df._validate_is_collected("Column.get_value")
        return self.column.iloc[row]

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PandasColumn:
        return self._from_series(self.column.iloc[start:stop:step])

    # Binary comparisons

    def __eq__(self, other: PandasColumn | Any) -> PandasColumn:  # type: ignore[override]
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser == other).rename(ser.name)

    def __ne__(self, other: Column | PermissiveColumn[Any]) -> PandasColumn:  # type: ignore[override]
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

    def __or__(self, other: Column | bool) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser | other).rename(ser.name)

    def __add__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser + other).rename(ser.name)

    def __sub__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser - other).rename(ser.name)

    def __mul__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser * other).rename(ser.name)

    def __truediv__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser / other).rename(ser.name)

    def __floordiv__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser // other).rename(ser.name)

    def __pow__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser**other).rename(ser.name)

    def __mod__(self, other: Column | Any) -> PandasColumn:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser % other).rename(ser.name)

    def __divmod__(self, other: Column | Any) -> tuple[PandasColumn, PandasColumn]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self: PandasColumn) -> PandasColumn:
        ser = self.column
        return self._from_series(~ser)

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> PandasColumn:
        ser = self.column
        return self._from_series(pd.Series([ser.any()], name=self.name))

    def all(self, *, skip_nulls: bool = True) -> PandasColumn:
        ser = self.column
        return self._from_series(pd.Series([ser.all()], name=self.name))

    def min(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.min()], name=self.name))

    def max(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.max()], name=self.name))

    def sum(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.sum()], name=self.name))

    def prod(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.prod()], name=self.name))

    def median(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.median()], name=self.name))

    def mean(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.mean()], name=self.name))

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.std(ddof=correction)], name=self.name))

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._from_series(pd.Series([ser.var(ddof=correction)], name=self.name))

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
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasColumn:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().rename(self.name))
        return self._from_series(ser.sort_values().rename(self.name)[::-1])

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasColumn:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().index.to_series(name=self.name))
        return self._from_series(ser.sort_values().index.to_series(name=self.name)[::-1])

    def is_in(self, values: Column) -> PandasColumn:
        ser = self.column
        return self._from_series(ser.isin(values.column))

    def unique_indices(self, *, skip_nulls: bool = True) -> PandasColumn:
        raise NotImplementedError("not yet supported")

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasColumn:
        ser = self.column.copy()
        ser[cast("pd.Series[bool]", np.isnan(ser)).fillna(False).to_numpy(bool)] = value
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
                np.where(np.isnan(ser).fillna(False), 0, 1), dtype=ser.dtype
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

    @property
    def dt(self) -> ColumnDatetimeAccessor:
        """
        Return accessor with functions which work on temporal dtypes.
        """
        return ColumnDatetimeAccessor(self)

    def to_array(self):
        self._df._validate_is_collected("Column.to_array")
        return self.column.to_numpy(
            dtype=NUMPY_MAPPING.get(self.column.dtype.name, self.column.dtype.name)
        )

    def __len__(self):
        self._df._validate_is_collected("Column.__len__")
        return len(self.column)


class ColumnDatetimeAccessor:
    def __init__(self, column: PandasColumn) -> None:
        self.eager = True
        self.column = column
        self._api_version = column._api_version

    def _from_series(self, series):
        return PandasColumn(
            series.reset_index(drop=True),
            api_version=self._api_version,
            df=self.column._df,
        )

    def year(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.year)

    def month(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.month)

    def day(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.day)

    def hour(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.hour)

    def minute(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.minute)

    def second(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.second)

    def microsecond(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.microsecond)

    def iso_weekday(self) -> Column:
        ser = self.column.column
        return self._from_series(ser.dt.weekday + 1)

    def floor(self, frequency: str) -> Column:
        frequency = (
            frequency.replace("day", "D")
            .replace("hour", "H")
            .replace("minute", "T")
            .replace("second", "S")
            .replace("millisecond", "ms")
            .replace("microsecond", "us")
            .replace("nanosecond", "ns")
        )
        ser = self.column.column
        return self._from_series(ser.dt.floor(frequency))

    def unix_timestamp(self) -> PandasColumn:
        ser = self.column.column
        if ser.dt.tz is None:
            return self._from_series(
                pd.Series(
                    np.floor(
                        ((ser - datetime(1970, 1, 1)).dt.total_seconds()).astype(
                            "float64"
                        )
                    ),
                    name=ser.name,
                )
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
                        ).astype("float64")
                    ),
                    name=ser.name,
                )
            )


class PandasGroupBy(GroupBy):
    def __init__(self, df: pd.DataFrame, keys: Sequence[str], api_version: str) -> None:
        self.df = df
        self.grouped = df.groupby(list(keys), sort=False, as_index=False)
        self.keys = list(keys)
        self._api_version = api_version

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self.df.columns.difference(result.columns)
        if len(failed_columns) > 0:
            # defensive check
            raise AssertionError(
                "Groupby operation could not be performed on columns "
                f"{failed_columns}. Please drop them before calling group_by."
            )

    def size(self) -> PandasDataFrame:
        # pandas-stubs is wrong
        return PandasDataFrame(self.grouped.size(), api_version=self._api_version)  # type: ignore[arg-type]

    def _validate_booleanness(self) -> None:
        if not (
            (self.df.drop(columns=self.keys).dtypes == "bool")
            | (self.df.drop(columns=self.keys).dtypes == "boolean")
        ).all():
            raise ValueError(
                "'function' can only be called on DataFrame "
                "where all dtypes are 'bool'"
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
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PandasDataFrame:
        result = self.grouped.std()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PandasDataFrame:
        result = self.grouped.var()
        self._validate_result(result)
        return PandasDataFrame(result, api_version=self._api_version)


LATEST_API_VERSION = "2023.09-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.08-beta"))


class PandasDataFrame(DataFrame):
    # Not technically part of the standard

    def __init__(
        self, dataframe: pd.DataFrame, api_version: str, is_collected=False
    ) -> None:
        self._is_collected = is_collected
        self._validate_columns(dataframe.columns)  # type: ignore[arg-type]
        self._dataframe = dataframe.reset_index(drop=True)
        if api_version not in SUPPORTED_VERSIONS:
            raise AssertionError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. Got: {api_version}"
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    def _validate_is_collected(self, method: str) -> None:
        if not self._is_collected:
            raise ValueError(
                f"Method {method} requires you to call `.collect` first.\n"
                "\n"
                "Note: `.collect` forces materialisation in lazy libraries and "
                "so should be called as late as possible in your pipeline, and "
                "only once per dataframe."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return self.dataframe.__repr__()

    def col(self, name):
        return PandasColumn(self.dataframe.loc[:, name], df=self)

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.pandas_standard.map_pandas_dtype_to_standard_dtype(
                dtype.name
            )
            for column_name, dtype in self.dataframe.dtypes.items()
        }

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                raise ValueError(
                    f"Expected unique column names, got {col} {count} time(s)"
                )
        for col in columns:
            if not isinstance(col, str):
                raise TypeError(
                    f"Expected column names to be of type str, got {col} "
                    f"of type {type(col)}"
                )

    def _validate_booleanness(self) -> None:
        if not (
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            raise NotImplementedError(
                "'any' can only be called on DataFrame " "where all dtypes are 'bool'"
            )

    def _validate_column(self, column: Column) -> None:
        if id(self) != id(column._df):
            raise ValueError("cannot compare columns from different dataframes")

    # In the standard
    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[start:stop:step], api_version=self._api_version
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def group_by(self, *keys: str) -> PandasGroupBy:
        for key in keys:
            if key not in self.column_names:
                raise KeyError(f"key {key} not present in DataFrame's columns")
        return PandasGroupBy(self.dataframe, keys, api_version=self._api_version)

    def select(self, *columns: str) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.loc[:, list(columns)],
            api_version=self._api_version,
        )

    def get_rows(self, indices: Column) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[indices.column, :],
            api_version=self._api_version,
        )

    def filter(self, mask: Column) -> PandasDataFrame:
        self._validate_column(mask)
        df = self.dataframe
        df = df.loc[mask.column]
        return PandasDataFrame(df, api_version=self._api_version)

    def assign(self, *columns: Column) -> PandasDataFrame:
        df = self.dataframe.copy()  # todo: remove defensive copy with CoW?
        for column in columns:
            self._validate_column(column)
            df[column.name] = column.column
        return PandasDataFrame(df, api_version=self._api_version)

    def drop_columns(self, *labels: str) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.drop(list(labels), axis=1), api_version=self._api_version
        )

    def rename_columns(self, mapping: Mapping[str, str]) -> PandasDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PandasDataFrame(
            self.dataframe.rename(columns=mapping), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:  # pragma: no cover
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
        return self.dataframe.columns.tolist()

    def sort(
        self,
        *keys: str | Column | PermissiveColumn[Any],
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasDataFrame:
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return PandasDataFrame(
            df.sort_values(list(keys), ascending=ascending), api_version=self._api_version
        )

    def __eq__(self, other: Any) -> PandasDataFrame:  # type: ignore[override]
        return PandasDataFrame(
            self.dataframe.__eq__(other), api_version=self._api_version
        )

    def __ne__(self, other: Any) -> PandasDataFrame:  # type: ignore[override]
        return PandasDataFrame(
            self.dataframe.__ne__(other), api_version=self._api_version
        )

    def __ge__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__ge__(other), api_version=self._api_version
        )

    def __gt__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__gt__(other), api_version=self._api_version
        )

    def __le__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__le__(other), api_version=self._api_version
        )

    def __lt__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__lt__(other), api_version=self._api_version
        )

    def __and__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__and__(other), api_version=self._api_version
        )

    def __or__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__or__(other), api_version=self._api_version
        )

    def __add__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__add__(other), api_version=self._api_version
        )

    def __sub__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__sub__(other), api_version=self._api_version
        )

    def __mul__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__mul__(other), api_version=self._api_version
        )

    def __truediv__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__truediv__(other), api_version=self._api_version
        )

    def __floordiv__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__floordiv__(other), api_version=self._api_version
        )

    def __pow__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__pow__(other), api_version=self._api_version
        )

    def __mod__(self, other: Any) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.__mod__(other), api_version=self._api_version
        )

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PandasDataFrame, PandasDataFrame]:
        quotient, remainder = self.dataframe.__divmod__(other)
        return PandasDataFrame(quotient, api_version=self._api_version), PandasDataFrame(
            remainder, api_version=self._api_version
        )

    def __invert__(self) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(self.dataframe.__invert__(), api_version=self._api_version)

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def any(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(
            self.dataframe.any().to_frame().T, api_version=self._api_version
        )

    def all(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(
            self.dataframe.all().to_frame().T, api_version=self._api_version
        )

    def min(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.min().to_frame().T, api_version=self._api_version
        )

    def max(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.max().to_frame().T, api_version=self._api_version
        )

    def sum(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.sum().to_frame().T, api_version=self._api_version
        )

    def prod(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.prod().to_frame().T, api_version=self._api_version
        )

    def median(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.median().to_frame().T, api_version=self._api_version
        )

    def mean(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.mean().to_frame().T, api_version=self._api_version
        )

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.std().to_frame().T, api_version=self._api_version
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.var().to_frame().T, api_version=self._api_version
        )

    def is_null(self, *, skip_nulls: bool = True) -> PandasDataFrame:
        result = []
        for column in self.dataframe.columns:
            result.append(self.dataframe[column].isna())
        return PandasDataFrame(pd.concat(result, axis=1), api_version=self._api_version)

    def is_nan(self) -> PandasDataFrame:
        result = []
        for column in self.dataframe.columns:
            if is_extension_array_dtype(self.dataframe[column].dtype):
                result.append(
                    np.isnan(self.dataframe[column]).replace(pd.NA, False).astype(bool)
                )
            else:
                result.append(self.dataframe[column].isna())
        return PandasDataFrame(pd.concat(result, axis=1), api_version=self._api_version)

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasDataFrame:
        new_cols = {}
        df = self.dataframe
        for col in df.columns:
            ser = df[col].copy()
            if is_extension_array_dtype(ser.dtype):
                if value is null:
                    ser[
                        cast("pd.Series[bool]", np.isnan(ser))
                        .fillna(False)
                        .to_numpy(bool)
                    ] = pd.NA
                else:
                    ser[
                        cast("pd.Series[bool]", np.isnan(ser))
                        .fillna(False)
                        .to_numpy(bool)
                    ] = value
            else:
                if value is null:
                    ser[
                        cast("pd.Series[bool]", np.isnan(ser))
                        .fillna(False)
                        .to_numpy(bool)
                    ] = np.nan
                else:
                    ser[
                        cast("pd.Series[bool]", np.isnan(ser))
                        .fillna(False)
                        .to_numpy(bool)
                    ] = value
            new_cols[col] = ser
        df = pd.DataFrame(new_cols)
        return PandasDataFrame(df, api_version=self._api_version)

    def fill_null(
        self,
        value: Any,
        *,
        column_names: list[str] | None = None,
    ) -> PandasColumn:
        if column_names is None:
            column_names = self.dataframe.columns.tolist()
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
                    np.where(np.isnan(col).fillna(False), 0, 1), dtype=col.dtype
                )
                col = num / other
            else:
                col = col.fillna(value)
            df[column] = col
        return PandasDataFrame(df, api_version=self._api_version)

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> PandasDataFrame:
        if how not in ["left", "inner", "outer"]:
            raise ValueError(f"Expected 'left', 'inner', 'outer', got: {how}")
        assert isinstance(other, PandasDataFrame)
        return PandasDataFrame(
            self.dataframe.merge(
                other.dataframe, left_on=left_on, right_on=right_on, how=how
            ),
            api_version=self._api_version,
        )

    def collect(self) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe, api_version=self._api_version, is_collected=True
        )

    def to_array(self, dtype):
        self._validate_is_collected("Column.to_array")
        return self.dataframe.to_numpy(dtype)
