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

if TYPE_CHECKING:
    from dataframe_api import Column as ColumnT
    from dataframe_api.typing import DType
    from dataframe_api.typing import NullType

    from dataframe_api_compat.pandas_standard.dataframe_object import DataFrame
    from dataframe_api_compat.pandas_standard.scalar_object import Scalar
else:
    ColumnT = object


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


class Column(ColumnT):
    def __init__(
        self,
        series: pd.Series[Any],
        *,
        df: DataFrame | None,
        api_version: str,
    ) -> None:
        """Parameters
        ----------
        df
            DataFrame this column originates from.
        """
        from dataframe_api_compat.pandas_standard.scalar_object import Scalar

        self._name = series.name or ""
        self._column = series
        self.api_version = api_version
        self.df = df
        self._scalar = Scalar

    def __repr__(self) -> str:  # pragma: no cover
        return self.column.__repr__()  # type: ignore[no-any-return]

    def __iter__(self) -> NoReturn:
        msg = ""
        raise NotImplementedError(msg)

    def _from_series(self, series: pd.Series) -> Column:
        return Column(
            series.reset_index(drop=True),
            api_version=self.api_version,
            df=self.df,
        )

    def _validate_comparand(self, other: Column | Any) -> Column | Any:
        from dataframe_api_compat.pandas_standard.scalar_object import Scalar

        if isinstance(other, Scalar):
            if id(self.df) != id(other.df):
                msg = "cannot compare columns/scalars from different dataframes"
                raise ValueError(
                    msg,
                )
            return other.value
        if isinstance(other, Column):
            if id(self.df) != id(other.df):
                msg = "cannot compare columns from different dataframes"
                raise ValueError(msg)
            return other.column
        return other

    def materialise(self) -> pd.Series:
        if self.df is not None:
            self.df.validate_is_persisted()
        return self.column

    # In the standard
    def __column_namespace__(
        self,
    ) -> dataframe_api_compat.pandas_standard.Namespace:
        return dataframe_api_compat.pandas_standard.Namespace(
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

    @property
    def parent_dataframe(self) -> DataFrame | None:
        return self.df

    def get_rows(self, indices: Column) -> Column:
        return self._from_series(self.column.iloc[indices.column])

    def filter(self, mask: Column) -> Column:
        ser = self.column
        return self._from_series(ser.loc[mask.column])

    def get_value(self, row_number: int) -> Any:
        ser = self.materialise()
        return ser.iloc[row_number]

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> Column:
        return self._from_series(self.column.iloc[start:stop:step])

    # Binary comparisons

    def __eq__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser == other).rename(ser.name)

    def __ne__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser != other).rename(ser.name)

    def __ge__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser >= other).rename(ser.name)

    def __gt__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser > other).rename(ser.name)

    def __le__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser <= other).rename(ser.name)

    def __lt__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        ser = self.column
        return self._from_series(ser < other).rename(ser.name)

    def __and__(self, other: Column | bool) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser & other).rename(ser.name)

    def __rand__(self, other: Column | Any) -> Column:
        return self.__and__(other)

    def __or__(self, other: Column | bool) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser | other).rename(ser.name)

    def __ror__(self, other: Column | Any) -> Column:
        return self.__or__(other)

    def __add__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser + other).rename(ser.name)

    def __radd__(self, other: Column | Any) -> Column:
        return self.__add__(other)

    def __sub__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser - other).rename(ser.name)

    def __rsub__(self, other: Column | Any) -> Column:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser * other).rename(ser.name)

    def __rmul__(self, other: Column | Any) -> Column:
        return self.__mul__(other)

    def __truediv__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser / other).rename(ser.name)

    def __rtruediv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __floordiv__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser // other).rename(ser.name)

    def __rfloordiv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __pow__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser**other).rename(ser.name)

    def __rpow__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Column | Any) -> Column:
        ser = self.column
        other = self._validate_comparand(other)
        return self._from_series(ser % other).rename(ser.name)

    def __rmod__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __divmod__(self, other: Column | Any) -> tuple[Column, Column]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self: Column) -> Column:
        ser = self.column
        return self._from_series(~ser)

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> Scalar:  # type: ignore[override]  # todo
        ser = self.column
        return self._scalar(ser.any(), api_version=self.api_version, df=self.df)

    def all(self, *, skip_nulls: bool = True) -> Scalar:  # type: ignore[override]  # todo
        ser = self.column
        return self._scalar(ser.all(), api_version=self.api_version, df=self.df)

    def min(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(ser.min(), api_version=self.api_version, df=self.df)

    def max(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(ser.max(), api_version=self.api_version, df=self.df)

    def sum(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(ser.sum(), api_version=self.api_version, df=self.df)

    def prod(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(ser.prod(), api_version=self.api_version, df=self.df)

    def median(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(ser.median(), api_version=self.api_version, df=self.df)

    def mean(self, *, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(ser.mean(), api_version=self.api_version, df=self.df)

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(
            ser.std(ddof=correction),
            api_version=self.api_version,
            df=self.df,
        )

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        ser = self.column
        return self._scalar(
            ser.var(ddof=correction),
            api_version=self.api_version,
            df=self.df,
        )

    # Transformations

    def is_null(self) -> Column:
        ser = self.column
        return self._from_series(ser.isna())

    def is_nan(self) -> Column:
        ser = self.column
        if is_extension_array_dtype(ser.dtype):
            return self._from_series(np.isnan(ser).replace(pd.NA, False).astype(bool))
        return self._from_series(ser.isna())

    def sort(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().rename(self.name))
        return self._from_series(ser.sort_values().rename(self.name)[::-1])

    def sorted_indices(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().index.to_series(name=self.name))
        return self._from_series(ser.sort_values().index.to_series(name=self.name)[::-1])

    def is_in(self, values: Column) -> Column:
        ser = self.column
        return self._from_series(ser.isin(values.column))

    def unique_indices(
        self,
        *,
        skip_nulls: bool = True,
    ) -> Column:  # pragma: no cover
        msg = "not yet supported"
        raise NotImplementedError(msg)

    def fill_nan(self, value: float | NullType) -> Column:
        ser = self.column.copy()
        if is_extension_array_dtype(ser.dtype):
            if self.__column_namespace__().is_null(value):
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = pd.NA
            else:
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
        else:
            if self.__column_namespace__().is_null(value):
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = np.nan
            else:
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
        return self._from_series(ser)

    def fill_null(
        self,
        value: Any,
    ) -> Column:
        value = self._validate_comparand(value)
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

    def cumulative_sum(self, *, skip_nulls: bool = True) -> Column:
        ser = self.column
        return self._from_series(ser.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool = True) -> Column:
        ser = self.column
        return self._from_series(ser.cumprod())

    def cumulative_max(self, *, skip_nulls: bool = True) -> Column:
        ser = self.column
        return self._from_series(ser.cummax())

    def cumulative_min(self, *, skip_nulls: bool = True) -> Column:
        ser = self.column
        return self._from_series(ser.cummin())

    def rename(self, name: str) -> Column:
        ser = self.column
        return self._from_series(ser.rename(name))

    def to_array(self) -> Any:
        ser = self.materialise()
        return ser.to_numpy(
            dtype=NUMPY_MAPPING.get(self.column.dtype.name, self.column.dtype.name),
        )

    def __len__(self) -> int:
        ser = self.materialise()
        return len(ser)

    def shift(self, offset: int) -> Column:
        ser = self.column
        return self._from_series(ser.shift(offset))

    # --- temporal methods ---

    def year(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.year)

    def month(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.month)

    def day(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.day)

    def hour(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.hour)

    def minute(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.minute)

    def second(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.second)

    def microsecond(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.microsecond)

    def nanosecond(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.microsecond * 1000 + ser.dt.nanosecond)

    def iso_weekday(self) -> Column:
        ser = self.column
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
        ser = self.column
        return self._from_series(ser.dt.floor(frequency))

    def unix_timestamp(
        self,
        *,
        time_unit: Literal["s", "ms", "us"] = "s",
    ) -> Column:
        ser = self.column
        if ser.dt.tz is None:
            result = ser - datetime(1970, 1, 1)
        else:  # pragma: no cover (todo: tz-awareness)
            result = ser.dt.tz_convert("UTC").dt.tz_localize(None) - datetime(1970, 1, 1)
        if time_unit == "s":
            return self._from_series(
                pd.Series(
                    np.floor(result.dt.total_seconds().astype("float64")),
                    name=ser.name,
                ),
            )
        elif time_unit == "ms":
            return self._from_series(
                pd.Series(
                    np.floor(
                        np.floor(result.dt.total_seconds()) * 1000
                        + result.dt.microseconds // 1000,
                    ),
                    name=ser.name,
                ),
            )
        elif time_unit == "us":
            return self._from_series(
                pd.Series(
                    np.floor(result.dt.total_seconds()) * 1_000_000
                    + result.dt.microseconds,
                    name=ser.name,
                ),
            )
        elif time_unit == "ns":
            return self._from_series(
                pd.Series(
                    (
                        np.floor(result.dt.total_seconds()).astype("Int64") * 1_000_000
                        + result.dt.microseconds.astype("Int64")
                    )
                    * 1000
                    + result.dt.nanoseconds.astype("Int64"),
                    name=ser.name,
                ),
            )
        else:  # pragma: no cover
            msg = "Got invalid time_unit"
            raise AssertionError(msg)
