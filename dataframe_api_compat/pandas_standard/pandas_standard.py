from __future__ import annotations

import collections
from typing import Any
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
        Column,
        DataFrame,
        GroupBy,
    )
else:

    class DataFrame(Generic[DType]):
        ...

    class Column(Generic[DType]):
        ...

    class GroupBy:
        ...


class PandasColumn(Column[DType]):
    # private, not technically part of the standard
    def __init__(self, column: pd.Series[Any], api_version: str) -> None:
        if (
            isinstance(column.index, pd.RangeIndex)
            and column.index.start == 0  # type: ignore[comparison-overlap]
            and column.index.step == 1  # type: ignore[comparison-overlap]
            and (column.index.stop == len(column))  # type: ignore[comparison-overlap]
        ):
            self._series = column
        else:
            self._series = column.reset_index(drop=True)
        self._api_version = api_version
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )

    def _validate_index(self, index: pd.Index) -> None:
        pd.testing.assert_index_equal(self.column.index, index)

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def name(self) -> str:
        return self.column.name  # type: ignore[return-value]

    @property
    def column(self) -> pd.Series[Any]:
        return self._series

    def __len__(self) -> int:
        return len(self.column)

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    @property
    def dtype(self) -> Any:
        return dataframe_api_compat.pandas_standard.DTYPE_MAP[self.column.dtype.name]

    def get_rows(self, indices: Column[Any]) -> PandasColumn[DType]:
        return PandasColumn(
            self.column.iloc[indices.column.to_numpy()], api_version=self._api_version
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PandasColumn[DType]:
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.column)
        if step is None:
            step = 1
        return PandasColumn(
            self.column.iloc[start:stop:step], api_version=self._api_version
        )

    def filter(self, mask: Column[Bool]) -> PandasColumn[DType]:
        series = mask.column
        self._validate_index(series.index)
        return PandasColumn(self.column.loc[series], api_version=self._api_version)

    def get_value(self, row: int) -> Any:
        return self.column.iloc[row]

    def __eq__(  # type: ignore[override]
        self, other: PandasColumn[DType] | Any
    ) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(
                self.column == other.column, api_version=self._api_version
            )
        return PandasColumn(self.column == other, api_version=self._api_version)

    def __ne__(  # type: ignore[override]
        self, other: Column[DType]
    ) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(
                self.column != other.column, api_version=self._api_version
            )
        return PandasColumn(self.column != other, api_version=self._api_version)

    def __ge__(self, other: Column[DType] | Any) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(
                self.column >= other.column, api_version=self._api_version
            )
        return PandasColumn(self.column >= other, api_version=self._api_version)

    def __gt__(self, other: Column[DType] | Any) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column > other.column, api_version=self._api_version)
        return PandasColumn(self.column > other, api_version=self._api_version)

    def __le__(self, other: Column[DType] | Any) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(
                self.column <= other.column, api_version=self._api_version
            )
        return PandasColumn(self.column <= other, api_version=self._api_version)

    def __lt__(self, other: Column[DType] | Any) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column < other.column, api_version=self._api_version)
        return PandasColumn(self.column < other, api_version=self._api_version)

    def __and__(self, other: Column[Bool] | bool) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column & other.column, api_version=self._api_version)
        result = self.column & other  # type: ignore[operator]
        return PandasColumn(result, api_version=self._api_version)

    def __or__(self, other: Column[Bool] | bool) -> PandasColumn[Bool]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column | other.column, api_version=self._api_version)
        return PandasColumn(self.column | other, api_version=self._api_version)  # type: ignore[operator]

    def __add__(self, other: Column[DType] | Any) -> PandasColumn[DType]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column + other.column, api_version=self._api_version)
        return PandasColumn(self.column + other, api_version=self._api_version)  # type: ignore[operator]

    def __sub__(self, other: Column[DType] | Any) -> PandasColumn[DType]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column - other.column, api_version=self._api_version)
        return PandasColumn(self.column - other, api_version=self._api_version)  # type: ignore[operator]

    def __mul__(self, other: Column[DType] | Any) -> PandasColumn[Any]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column * other.column, api_version=self._api_version)
        return PandasColumn(self.column * other, api_version=self._api_version)  # type: ignore[operator]

    def __truediv__(self, other: Column[DType] | Any) -> PandasColumn[Any]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column / other.column, api_version=self._api_version)
        return PandasColumn(self.column / other, api_version=self._api_version)  # type: ignore[operator]

    def __floordiv__(self, other: Column[DType] | Any) -> PandasColumn[Any]:
        if isinstance(other, PandasColumn):
            return PandasColumn(
                self.column // other.column, api_version=self._api_version
            )
        return PandasColumn(self.column // other, api_version=self._api_version)  # type: ignore[operator]

    def __pow__(self, other: Column[DType] | Any) -> PandasColumn[Any]:
        if isinstance(other, PandasColumn):
            return PandasColumn(
                self.column**other.column, api_version=self._api_version
            )
        return PandasColumn(self.column**other, api_version=self._api_version)  # type: ignore[operator]

    def __mod__(self, other: Column[DType] | Any) -> PandasColumn[Any]:
        if isinstance(other, PandasColumn):
            return PandasColumn(self.column % other.column, api_version=self._api_version)
        return PandasColumn(self.column % other, api_version=self._api_version)  # type: ignore[operator]

    def __divmod__(
        self, other: Column[DType] | Any
    ) -> tuple[PandasColumn[Any], PandasColumn[Any]]:
        if isinstance(other, PandasColumn):
            quotient, remainder = self.column.__divmod__(other.column)
        else:
            quotient, remainder = self.column.__divmod__(other)
        return PandasColumn(quotient, api_version=self._api_version), PandasColumn(
            remainder, api_version=self._api_version
        )

    def __invert__(self: PandasColumn[Bool]) -> PandasColumn[Bool]:
        return PandasColumn(~self.column, api_version=self._api_version)

    def any(self, *, skip_nulls: bool = True) -> bool:
        return self.column.any()

    def all(self, *, skip_nulls: bool = True) -> bool:
        return self.column.all()

    def min(self, *, skip_nulls: bool = True) -> Any:
        return self.column.min()

    def max(self, *, skip_nulls: bool = True) -> Any:
        return self.column.max()

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return self.column.sum()

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return self.column.prod()

    def median(self, *, skip_nulls: bool = True) -> Any:
        return self.column.median()

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return self.column.mean()

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self.column.std()

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self.column.var()

    def is_null(self) -> PandasColumn[Bool]:
        return PandasColumn(self.column.isna(), api_version=self._api_version)

    def is_nan(self) -> PandasColumn[Bool]:
        if is_extension_array_dtype(self.column.dtype):
            return PandasColumn(
                np.isnan(self.column).replace(pd.NA, False).astype(bool),
                api_version=self._api_version,
            )
        return PandasColumn(self.column.isna(), api_version=self._api_version)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasColumn[Any]:
        if ascending:
            return PandasColumn(
                pd.Series(self.column.argsort()), api_version=self._api_version
            )
        return PandasColumn(
            pd.Series(self.column.argsort()[::-1]), api_version=self._api_version
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasColumn[Any]:
        if self._api_version == "2023.08-beta":
            raise NotImplementedError("dataframe.sort only available after 2023.08-beta")
        return PandasColumn(
            self.column.sort_values(ascending=ascending), api_version=self._api_version
        )

    def is_in(self, values: Column[DType]) -> PandasColumn[Bool]:
        if values.dtype != self.dtype:
            raise ValueError(f"`value` has dtype {values.dtype}, expected {self.dtype}")
        return PandasColumn(
            self.column.isin(values.column), api_version=self._api_version
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PandasColumn[Any]:
        return PandasColumn(
            self.column.drop_duplicates().index.to_series(), api_version=self._api_version
        )

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasColumn[DType]:
        ser = self.column.copy()
        ser[cast("pd.Series[bool]", np.isnan(ser)).fillna(False).to_numpy(bool)] = value
        return PandasColumn(ser, api_version=self._api_version)

    def fill_null(
        self,
        value: Any,
    ) -> PandasColumn[DType]:
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
        return PandasColumn(pd.Series(ser), api_version=self._api_version)

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return PandasColumn(self.column.cumsum(), api_version=self._api_version)

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return PandasColumn(self.column.cumprod(), api_version=self._api_version)

    def cumulative_max(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return PandasColumn(self.column.cummax(), api_version=self._api_version)

    def cumulative_min(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return PandasColumn(self.column.cummin(), api_version=self._api_version)

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy(dtype=dtype)

    def rename(self, name: str | None) -> PandasColumn[DType]:
        return PandasColumn(self.column.rename(name), api_version=self._api_version)


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
                f"{failed_columns}. Please drop them before calling groupby."
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

    def __init__(self, dataframe: pd.DataFrame, api_version: str) -> None:
        self._validate_columns(dataframe.columns)  # type: ignore[arg-type]
        if (
            isinstance(dataframe.index, pd.RangeIndex)
            and dataframe.index.start == 0  # type: ignore[comparison-overlap]
            and dataframe.index.step == 1  # type: ignore[comparison-overlap]
            and (
                dataframe.index.stop == len(dataframe)  # type: ignore[comparison-overlap]
            )
        ):
            self._dataframe = dataframe
        else:
            self._dataframe = dataframe.reset_index(drop=True)
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. Got: {api_version}"
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

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

    def _validate_index(self, index: pd.Index) -> None:
        pd.testing.assert_index_equal(self.dataframe.index, index)

    def _validate_booleanness(self) -> None:
        if not (
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            raise NotImplementedError(
                "'any' can only be called on DataFrame " "where all dtypes are 'bool'"
            )

    # In the standard
    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def shape(self) -> tuple[int, int]:
        return self.dataframe.shape

    def groupby(self, keys: Sequence[str]) -> PandasGroupBy:
        if not isinstance(keys, collections.abc.Sequence):
            raise TypeError(f"Expected sequence of strings, got: {type(keys)}")
        if isinstance(keys, str):
            raise TypeError("Expected sequence of strings, got: str")
        for key in keys:
            if key not in self.get_column_names():
                raise KeyError(f"key {key} not present in DataFrame's columns")
        return PandasGroupBy(self.dataframe, keys, api_version=self._api_version)

    def get_column_by_name(self, name: str) -> PandasColumn[DType]:
        if not isinstance(name, str):
            raise ValueError(f"Expected str, got: {type(name)}")
        return PandasColumn(self.dataframe.loc[:, name], api_version=self._api_version)

    def select(self, names: Sequence[str]) -> PandasDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        self._validate_columns(names)
        return PandasDataFrame(
            self.dataframe.loc[:, list(names)], api_version=self._api_version
        )

    def get_rows(self, indices: Column[Any]) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[indices.column, :], api_version=self._api_version
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PandasDataFrame:
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.dataframe)
        if step is None:
            step = 1
        return PandasDataFrame(
            self.dataframe.iloc[start:stop:step], api_version=self._api_version
        )

    def filter(self, mask: Column[Bool]) -> PandasDataFrame:
        series = mask.column
        self._validate_index(series.index)
        return PandasDataFrame(
            self.dataframe.loc[series, :], api_version=self._api_version
        )

    def insert(self, loc: int, label: str, value: Column[Any]) -> PandasDataFrame:
        series = value.column
        self._validate_index(series.index)
        before = self.dataframe.iloc[:, :loc]
        after = self.dataframe.iloc[:, loc:]
        to_insert = value.column.rename(label)
        return PandasDataFrame(
            pd.concat([before, to_insert, after], axis=1), api_version=self._api_version
        )

    def insert_column(self, value: Column[Any]) -> PandasDataFrame:
        series = value.column
        self._validate_index(series.index)
        before = self.dataframe
        to_insert = value.column
        return PandasDataFrame(
            pd.concat([before, to_insert], axis=1), api_version=self._api_version
        )

    def update_columns(self, columns: PandasColumn[Any] | Sequence[PandasColumn[Any]], /) -> PandasDataFrame:  # type: ignore[override]
        if isinstance(columns, PandasColumn):
            columns = [columns]
        df = self.dataframe.copy()
        for col in columns:
            self._validate_index(col.column.index)
            if col.name not in df.columns:
                raise ValueError(
                    f"column {col.name} not in dataframe, use insert instead"
                )
            df[col.name] = col.column
        return PandasDataFrame(df, api_version=self._api_version)

    def drop_column(self, label: str) -> PandasDataFrame:
        if not isinstance(label, str):
            raise TypeError(f"Expected str, got: {type(label)}")
        return PandasDataFrame(
            self.dataframe.drop(label, axis=1), api_version=self._api_version
        )

    def rename_columns(self, mapping: Mapping[str, str]) -> PandasDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PandasDataFrame(
            self.dataframe.rename(columns=mapping), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()

    def sorted_indices(
        self,
        keys: Sequence[str] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasColumn[Any]:
        if keys is None:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe.loc[:, list(keys)]
        if ascending:
            return PandasColumn(
                df.sort_values(keys).index.to_series(), api_version=self._api_version
            )
        return PandasColumn(
            df.sort_values(keys).index.to_series()[::-1], api_version=self._api_version
        )

    def sort(
        self,
        keys: Sequence[str] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasDataFrame:
        if keys is None:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return PandasDataFrame(
            df.sort_values(keys, ascending=ascending), api_version=self._api_version
        )

    def unique_indices(
        self,
        keys: Sequence[str] | None = None,
        *,
        skip_nulls: bool = True,
    ) -> PandasColumn[Any]:
        return PandasColumn(
            self.dataframe.drop_duplicates(subset=keys).index.to_series(),
            api_version=self._api_version,
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

    def any_rowwise(self, *, skip_nulls: bool = True) -> PandasColumn[Bool]:
        self._validate_booleanness()
        return PandasColumn(self.dataframe.any(axis=1), api_version=self._api_version)

    def all_rowwise(self, *, skip_nulls: bool = True) -> PandasColumn[Bool]:
        self._validate_booleanness()
        return PandasColumn(self.dataframe.all(axis=1), api_version=self._api_version)

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
    ) -> PandasDataFrame:
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

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.dataframe.to_numpy(dtype=dtype)

    def join(
        self,
        other: DataFrame,
        left_on: str | list[str],
        right_on: str | list[str],
        how: Literal["left", "inner", "outer"],
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
