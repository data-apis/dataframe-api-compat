from __future__ import annotations

import collections
from typing import Any
from typing import cast
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
        Expression,
        DataFrame,
        GroupBy,
    )
else:

    class DataFrame:
        ...

    class Expression:
        ...

    class GroupBy:
        ...


def col(label: str):
    return lambda df: df.loc[:, label]


class PandasExpression(Expression):
    def __init__(self, base_call=None, extra_calls=None) -> None:
        self._base_call = base_call
        self._calls = extra_calls or []

    def _record_call(self, kind, func, rhs):
        calls = [*self._calls, (kind, func, self, rhs)]
        return PandasExpression(extra_calls=calls)

    def get_rows(self, indices: Expression) -> PandasExpression[DType]:
        return self._record_call(
            "binary",
            lambda ser, indices: ser.iloc[indices].reset_index(drop=True),
            indices,
        )

    def get_rows_by_mask(self, mask: Expression) -> PandasExpression[DType]:
        return self._record_call("binary", lambda ser, mask: ser.loc[mask], mask)

    def __eq__(  # type: ignore[override]
        self, other: PandasExpression[DType] | Any
    ) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser == other, other)

    def __ne__(  # type: ignore[override]
        self, other: Expression
    ) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser != other, other)

    def __ge__(self, other: Expression | Any) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser >= other, other)

    def __gt__(self, other: Expression | Any) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser > other, other)

    def __le__(self, other: Expression | Any) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser <= other, other)

    def __lt__(self, other: Expression | Any) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser < other, other)

    def __and__(self, other: Expression | bool) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser & other, other)

    def __or__(self, other: Expression | bool) -> PandasExpression[Bool]:
        return self._record_call("binary", lambda ser, other: ser | other, other)

    def __add__(self, other: Expression | Any) -> PandasExpression[DType]:
        return self._record_call("binary", lambda ser, other: ser + other, other)

    def __sub__(self, other: Expression | Any) -> PandasExpression[DType]:
        return self._record_call("binary", lambda ser, other: ser - other, other)

    def __mul__(self, other: Expression | Any) -> PandasExpression[Any]:
        return self._record_call("binary", lambda ser, other: ser * other, other)

    def __truediv__(self, other: Expression | Any) -> PandasExpression[Any]:
        return self._record_call("binary", lambda ser, other: ser / other, other)

    def __floordiv__(self, other: Expression | Any) -> PandasExpression[Any]:
        return self._record_call("binary", lambda ser, other: ser // other, other)

    def __pow__(self, other: Expression | Any) -> PandasExpression[Any]:
        return self._record_call("binary", lambda ser, other: ser**other, other)

    def __mod__(self, other: Expression | Any) -> PandasExpression[Any]:
        return self._record_call("binary", lambda ser, other: ser % other, other)

    def __divmod__(
        self, other: Expression | Any
    ) -> tuple[PandasExpression[Any], PandasExpression[Any]]:
        if isinstance(other, PandasExpression):
            quotient, remainder = self.column.__divmod__(other.column)
        else:
            quotient, remainder = self.column.__divmod__(other)
        return PandasExpression(
            quotient, api_version=self._api_version
        ), PandasExpression(remainder, api_version=self._api_version)

    def __invert__(self: PandasExpression[Bool]) -> PandasExpression[Bool]:
        return self._record_call("unary", lambda ser: ~ser, None)

    def min(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.min(), None)

    def max(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.max(), None)

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.sum(), None)

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.prod(), None)

    def median(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.median(), None)

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.mean(), None)

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.std(), None)

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._record_call("unary", lambda ser: ser.var(), None)

    def is_null(self) -> PandasExpression[Bool]:
        return self._record_call("unary", lambda ser: ser.isna(), None)

    def is_nan(self) -> PandasExpression[Bool]:
        def func(ser):
            if is_extension_array_dtype(ser.dtype):
                return np.isnan(ser).replace(pd.NA, False).astype(bool)
            return ser.isna()

        return self._record_call("unary", func, None)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasExpression[Any]:
        def func(ser):
            if ascending:
                return ser.argsort().reset_index(drop=True)
            return ser.argsort()[::-1].reset_index(drop=True)

        return self._record_call("unary", func, None)

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasExpression[Any]:
        return self._record_call(
            "unary",
            lambda ser: ser.sort_values(ascending=ascending).reset_index(drop=True),
            None,
        )

    def is_in(self, values: Expression) -> PandasExpression[Bool]:
        return self._record_call(
            "binary",
            lambda ser, other: ser.isin(other),
            values,
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PandasExpression[Any]:
        return PandasExpression(
            self.column.drop_duplicates().index.to_series(), api_version=self._api_version
        )

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasExpression[DType]:
        def func(ser, value):
            ser = ser.copy()
            ser[
                cast("pd.Series[bool]", np.isnan(ser)).fillna(False).to_numpy(bool)
            ] = value
            return ser

        # return PandasExpression(ser, api_version=self._api_version)
        return self._record_call(
            "unary",
            lambda ser: func(ser, value),
            None,
        )

    def fill_null(
        self,
        value: Any,
    ) -> PandasExpression[DType]:
        def func(ser, value):
            ser = ser.copy()
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
            return ser

        return self._record_call(
            "unary",
            lambda ser: func(ser, value),
            None,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PandasExpression[DType]:
        return self._record_call(
            "unary",
            lambda ser: ser.cumsum(),
            None,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PandasExpression[DType]:
        return self._record_call(
            "unary",
            lambda ser: ser.cumprod(),
            None,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PandasExpression[DType]:
        return self._record_call(
            "unary",
            lambda ser: ser.cummax(),
            None,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PandasExpression[DType]:
        return self._record_call(
            "unary",
            lambda ser: ser.cummin(),
            None,
        )

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy(dtype=dtype)

    def rename(self, name: str | None) -> PandasExpression[DType]:
        expr = self._record_call("unary", lambda ser: ser.rename(name), None)
        return expr


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


LATEST_API_VERSION = "2023.08-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.09-beta"))


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
                f"{SUPPORTED_VERSIONS}. "
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

    def _validate_comparand(self, other: DataFrame) -> None:
        if isinstance(other, PandasDataFrame) and not (
            self.dataframe.index.equals(other.dataframe.index)
            and self.dataframe.shape == other.dataframe.shape
            and self.dataframe.columns.equals(other.dataframe.columns)
        ):
            raise ValueError(
                "Expected DataFrame with same length, matching columns, "
                "and matching index."
            )

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

    def select(self, names: PandasExpression | list[PandasExpression]) -> PandasDataFrame:
        if not isinstance(names, list):
            names = [names]
        return PandasDataFrame(
            pd.concat([self._resolve_expression(name) for name in names], axis=1),
            api_version=self._api_version,
        )

    def get_rows(self, indices: Expression) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[self._resolve_expression(indices), :],
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[start:stop:step], api_version=self._api_version
        )

    def _resolve_expression(self, expression: PandasExpression) -> pd.Series:
        if not isinstance(expression, PandasExpression):
            # e.g. scalar
            return expression
        if not expression._calls:
            return expression._base_call(self.dataframe)
        for kind, func, lhs, rhs in expression._calls:
            lhs = self._resolve_expression(lhs)
            rhs = self._resolve_expression(rhs)
            if kind == "unary":
                if rhs is not None:
                    raise AssertionError("rhs of unary expression is not None")
                expression = func(lhs)
            elif kind == "binary":
                expression = func(lhs, rhs)
            else:
                raise AssertionError(f"expected unary or binary, got: {kind}")
        return expression

    def get_rows_by_mask(self, mask: Expression) -> PandasDataFrame:
        df = self.dataframe
        df = df.loc[self._resolve_expression(mask)]
        return PandasDataFrame(df, api_version=self._api_version)

    def insert_column(self, value: Expression) -> PandasDataFrame:
        # if self._api_version == "2023.08-beta":
        #     raise NotImplementedError(
        #         "DataFrame.insert_column is only available for api versions after 2023.08-beta. "
        #     )
        before = self.dataframe
        to_insert = self._resolve_expression(value)
        return PandasDataFrame(
            pd.concat([before, to_insert], axis=1), api_version=self._api_version
        )

    def update_columns(self, columns: PandasExpression[Any] | Sequence[PandasExpression[Any]], /) -> PandasDataFrame:  # type: ignore[override]
        if self._api_version == "2023.08-beta":
            raise NotImplementedError(
                "DataFrame.insert_column is only available for api versions after 2023.08-beta. "
            )
        if isinstance(columns, PandasExpression):
            columns = [columns]
        df = self.dataframe.copy()
        for col in columns:
            new_column = self._resolve_expression(col)
            if new_column.name not in df.columns:
                raise ValueError(
                    f"column {col.name} not in dataframe, use insert instead"
                )
            df[new_column.name] = new_column
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
    ) -> PandasExpression[Any]:
        if keys is None:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe.loc[:, list(keys)]
        if ascending:
            return PandasExpression(
                df.sort_values(keys).index.to_series().reset_index(drop=True)
            )
        return PandasExpression(
            df.sort_values(keys).index.to_series()[::-1].reset_index(drop=True)
        )

    def sort(
        self,
        keys: Sequence[str] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasDataFrame:
        if self._api_version == "2023.08-beta":
            raise NotImplementedError("dataframe.sort only available after 2023.08-beta")
        if keys is None:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe.loc[:, list(keys)]
        return PandasDataFrame(
            df.sort_values(keys, ascending=ascending), api_version=self._api_version
        )

    def unique_indices(
        self,
        keys: Sequence[str] | None = None,
        *,
        skip_nulls: bool = True,
    ) -> PandasExpression[Any]:
        return PandasExpression(
            self.dataframe.drop_duplicates(subset=keys).index.to_series(),
        )

    def __eq__(self, other: DataFrame | Any) -> PandasDataFrame:  # type: ignore[override]
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__eq__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__eq__(other), api_version=self._api_version
        )

    def __ne__(self, other: DataFrame | Any) -> PandasDataFrame:  # type: ignore[override]
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__ne__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__ne__(other), api_version=self._api_version
        )

    def __ge__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__ge__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__ge__(other), api_version=self._api_version
        )

    def __gt__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__gt__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__gt__(other), api_version=self._api_version
        )

    def __le__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__le__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__le__(other), api_version=self._api_version
        )

    def __lt__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__lt__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__lt__(other), api_version=self._api_version
        )

    def __and__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__and__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__and__(other), api_version=self._api_version
        )

    def __or__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__or__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__or__(other), api_version=self._api_version
        )

    def __add__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__add__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__add__(other), api_version=self._api_version
        )

    def __sub__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__sub__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__sub__(other), api_version=self._api_version
        )

    def __mul__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__mul__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__mul__(other), api_version=self._api_version
        )

    def __truediv__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__truediv__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__truediv__(other), api_version=self._api_version
        )

    def __floordiv__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__floordiv__(other.dataframe),
                api_version=self._api_version,
            )
        return PandasDataFrame(
            self.dataframe.__floordiv__(other), api_version=self._api_version
        )

    def __pow__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__pow__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__pow__(other), api_version=self._api_version
        )

    def __mod__(self, other: DataFrame | Any) -> PandasDataFrame:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            return PandasDataFrame(
                self.dataframe.__mod__(other.dataframe), api_version=self._api_version
            )
        return PandasDataFrame(
            self.dataframe.__mod__(other), api_version=self._api_version
        )

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PandasDataFrame, PandasDataFrame]:
        if isinstance(other, PandasDataFrame):
            self._validate_comparand(other)
            quotient, remainder = self.dataframe.__divmod__(other.dataframe)
        else:
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
