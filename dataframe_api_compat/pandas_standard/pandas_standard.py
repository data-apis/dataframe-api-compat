from __future__ import annotations

import collections
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
        EagerColumn,
        Expression,
        DataFrame,
        EagerFrame,
        GroupBy,
    )
else:

    class DataFrame:
        ...

    class EagerFrame:
        ...

    class EagerColumn(Generic[DType]):
        ...

    class Expression:
        ...

    class GroupBy:
        ...

    class Bool:
        ...


ExtraCall = tuple[
    Callable[[pd.Series, pd.Series | None], pd.Series], pd.Series, pd.Series
]


class PandasExpression(Expression):
    def __init__(
        self,
        root_names: list[str] | None,
        output_name: str,
        base_call: Callable[[pd.DataFrame], pd.Series] | None = None,
        extra_calls: list[ExtraCall] | None = None,
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
        self._base_call = base_call
        self._calls = extra_calls or []
        self._root_names = root_names
        self._output_name = output_name
        # TODO: keep track of output name

    @property
    def root_names(self):
        # todo need to merge these when doing e.g. col('a') + col('b')
        return self._root_names

    @property
    def output_name(self):
        return self._output_name

    def _record_call(
        self,
        func: Callable[[pd.Series, pd.Series | None], pd.Series],
        rhs: pd.Series | None,
        output_name: str | None = None,
    ) -> PandasExpression:
        calls = [*self._calls, (func, self, rhs)]
        if isinstance(rhs, PandasExpression):
            root_names = self.root_names + rhs.root_names
        else:
            root_names = self.root_names
        return PandasExpression(
            root_names=root_names,
            output_name=output_name or self.output_name,
            extra_calls=calls,
        )

    def get_rows(self, indices: Expression | EagerColumn[Any]) -> PandasExpression:
        def func(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
            return lhs.iloc[rhs].reset_index(drop=True)

        return self._record_call(
            func,
            indices,
        )

    def len(self) -> PandasExpression:
        return self._record_call(lambda ser, _rhs: len(ser), None)

    def filter(self, mask: Expression | EagerColumn[Any]) -> PandasExpression:
        return self._record_call(lambda ser, mask: ser.loc[mask], mask)

    def __eq__(self, other: PandasExpression | Any) -> PandasExpression:  # type: ignore[override]
        return self._record_call(
            lambda ser, other: (ser == other).rename(ser.name), other
        )

    def __ne__(self, other: Expression | EagerColumn[Any]) -> PandasExpression:  # type: ignore[override]
        return self._record_call(
            lambda ser, other: (ser != other).rename(ser.name), other
        )

    def __ge__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(
            lambda ser, other: (ser >= other).rename(ser.name), other
        )

    def __gt__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser > other).rename(ser.name), other)

    def __le__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(
            lambda ser, other: (ser <= other).rename(ser.name), other
        )

    def __lt__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser < other).rename(ser.name), other)

    def __and__(self, other: Expression | bool) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser & other).rename(ser.name), other)

    def __or__(self, other: Expression | bool) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser | other).rename(ser.name), other)

    def __add__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(
            lambda ser, other: ((ser + other).rename(ser.name)).rename(ser.name), other
        )

    def __sub__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser - other).rename(ser.name), other)

    def __mul__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser * other).rename(ser.name), other)

    def __truediv__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser / other).rename(ser.name), other)

    def __floordiv__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(
            lambda ser, other: (ser // other).rename(ser.name), other
        )

    def __pow__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(
            lambda ser, other: (ser**other).rename(ser.name), other
        )

    def __mod__(self, other: Expression | Any) -> PandasExpression:
        return self._record_call(lambda ser, other: (ser % other).rename(ser.name), other)

    def __divmod__(
        self, other: Expression | Any
    ) -> tuple[PandasExpression, PandasExpression]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self: PandasExpression) -> PandasExpression:
        return self._record_call(lambda ser, _rhs: ~ser, None)

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.any()], name=ser.name), None
        )

    def all(self, *, skip_nulls: bool = True) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.all()], name=ser.name), None
        )

    def min(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.min()], name=ser.name), None
        )

    def max(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.max()], name=ser.name), None
        )

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.sum()], name=ser.name), None
        )

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.prod()], name=ser.name), None
        )

    def median(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.median()], name=ser.name), None
        )

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.mean()], name=ser.name), None
        )

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.std()], name=ser.name), None
        )

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._record_call(
            lambda ser, _rhs: pd.Series([ser.var()], name=ser.name), None
        )

    def is_null(self) -> PandasExpression:
        return self._record_call(lambda ser, _rhs: ser.isna(), None)

    def is_nan(self) -> PandasExpression:
        def func(ser, _rhs):
            if is_extension_array_dtype(ser.dtype):
                return np.isnan(ser).replace(pd.NA, False).astype(bool)
            return ser.isna()

        return self._record_call(func, None)

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: ser.sort_values(ascending=ascending).reset_index(drop=True),
            None,
        )

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasExpression:
        def func(ser, _rhs):
            if ascending:
                return (
                    ser.sort_values()
                    .index.to_series(name=self.output_name)
                    .reset_index(drop=True)
                )
            return (
                ser.sort_values()
                .index.to_series(name=self.output_name)[::-1]
                .reset_index(drop=True)
            )

        return self._record_call(
            func,
            None,
        )

    def is_in(self, values: Expression | EagerColumn[Any]) -> PandasExpression:
        return self._record_call(
            lambda ser, other: ser.isin(other),
            values,
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PandasExpression:
        raise NotImplementedError("not yet supported")
        return PandasExpression(
            self.column.drop_duplicates().index.to_series(), api_version=self._api_version
        )

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasExpression:
        def func(ser, _rhs):
            ser = ser.copy()
            ser[
                cast("pd.Series[bool]", np.isnan(ser)).fillna(False).to_numpy(bool)
            ] = value
            return ser

        return self._record_call(
            func,
            None,
        )

    def fill_null(
        self,
        value: Any,
    ) -> PandasExpression:
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
            return ser.rename(self.output_name)

        return self._record_call(
            lambda ser, _rhs: func(ser, value),
            None,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: ser.cumsum(),
            None,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: ser.cumprod(),
            None,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: ser.cummax(),
            None,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PandasExpression:
        return self._record_call(
            lambda ser, _rhs: ser.cummin(),
            None,
        )

    def rename(self, name: str) -> PandasExpression:
        expr = self._record_call(
            lambda ser, _rhs: ser.rename(name), None, output_name=name
        )
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


class PandasColumn(EagerColumn[DType]):
    # private, not technically part of the standard
    def __init__(self, column: pd.Series[Any], api_version: str) -> None:
        self._name = column.name
        self._series = column.reset_index(drop=True)
        self._api_version = api_version
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )

    def to_expression(self) -> PandasExpression:
        return PandasExpression(
            root_names=[],
            output_name=self.name,
            base_call=lambda _df: self.column.rename(self.name),
        )

    def _reuse_expression_implementation(self, function_name, *args, **kwargs):
        return (
            PandasDataFrame(pd.DataFrame(), api_version=self._api_version)
            .select(getattr(self.to_expression(), function_name)(*args, **kwargs))
            .collect()
            .get_column(self.name)
        )

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def name(self) -> str:
        return self._name

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

    def get_rows(self, indices: EagerColumn[Any]) -> PandasColumn[DType]:
        return self._reuse_expression_implementation("get_rows", indices)

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

    def filter(self, mask: Expression | EagerColumn[Any]) -> PandasColumn[DType]:
        return self._reuse_expression_implementation("filter", mask)

    def get_value(self, row: int) -> Any:
        return self.column.iloc[row]

    def __eq__(  # type: ignore[override]
        self, other: PandasColumn[DType] | Any
    ) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__eq__", other)

    def __ne__(  # type: ignore[override]
        self, other: EagerColumn[DType]
    ) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__ne__", other)

    def __ge__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__ge__", other)

    def __gt__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__gt__", other)

    def __le__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__le__", other)

    def __lt__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__lt__", other)

    def __and__(self, other: EagerColumn[Bool] | bool) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__and__", other)

    def __or__(self, other: EagerColumn[Bool] | bool) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__or__", other)

    def __add__(self, other: EagerColumn[DType] | Any) -> PandasColumn[DType]:
        return self._reuse_expression_implementation("__add__", other)

    def __sub__(self, other: EagerColumn[DType] | Any) -> PandasColumn[DType]:
        return self._reuse_expression_implementation("__sub__", other)

    def __mul__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Any]:
        return self._reuse_expression_implementation("__mul__", other)

    def __truediv__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Any]:
        return self._reuse_expression_implementation("__truediv__", other)

    def __floordiv__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Any]:
        return self._reuse_expression_implementation("__floordiv__", other)

    def __pow__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Any]:
        return self._reuse_expression_implementation("__pow__", other)

    def __mod__(self, other: EagerColumn[DType] | Any) -> PandasColumn[Any]:
        return self._reuse_expression_implementation("__mod__", other)

    def __divmod__(
        self, other: EagerColumn[DType] | Any
    ) -> tuple[PandasColumn[Any], PandasColumn[Any]]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self: PandasColumn[Bool]) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("__invert__")

    # Reductions
    # Can't reuse the expressions implementation here as these return scalars.

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

    # Transformations, defer to expressions impl

    def is_null(self) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("is_null")

    def is_nan(self) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("is_nan")

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasColumn[Any]:
        return self._reuse_expression_implementation(
            "sorted_indices", ascending=ascending, nulls_position=nulls_position
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PandasColumn[Any]:
        return self._reuse_expression_implementation(
            "sort", ascending=ascending, nulls_position=nulls_position
        )

    def is_in(self, values: EagerColumn[DType]) -> PandasColumn[Bool]:
        return self._reuse_expression_implementation("is_in", values)

    def unique_indices(self, *, skip_nulls: bool = True) -> PandasColumn[Any]:
        raise NotImplementedError("not yet supported")

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasColumn[DType]:
        return self._reuse_expression_implementation("fill_nan", value)

    def fill_null(
        self,
        value: Any,
    ) -> PandasColumn[DType]:
        return self._reuse_expression_implementation("fill_null", value)

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return self._reuse_expression_implementation(
            "cumulative_sum", skip_nulls=skip_nulls
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return self._reuse_expression_implementation(
            "cumulative_prod", skip_nulls=skip_nulls
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return self._reuse_expression_implementation(
            "cumulative_max", skip_nulls=skip_nulls
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PandasColumn[DType]:
        return self._reuse_expression_implementation(
            "cumulative_min", skip_nulls=skip_nulls
        )

    def rename(self, name: str) -> PandasColumn[DType]:
        self._name = name
        return self._reuse_expression_implementation("rename", name=name)

    # Eager-only

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy(dtype=dtype)


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

    def __repr__(self) -> str:
        return self.dataframe.__repr__()

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

    # In the standard
    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def groupby(self, *keys: str) -> PandasGroupBy:
        if not isinstance(keys, collections.abc.Sequence):
            raise TypeError(f"Expected sequence of strings, got: {type(keys)}")
        if isinstance(keys, str):
            raise TypeError("Expected sequence of strings, got: str")
        for key in keys:
            if key not in self.get_column_names():
                raise KeyError(f"key {key} not present in DataFrame's columns")
        return PandasGroupBy(self.dataframe, keys, api_version=self._api_version)

    def select(self, *columns: str | Expression | EagerColumn[Any]) -> PandasDataFrame:
        new_columns = []
        lengths = []
        for name in columns:
            if isinstance(name, str):
                new_columns.append(self.dataframe.loc[:, name])
            else:
                new_columns.append(self._resolve_expression(name))
            lengths.append(len(new_columns[-1]))
        if len(set(lengths)) > 1:
            # need to broadcast
            max_len = max(lengths)
            for i, length in enumerate(lengths):
                if length == 1:
                    new_columns[i] = pd.Series(
                        [new_columns[i][0]] * max_len, name=new_columns[i].name
                    )
        return PandasDataFrame(
            pd.concat(new_columns, axis=1),
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

    def _broadcast(self, lhs, rhs):
        if (
            isinstance(lhs, pd.Series)
            and isinstance(rhs, pd.Series)
            and len(lhs) != 1
            and len(rhs) == 1
        ):
            rhs = pd.Series([rhs[0]] * len(lhs), name=rhs.name)
        elif (
            isinstance(lhs, pd.Series)
            and isinstance(rhs, pd.Series)
            and len(lhs) == 1
            and len(rhs) != 1
        ):
            lhs = pd.Series([lhs[0]] * len(rhs), name=lhs.name)
        return lhs, rhs

    def _resolve_expression(
        self, expression: PandasExpression | PandasColumn | pd.Series | object, *, level=0
    ) -> pd.Series:
        if isinstance(expression, PandasColumn):
            return expression.column
        if not isinstance(expression, PandasExpression):
            # e.g. scalar
            return expression
        if not expression._calls:
            return expression._base_call(self.dataframe)
        output_name = expression.output_name
        for func, lhs, rhs in expression._calls:
            lhs = self._resolve_expression(lhs, level=level + 1)
            rhs = self._resolve_expression(rhs, level=level + 1)
            lhs, rhs = self._broadcast(lhs, rhs)
            expression = func(lhs, rhs)
        if isinstance(expression, pd.Series):
            assert output_name == expression.name, f"{output_name} != {expression.name}"
        return expression

    def filter(self, mask: Expression | EagerColumn[Any]) -> PandasDataFrame:
        df = self.dataframe
        df = df.loc[self._resolve_expression(mask)]
        return PandasDataFrame(df, api_version=self._api_version)

    def insert_columns(self, *columns: Expression) -> PandasDataFrame:
        new_columns = pd.concat(
            [self._resolve_expression(column) for column in columns], axis=1
        )
        if (len(new_columns) == 1) & (len(self.dataframe) > 1):
            new_columns = pd.DataFrame(
                {
                    col_name: [new_columns[col_name][0]] * len(self.dataframe)
                    for col_name in new_columns.columns
                }
            )
        return PandasDataFrame(
            pd.concat([self.dataframe, new_columns], axis=1),
            api_version=self._api_version,
        )

    def update_columns(self, *columns: Expression | EagerColumn[Any]) -> PandasDataFrame:
        df = self.dataframe.copy()
        for col in columns:
            new_column = self._resolve_expression(col)
            if new_column.name not in df.columns:
                raise ValueError(
                    f"column {new_column.name} not in dataframe, use insert instead"
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

    def sort(
        self,
        *keys: str | Expression | EagerColumn[Any],
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
    ) -> PandasEagerFrame:
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

    def collect(self) -> PandasEagerFrame:
        return PandasEagerFrame(self.dataframe, api_version=self._api_version)


class PandasEagerFrame(EagerFrame):
    # Not technically part of the standard

    def __init__(self, dataframe: pd.DataFrame, api_version: str) -> None:
        # note: less validation is needed here, as the validation will already
        # have happened in DataFrame, and EagerFrame can only be created from that.
        self._dataframe = dataframe.reset_index(drop=True)
        self._api_version = api_version

    def __repr__(self) -> str:
        return self.dataframe.__repr__()

    def _reuse_dataframe_implementation(self, function_name, *args, **kwargs):
        return getattr(self.relax(), function_name)(*args, **kwargs).collect()

    # In the standard
    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.pandas_standard

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def groupby(self, *keys: str) -> PandasGroupBy:
        if not isinstance(keys, collections.abc.Sequence):
            raise TypeError(f"Expected sequence of strings, got: {type(keys)}")
        if isinstance(keys, str):
            raise TypeError("Expected sequence of strings, got: str")
        for key in keys:
            if key not in self.get_column_names():
                raise KeyError(f"key {key} not present in DataFrame's columns")
        return PandasGroupBy(self.dataframe, keys, api_version=self._api_version)

    def select(self, *columns: str | Expression | EagerColumn[Any]) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("select", *columns)

    def get_column(self, name) -> PandasColumn:
        return PandasColumn(self.dataframe.loc[:, name], api_version=self._api_version)

    def get_rows(self, indices: Expression | EagerColumn) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("get_rows", indices)

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation(
            "slice_rows", start=start, stop=stop, step=step
        )

    def filter(self, mask: Expression | EagerColumn) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("filter", mask)

    def insert_columns(self, *columns: Expression | EagerColumn[Any]) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("insert_columns", *columns)

    def update_columns(self, *columns: Expression | EagerColumn[Any]) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("update_columns", *columns)

    def drop_column(self, label: str) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("drop_column", label=label)

    def rename_columns(self, mapping: Mapping[str, str]) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("rename_columns", mapping=mapping)

    def get_column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()

    def sort(
        self,
        *keys: str | Expression | EagerColumn,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation(
            "sort", *keys, ascending=ascending, nulls_position=nulls_position
        )

    def __eq__(self, other: Any) -> PandasEagerFrame:  # type: ignore[override]
        return self._reuse_dataframe_implementation("__eq__", other)

    def __ne__(self, other: Any) -> PandasEagerFrame:  # type: ignore[override]
        return self._reuse_dataframe_implementation("__ne__", other)

    def __ge__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__ge__", other)

    def __gt__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__gt__", other)

    def __le__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__le__", other)

    def __lt__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__lt__", other)

    def __and__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__and__", other)

    def __or__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__or__", other)

    def __add__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__add__", other)

    def __sub__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__sub__", other)

    def __mul__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__mul__", other)

    def __truediv__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__truediv__", other)

    def __floordiv__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__floordiv__", other)

    def __pow__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__pow__", other)

    def __mod__(self, other: Any) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__mod__", other)

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PandasEagerFrame, PandasEagerFrame]:
        quotient, remainder = self.dataframe.__divmod__(other)
        return PandasEagerFrame(
            quotient, api_version=self._api_version
        ), PandasEagerFrame(remainder, api_version=self._api_version)

    def __invert__(self) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("__invert__")

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def any(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("any", skip_nulls=skip_nulls)

    def all(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("all", skip_nulls=skip_nulls)

    def min(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("min", skip_nulls=skip_nulls)

    def max(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("max", skip_nulls=skip_nulls)

    def sum(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("sum", skip_nulls=skip_nulls)

    def prod(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("prod", skip_nulls=skip_nulls)

    def median(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("median", skip_nulls=skip_nulls)

    def mean(self, *, skip_nulls: bool = True) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("mean", skip_nulls=skip_nulls)

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation(
            "std", correction=correction, skip_nulls=skip_nulls
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation(
            "var", correction=correction, skip_nulls=skip_nulls
        )

    def is_null(self) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("is_null")

    def is_nan(self) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("is_nan")

    def fill_nan(
        self, value: float | pd.NAType  # type: ignore[name-defined]
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("fill_nan", value)

    def fill_null(
        self,
        value: Any,
        *,
        column_names: list[str] | None = None,
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation("fill_null", value)

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.dataframe.to_numpy(dtype=dtype)

    def join(
        self,
        other: EagerFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> PandasEagerFrame:
        return self._reuse_dataframe_implementation(
            "join",
            other=other.relax(),
            left_on=left_on,
            right_on=right_on,
            how=how,
        )

    def relax(self) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe, api_version=self._api_version)
