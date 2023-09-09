from __future__ import annotations

import collections
from typing import Any
from typing import Generic
from typing import Literal
from typing import NoReturn
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar

import polars as pl

import dataframe_api_compat.polars_standard

# do we need a separate class for polars lazy?
# might be best - after all, we can't mix lazy
# and eager
# BUT most things will probably work the same way?

col = None

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
DType = TypeVar("DType")

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dataframe_api import (
        Bool,
        DataFrame,
        Column,
        GroupBy,
    )
else:

    class DataFrame:
        ...

    class Column(Generic[DType]):
        ...

    class Expression:
        ...

    class GroupBy:
        ...

    class Bool:
        ...


class Null:
    ...


null = Null()
NullType = Type[Null]


def _is_integer_dtype(dtype: Any) -> bool:
    return any(
        dtype is _dtype
        for _dtype in (
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.UInt16,
            pl.UInt8,
        )
    )


LATEST_API_VERSION = "2023.08-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.09-beta"))


class PolarsColumn(Column[DType]):
    def __init__(
        self,
        column: pl.Series,
        *,
        api_version: str,
    ) -> None:
        if column is NotImplemented:
            raise NotImplementedError("operation not implemented")
        self._series = column
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version
        self._dtype = column.dtype

    def _validate_column(self, column: PolarsColumn[Any] | Column[Any]) -> None:
        assert isinstance(column, PolarsColumn)
        if isinstance(column.column, pl.Expr) and column._id != self._id:
            raise ValueError(
                "Column was created from a different dataframe!",
                column._id,
                self._id,
            )

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def name(self) -> str:
        if isinstance(self.column, pl.Series):
            return self.column.name
        name = self.column.meta.output_name()
        return name

    @property
    def column(self) -> pl.Series | pl.Expr:
        return self._series

    def __len__(self) -> int:
        if isinstance(self.column, pl.Series):
            return len(self.column)
        raise NotImplementedError(
            "__len__ intentionally not implemented for lazy columns"
        )

    @property
    def dtype(self) -> Any:
        return dataframe_api_compat.polars_standard.DTYPE_MAP[self._dtype]

    def get_rows(self, indices: Column[Any]) -> PolarsColumn[DType]:
        return PolarsColumn(
            self.column.take(indices.column),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsColumn[DType]:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("slice_rows not implemented for lazy columns")
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.column)
        if step is None:
            step = 1
        return PolarsColumn(
            self.column[start:stop:step],
            api_version=self._api_version,
        )

    def filter(self, mask: PolarsColumn[Any]) -> PolarsColumn[DType]:
        self._validate_column(mask)
        return PolarsColumn(
            self.column.filter(mask.column),
            api_version=self._api_version,
        )

    def get_value(self, row: int) -> Any:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("get_value not implemented for lazy columns")
        return self.column[row]

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsColumn[DType]) -> PolarsColumn[Bool]:  # type: ignore[override]
        self._validate_column(values)
        if values.dtype != self.dtype:
            raise ValueError(f"`value` has dtype {values.dtype}, expected {self.dtype}")
        return PolarsColumn(
            self.column.is_in(values.column), api_version=self._api_version  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn[Any]:
        raise NotImplementedError("not yet supported")
        df = self.column.to_frame()
        keys = df.columns
        return PolarsColumn(
            df.with_row_count().unique(keys).get_column("row_nr"),
            api_version=self._api_version,
        )

    def is_null(self) -> PolarsColumn[Bool]:
        return PolarsColumn(
            self.column.is_null(),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsColumn[Bool]:
        return PolarsColumn(
            self.column.is_nan(),
            api_version=self._api_version,
        )

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("any not implemented for lazy columns")
        return self.column.any()

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("all not implemented for lazy columns")
        return self.column.all()

    def min(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.min(),
                api_version=self._api_version,
            )
        return self.column.min()

    def max(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.max(),
                api_version=self._api_version,
            )
        return self.column.max()

    def sum(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.sum(),
                api_version=self._api_version,
            )
        return self.column.sum()

    def prod(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.product(),
                api_version=self._api_version,
            )
        return self.column.product()

    def mean(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.mean(),
                api_version=self._api_version,
            )
        return self.column.mean()

    def median(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.median(),
                api_version=self._api_version,
            )
        return self.column.median()

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.std(),
                api_version=self._api_version,
            )
        return self.column.std()

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            return PolarsColumn(
                self.column.var(),
                api_version=self._api_version,
            )
        return self.column.var()

    def __eq__(  # type: ignore[override]
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column == other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column == other,
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column != other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column != other,
            api_version=self._api_version,
        )

    def __ge__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column >= other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column >= other,
            api_version=self._api_version,
        )

    def __gt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column > other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column > other,
            api_version=self._api_version,
        )

    def __le__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column <= other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column <= other,
            api_version=self._api_version,
        )

    def __lt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column < other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column < other,
            api_version=self._api_version,
        )

    def __mul__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            res = self.column * other.column
            return PolarsColumn(res, api_version=self._api_version)
        res = self.column * other
        return PolarsColumn(res, api_version=self._api_version)

    def __floordiv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column // other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column // other,
            api_version=self._api_version,
        )

    def __truediv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res = self.column / other.column
            return PolarsColumn(res, api_version=self._api_version)
        res = self.column / other
        return PolarsColumn(res, api_version=self._api_version)

    def __pow__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            ret = self.column**other.column  # type: ignore[operator]
        else:
            ret = self.column.pow(other)  # type: ignore[arg-type]
        return PolarsColumn(ret, api_version=self._api_version)

    def __mod__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column % other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column % other,
            api_version=self._api_version,
        )

    def __divmod__(
        self,
        other: Column[DType] | Any,
    ) -> tuple[PolarsColumn[Any], PolarsColumn[Any]]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: Column[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column & other.column, api_version=self._api_version  # type: ignore[operator]
            )
        return PolarsColumn(self.column & other, api_version=self._api_version)  # type: ignore[operator]

    def __or__(self, other: Column[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column | other.column, api_version=self._api_version  # type: ignore[operator]
            )
        return PolarsColumn(self.column | other, api_version=self._api_version)  # type: ignore[operator]

    def __invert__(self) -> PolarsColumn[Bool]:
        return PolarsColumn(~self.column, api_version=self._api_version)

    def __add__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column + other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column + other,
            api_version=self._api_version,
        )

    def __sub__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column - other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column - other,
            api_version=self._api_version,
        )

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn[Any]:
        expr = self.column.arg_sort(descending=not ascending)
        return PolarsColumn(
            expr,
            api_version=self._api_version,
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn[Any]:
        expr = self.column.sort(descending=not ascending)
        return PolarsColumn(
            expr,
            api_version=self._api_version,
        )

    def fill_nan(self, value: float | NullType) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsColumn[DType]:
        return PolarsColumn(
            self.column.fill_null(value),
            api_version=self._api_version,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(
            self.column.cumsum(),
            api_version=self._api_version,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(
            self.column.cumprod(),
            api_version=self._api_version,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(
            self.column.cummax(),
            api_version=self._api_version,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(
            self.column.cummin(),
            api_version=self._api_version,
        )

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy().astype(dtype)

    def rename(self, name: str) -> PolarsColumn[DType]:
        if isinstance(self.column, pl.Series):
            return PolarsColumn(
                self.column.rename(name),
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column.alias(name),
            api_version=self._api_version,
        )


class PolarsGroupBy(GroupBy):
    def __init__(self, df: pl.LazyFrame, keys: Sequence[str], api_version: str) -> None:
        for key in keys:
            if key not in df.columns:
                raise KeyError(f"key {key} not present in DataFrame's columns")
        self.df = df
        self.keys = keys
        self._api_version = api_version
        self.group_by = self.df.groupby if pl.__version__ < "0.19.0" else self.df.group_by

    def size(self) -> PolarsDataFrame:
        result = self.group_by(self.keys).count().rename({"count": "size"})
        return PolarsDataFrame(result, api_version=self._api_version)

    def any(self, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.group_by(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").any())
        return PolarsDataFrame(result, api_version=self._api_version)

    def all(self, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.group_by(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").all())
        return PolarsDataFrame(result, api_version=self._api_version)

    def min(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result, api_version=self._api_version)

    def max(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result, api_version=self._api_version)

    def sum(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result, api_version=self._api_version)

    def prod(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result, api_version=self._api_version)

    def median(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result, api_version=self._api_version)

    def mean(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result, api_version=self._api_version)

    def std(
        self, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result, api_version=self._api_version)

    def var(
        self, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result, api_version=self._api_version)


class PolarsExpression:
    def __init__(
        self,
        expr: pl.Series | pl.Expr,
        *,
        api_version: str | None = None,
    ) -> None:
        if expr is NotImplemented:
            raise NotImplementedError("operation not implemented")
        if isinstance(expr, str):
            self.column = pl.col(expr)
        else:
            self.column = expr
        # need to pass this down from namespace.col
        self._api_version = api_version or LATEST_API_VERSION

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def name(self) -> str:
        if isinstance(self.column, pl.Series):
            # TODO: can we avoid this completely?
            name = self.column.name
        else:
            name = self.column.meta.output_name()
        return name

    def get_rows(self, indices: PolarsExpression) -> PolarsExpression:
        return PolarsExpression(
            self.column.take(indices.column),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsExpression:
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.column)
        if step is None:
            step = 1
        return PolarsExpression(
            self.column[start:stop:step],
            api_version=self._api_version,
        )

    def filter(self, mask: PolarsExpression) -> PolarsExpression:
        return PolarsExpression(
            self.column.filter(mask.column), api_version=self._api_version  # type: ignore[arg-type]
        )

    def get_value(self, row: int) -> Any:
        return self.Expression

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsExpression) -> PolarsExpression:  # type: ignore[override]
        return PolarsExpression(
            self.column.is_in(values.column), api_version=self._api_version  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsExpression:
        raise NotImplementedError()

    def is_null(self) -> PolarsExpression:
        return PolarsExpression(
            self.column.is_null(),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsExpression:
        return PolarsExpression(
            self.column.is_nan(),
            api_version=self._api_version,
        )

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        return self.column.any()

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        return self.column.all()

    def min(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.min(),
            api_version=self._api_version,
        )

    def max(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.max(),
            api_version=self._api_version,
        )

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.sum(),
            api_version=self._api_version,
        )

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.product(),
            api_version=self._api_version,
        )

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.mean(),
            api_version=self._api_version,
        )

    def median(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.median(),
            api_version=self._api_version,
        )

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.std(),
            api_version=self._api_version,
        )

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self.column.var(),
            api_version=self._api_version,
        )

    def __eq__(  # type: ignore[override]
        self, other: PolarsExpression | Any
    ) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column == other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column == other,
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self, other: PolarsExpression | Any
    ) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column != other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column != other,
            api_version=self._api_version,
        )

    def __ge__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column >= other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column >= other,
            api_version=self._api_version,
        )

    def __gt__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column > other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column > other,
            api_version=self._api_version,
        )

    def __le__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column <= other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column <= other,
            api_version=self._api_version,
        )

    def __lt__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column < other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column < other,
            api_version=self._api_version,
        )

    def __mul__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res = self.column * other.column
            return PolarsExpression(res, api_version=self._api_version)
        res = self.column * other
        return PolarsExpression(res, api_version=self._api_version)

    def __floordiv__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column // other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column // other,
            api_version=self._api_version,
        )

    def __truediv__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res = self.column / other.column
            return PolarsExpression(res, api_version=self._api_version)
        res = self.column / other
        return PolarsExpression(res, api_version=self._api_version)

    def __pow__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            ret = self.column**other.column  # type: ignore[operator]
        else:
            ret = self.column.pow(other)  # type: ignore[arg-type]
        return PolarsExpression(ret, api_version=self._api_version)

    def __mod__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column % other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column % other,
            api_version=self._api_version,
        )

    def __divmod__(
        self,
        other: PolarsExpression | Any,
    ) -> tuple[PolarsExpression, PolarsExpression]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: PolarsExpression | bool) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(self.column & other.column)
        return PolarsExpression(self.column & other)  # type: ignore[operator]

    def __or__(self, other: PolarsExpression | bool) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(self.column | other.column)
        return PolarsExpression(self.column | other)

    def __invert__(self) -> PolarsExpression:
        return PolarsExpression(~self.column, api_version=self._api_version)

    def __add__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column + other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column + other,
            api_version=self._api_version,
        )

    def __sub__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self.column - other.column,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.column - other,
            api_version=self._api_version,
        )

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsExpression:
        expr = self.column.arg_sort(descending=not ascending)
        return PolarsExpression(
            expr,
            api_version=self._api_version,
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsExpression:
        expr = self.column.sort(descending=not ascending)
        return PolarsExpression(
            expr,
            api_version=self._api_version,
        )

    def fill_nan(self, value: float | NullType) -> PolarsExpression:
        return PolarsExpression(self.column.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsExpression:
        return PolarsExpression(
            self.column.fill_null(value),
            api_version=self._api_version,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self.column.cumsum(),
            api_version=self._api_version,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self.column.cumprod(),
            api_version=self._api_version,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self.column.cummax(),
            api_version=self._api_version,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self.column.cummin(),
            api_version=self._api_version,
        )

    def rename(self, name: str) -> PolarsExpression:
        return PolarsExpression(
            self.column.alias(name),
            api_version=self._api_version,
        )


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.LazyFrame, api_version: str) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        if df is NotImplemented:
            raise NotImplementedError("operation not implemented")
        self.df = df
        self._id = id(df)
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def dataframe(self) -> pl.LazyFrame:
        return self.df

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys, api_version=self._api_version)

    def select(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names), api_version=self._api_version)

    def get_rows(self, indices: PolarsColumn[Any]) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices.column)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self._api_version)

    def filter(self, mask: PolarsExpression) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask.column), api_version=self._api_version)

    def insert(self, loc: int, label: str, value: PolarsExpression) -> PolarsDataFrame:
        if self._api_version != "2023.08-beta":
            raise NotImplementedError(
                "DataFrame.insert is only available for api version 2023.08-beta. "
                "Please use `DataFrame.insert_column` instead."
            )
        columns = self.dataframe.columns
        new_columns = columns[:loc] + [label] + columns[loc:]
        df = self.dataframe.with_columns(value.column.alias(label)).select(new_columns)
        return PolarsDataFrame(df, api_version=self._api_version)

    def insert_column(self, value: PolarsExpression) -> PolarsDataFrame:
        # if self._api_version == "2023.08-beta":
        #     raise NotImplementedError(
        #         "DataFrame.insert is only available for api version 2023.08-beta. "
        #         "Please use `DataFrame.insert_column` instead."
        #     )
        columns = self.dataframe.columns
        label = value.name
        new_columns = [*columns, label]
        df = self.dataframe.with_columns(value.column).select(new_columns)
        return PolarsDataFrame(df, api_version=self._api_version)

    def update_columns(self, columns: PolarsExpression | Sequence[PolarsExpression], /) -> PolarsDataFrame:  # type: ignore[override]
        if self._api_version == "2023.08-beta":
            raise NotImplementedError(
                "DataFrame.update_columns is only available for api version 2023.08-beta. "
                "Please use `DataFrame.insert_column` instead."
            )
        if isinstance(columns, PolarsExpression):
            columns = [columns]
        for col in columns:
            if col.name not in self.dataframe.columns:
                raise ValueError(
                    f"column {col.name} not in dataframe, please use insert_column instead"
                )
        return PolarsDataFrame(
            self.dataframe.with_columns([col.column for col in columns]),
            api_version=self._api_version,
        )

    def drop_column(self, label: str) -> PolarsDataFrame:
        if not isinstance(label, str):
            raise TypeError(f"Expected str, got: {type(label)}")
        return PolarsDataFrame(self.dataframe.drop(label), api_version=self._api_version)

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsDataFrame(
            self.dataframe.rename(dict(mapping)), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:
        return self.dataframe.columns

    def __eq__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__eq__(other)),
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
            api_version=self._api_version,
        )

    def __ge__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
            api_version=self._api_version,
        )

    def __gt__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
            api_version=self._api_version,
        )

    def __le__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
            api_version=self._api_version,
        )

    def __lt__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
            api_version=self._api_version,
        )

    def __and__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") & other),
            api_version=self._api_version,
        )

    def __or__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self._api_version,
        )

    def __add__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self._api_version,
        )

    def __sub__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self._api_version,
        )

    def __mul__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self._api_version,
        )

    def __truediv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self._api_version,
        )

    def __floordiv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__floordiv__(other)),
            api_version=self._api_version,
        )

    def __pow__(self, other: Any) -> PolarsDataFrame:
        original_type = self.dataframe.schema
        ret = self.dataframe.select(
            [pl.col(col).pow(other) for col in self.get_column_names()]
        )
        for column in self.dataframe.columns:
            if _is_integer_dtype(original_type[column]) and isinstance(other, int):
                if other < 0:  # pragma: no cover (todo)
                    raise ValueError("Cannot raise integer to negative power")
                ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        return PolarsDataFrame(ret, api_version=self._api_version)

    def __mod__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") % other),
            api_version=self._api_version,
        )

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PolarsDataFrame, PolarsDataFrame]:
        quotient_df = self.dataframe.with_columns(pl.col("*") // other)
        remainder_df = self.dataframe.with_columns(
            pl.col("*") - (pl.col("*") // other) * other
        )
        return PolarsDataFrame(
            quotient_df, api_version=self._api_version
        ), PolarsDataFrame(remainder_df, api_version=self._api_version)

    def __invert__(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(~pl.col("*")), api_version=self._api_version
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").is_null()),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsDataFrame(df, api_version=self._api_version)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").any()), api_version=self._api_version
        )

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").all()), api_version=self._api_version
        )

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").min()), api_version=self._api_version
        )

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").max()), api_version=self._api_version
        )

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").sum()), api_version=self._api_version
        )

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").product()), api_version=self._api_version
        )

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").mean()), api_version=self._api_version
        )

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").median()), api_version=self._api_version
        )

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").std()), api_version=self._api_version
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").var()), api_version=self._api_version
        )

    def sort(
        self,
        keys: Sequence[Any] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsDataFrame:
        if keys is None:
            keys = self.dataframe.columns
        # TODO: what if there's multiple `ascending`?
        return PolarsDataFrame(
            self.dataframe.sort(keys, descending=not ascending),
            api_version=self._api_version,
        )

    def fill_nan(
        self,
        value: float | NullType,
    ) -> PolarsDataFrame:
        if isinstance(value, Null):
            value = None
        return PolarsDataFrame(self.dataframe.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(
        self,
        value: Any,
        *,
        column_names: list[str] | None = None,
    ) -> PolarsDataFrame:
        if column_names is None:
            column_names = self.dataframe.columns
        df = self.dataframe.with_columns(
            pl.col(col).fill_null(value) for col in column_names
        )
        return PolarsDataFrame(df, api_version=self._api_version)

    def join(
        self,
        other: DataFrame,
        left_on: str | list[str],
        right_on: str | list[str],
        how: Literal["left", "inner", "outer"],
    ) -> PolarsDataFrame:
        if how not in ["left", "inner", "outer"]:
            raise ValueError(f"Expected 'left', 'inner', 'outer', got: {how}")

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        # need to do some extra work to preserve all names
        # https://github.com/pola-rs/polars/issues/9335
        extra_right_keys = set(right_on).difference(left_on)
        assert isinstance(other, PolarsDataFrame)
        other_df = other.dataframe
        # todo: make more robust
        other_df = other_df.with_columns(
            [pl.col(i).alias(f"{i}_tmp") for i in extra_right_keys]
        )
        result = self.dataframe.join(
            other_df, left_on=left_on, right_on=right_on, how=how
        )
        result = result.rename({f"{i}_tmp": i for i in extra_right_keys})

        return PolarsDataFrame(result, api_version=self._api_version)

    def collect(self) -> PolarsEagerFrame:
        return PolarsEagerFrame(self.dataframe.collect(), api_version=self._api_version)


class PolarsEagerFrame(DataFrame):
    def __init__(self, df: pl.LazyFrame, api_version: str) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        if df is NotImplemented:
            raise NotImplementedError("operation not implemented")
        self.df = df
        self._id = id(df)
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def dataframe(self) -> pl.LazyFrame:
        return self.df

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys, api_version=self._api_version)

    def select(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names), api_version=self._api_version)

    def get_column_by_name(self, name) -> PolarsColumn:
        return PolarsColumn(
            self.dataframe.get_column(name), api_version=self._api_version
        )

    def get_rows(self, indices: PolarsColumn[Any]) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices.column)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self._api_version)

    def filter(self, mask: PolarsExpression) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask.column), api_version=self._api_version)

    def insert(self, loc: int, label: str, value: PolarsExpression) -> PolarsDataFrame:
        if self._api_version != "2023.08-beta":
            raise NotImplementedError(
                "DataFrame.insert is only available for api version 2023.08-beta. "
                "Please use `DataFrame.insert_column` instead."
            )
        columns = self.dataframe.columns
        new_columns = columns[:loc] + [label] + columns[loc:]
        df = self.dataframe.with_columns(value.column.alias(label)).select(new_columns)
        return PolarsDataFrame(df, api_version=self._api_version)

    def insert_column(self, value: PolarsExpression) -> PolarsDataFrame:
        # if self._api_version == "2023.08-beta":
        #     raise NotImplementedError(
        #         "DataFrame.insert is only available for api version 2023.08-beta. "
        #         "Please use `DataFrame.insert_column` instead."
        #     )
        columns = self.dataframe.columns
        label = value.name
        new_columns = [*columns, label]
        df = self.dataframe.with_columns(value.column).select(new_columns)
        return PolarsDataFrame(df, api_version=self._api_version)

    def update_columns(self, columns: PolarsExpression | Sequence[PolarsExpression], /) -> PolarsDataFrame:  # type: ignore[override]
        if self._api_version == "2023.08-beta":
            raise NotImplementedError(
                "DataFrame.update_columns is only available for api version 2023.08-beta. "
                "Please use `DataFrame.insert_column` instead."
            )
        if isinstance(columns, PolarsExpression):
            columns = [columns]
        for col in columns:
            if col.name not in self.dataframe.columns:
                raise ValueError(
                    f"column {col.name} not in dataframe, please use insert_column instead"
                )
        return PolarsDataFrame(
            self.dataframe.with_columns([col.column for col in columns]),
            api_version=self._api_version,
        )

    def drop_column(self, label: str) -> PolarsDataFrame:
        if not isinstance(label, str):
            raise TypeError(f"Expected str, got: {type(label)}")
        return PolarsDataFrame(self.dataframe.drop(label), api_version=self._api_version)

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsDataFrame(
            self.dataframe.rename(dict(mapping)), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:
        return self.dataframe.columns

    def __eq__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__eq__(other)),
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
            api_version=self._api_version,
        )

    def __ge__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
            api_version=self._api_version,
        )

    def __gt__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
            api_version=self._api_version,
        )

    def __le__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
            api_version=self._api_version,
        )

    def __lt__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
            api_version=self._api_version,
        )

    def __and__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") & other),
            api_version=self._api_version,
        )

    def __or__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self._api_version,
        )

    def __add__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self._api_version,
        )

    def __sub__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self._api_version,
        )

    def __mul__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self._api_version,
        )

    def __truediv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self._api_version,
        )

    def __floordiv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__floordiv__(other)),
            api_version=self._api_version,
        )

    def __pow__(self, other: Any) -> PolarsDataFrame:
        original_type = self.dataframe.schema
        ret = self.dataframe.select(
            [pl.col(col).pow(other) for col in self.get_column_names()]
        )
        for column in self.dataframe.columns:
            if _is_integer_dtype(original_type[column]) and isinstance(other, int):
                if other < 0:  # pragma: no cover (todo)
                    raise ValueError("Cannot raise integer to negative power")
                ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        return PolarsDataFrame(ret, api_version=self._api_version)

    def __mod__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") % other),
            api_version=self._api_version,
        )

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PolarsDataFrame, PolarsDataFrame]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(~pl.col("*")), api_version=self._api_version
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").is_null()),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsDataFrame(df, api_version=self._api_version)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").any()), api_version=self._api_version
        )

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").all()), api_version=self._api_version
        )

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").min()), api_version=self._api_version
        )

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").max()), api_version=self._api_version
        )

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").sum()), api_version=self._api_version
        )

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").product()), api_version=self._api_version
        )

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").mean()), api_version=self._api_version
        )

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").median()), api_version=self._api_version
        )

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").std()), api_version=self._api_version
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").var()), api_version=self._api_version
        )

    def sort(
        self,
        keys: Sequence[Any] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsDataFrame:
        if keys is None:
            keys = self.dataframe.columns
        # TODO: what if there's multiple `ascending`?
        return PolarsDataFrame(
            self.dataframe.sort(keys, descending=not ascending),
            api_version=self._api_version,
        )

    def fill_nan(
        self,
        value: float | NullType,
    ) -> PolarsDataFrame:
        if isinstance(value, Null):
            value = None
        return PolarsDataFrame(self.dataframe.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(
        self,
        value: Any,
        *,
        column_names: list[str] | None = None,
    ) -> PolarsDataFrame:
        if column_names is None:
            column_names = self.dataframe.columns
        df = self.dataframe.with_columns(
            pl.col(col).fill_null(value) for col in column_names
        )
        return PolarsDataFrame(df, api_version=self._api_version)

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.dataframe.to_numpy().astype(dtype)

    def join(
        self,
        other: DataFrame,
        left_on: str | list[str],
        right_on: str | list[str],
        how: Literal["left", "inner", "outer"],
    ) -> PolarsDataFrame:
        return self.maybe_lazify().join(other)

    def maybe_lazify(self) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.lazy(), api_version=self._api_version)
