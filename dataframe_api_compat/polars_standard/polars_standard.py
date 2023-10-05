from __future__ import annotations

import collections
from typing import Any
from typing import Generic
from typing import Literal
from typing import NoReturn
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
        Column,
        Bool,
        DataFrame,
        PermissiveFrame,
        PermissiveColumn,
        GroupBy,
    )
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


class Null:
    ...


null = Null()
NullType = type[Null]


def _is_integer_dtype(dtype: Any) -> bool:
    return any(  # pragma: no cover
        # definitely covered, not sure what this is
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


LATEST_API_VERSION = "2023.09-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.08-beta"))


class PolarsPermissiveColumn(PermissiveColumn[DType]):
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
            raise AssertionError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version
        self._dtype = column.dtype

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def name(self) -> str:
        return self.column.name

    @property
    def column(self) -> pl.Series | pl.Expr:
        return self._series

    def len(self) -> int:
        return len(self.column)

    @property
    def dtype(self) -> Any:
        return dataframe_api_compat.polars_standard.map_polars_dtype_to_standard_dtype(
            self._dtype
        )

    def get_rows(self, indices: PermissiveColumn[Any]) -> PolarsPermissiveColumn[DType]:
        return PolarsPermissiveColumn(
            self.column.take(indices.column),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsPermissiveColumn[DType]:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("slice_rows not implemented for lazy columns")
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.column)
        if step is None:
            step = 1
        return PolarsPermissiveColumn(
            self.column[start:stop:step],
            api_version=self._api_version,
        )

    def filter(
        self, mask: Column | PermissiveColumn[Any]
    ) -> PolarsPermissiveColumn[DType]:
        return PolarsPermissiveColumn(
            self.column.filter(mask.column),
            api_version=self._api_version,
        )

    def get_value(self, row: int) -> Any:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("get_value not implemented for lazy columns")
        return self.column[row]

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsPermissiveColumn[DType]) -> PolarsPermissiveColumn[Bool]:  # type: ignore[override]
        return PolarsPermissiveColumn(
            self.column.is_in(values.column), api_version=self._api_version  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsPermissiveColumn[Any]:
        raise NotImplementedError("not yet supported")

    def is_null(self) -> PolarsPermissiveColumn[Bool]:
        return self._from_expression(self._to_expression().is_null())

    def is_nan(self) -> PolarsPermissiveColumn[Bool]:
        return self._from_expression(self._to_expression().is_nan())

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        return self.column.any()

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        return self.column.all()

    def min(self, *, skip_nulls: bool = True) -> Any:
        return self.column.min()

    def max(self, *, skip_nulls: bool = True) -> Any:
        return self.column.max()

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return self.column.sum()

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return self.column.product()

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return self.column.mean()

    def median(self, *, skip_nulls: bool = True) -> Any:
        return self.column.median()

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self.column.std()

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self.column.var()

    def __eq__(  # type: ignore[override]
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__eq__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__eq__(other))

    def __ne__(  # type: ignore[override]
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__ne__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__ne__(other))

    def __ge__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__ge__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__ge__(other))

    def __gt__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__gt__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__gt__(other))

    def __le__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__le__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__le__(other))

    def __lt__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__lt__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__lt__(other))

    def __mul__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__mul__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__mul__(other))

    def __floordiv__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__floordiv__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__floordiv__(other))

    def __truediv__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__truediv__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__truediv__(other))

    def __pow__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__pow__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__pow__(other))

    def __mod__(
        self, other: PermissiveColumn[DType] | Any
    ) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__mod__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__mod__(other))

    def __divmod__(
        self,
        other: PermissiveColumn[DType] | Any,
    ) -> tuple[PolarsPermissiveColumn[Any], PolarsPermissiveColumn[Any]]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(
        self, other: PermissiveColumn[Bool] | bool
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__and__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__and__(other))

    def __or__(
        self, other: PermissiveColumn[Bool] | bool
    ) -> PolarsPermissiveColumn[Bool]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__or__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__or__(other))

    def __invert__(self) -> PolarsPermissiveColumn[Bool]:
        return self._from_expression(self._to_expression().__invert__())

    def __add__(self, other: PermissiveColumn[Any] | Any) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__add__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__add__(other))

    def __sub__(self, other: PermissiveColumn[Any] | Any) -> PolarsPermissiveColumn[Any]:
        if isinstance(other, PermissiveColumn):
            return self._from_expression(
                self._to_expression().__sub__(other._to_expression())
            )
        return self._from_expression(self._to_expression().__sub__(other))

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsPermissiveColumn[Any]:
        expr = self.column.arg_sort(descending=not ascending)
        return PolarsPermissiveColumn(
            expr,
            api_version=self._api_version,
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsPermissiveColumn[Any]:
        expr = self.column.sort(descending=not ascending)
        return PolarsPermissiveColumn(
            expr,
            api_version=self._api_version,
        )

    def fill_nan(self, value: float | NullType) -> PolarsPermissiveColumn[DType]:
        return self._from_expression(self._to_expression().fill_nan(value))

    def fill_null(self, value: Any) -> PolarsPermissiveColumn[DType]:
        return self._from_expression(self._to_expression().fill_null(value))

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expression(
            self._to_expression().cumulative_sum(skip_nulls=skip_nulls)
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expression(
            self._to_expression().cumulative_prod(skip_nulls=skip_nulls)
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expression(
            self._to_expression().cumulative_max(skip_nulls=skip_nulls)
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expression(
            self._to_expression().cumulative_min(skip_nulls=skip_nulls)
        )

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy().astype(dtype)

    def rename(self, name: str) -> PolarsPermissiveColumn[DType]:
        return PolarsPermissiveColumn(
            self.column.rename(name),
            api_version=self._api_version,
        )

    def _to_expression(self) -> PolarsColumn:
        return PolarsColumn(pl.lit(self.column), api_version=self._api_version)

    def _from_expression(self, expression: PolarsColumn):
        df = pl.select(expression._expr)
        return PolarsPermissiveColumn(
            df.get_column(df.columns[0]), api_version=self._api_version
        )


class PolarsGroupBy(GroupBy):
    def __init__(self, df: pl.LazyFrame, keys: Sequence[str], api_version: str) -> None:
        assert isinstance(df, pl.LazyFrame)
        for key in keys:
            if key not in df.columns:
                raise KeyError(f"key {key} not present in DataFrame's columns")
        self.df = df
        self.keys = keys
        self._api_version = api_version
        self.group_by = (
            self.df.group_by if pl.__version__ < "0.19.0" else self.df.group_by
        )

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


class PolarsColumn:
    def __init__(
        self,
        expr: pl.Series | pl.Expr,
        *,
        api_version: str | None = None,
    ) -> None:
        if expr is NotImplemented:
            raise NotImplementedError("operation not implemented")
        if isinstance(expr, str):
            self._expr = pl.col(expr)
        else:
            self._expr = expr
        # need to pass this down from namespace.col
        self._api_version = api_version or LATEST_API_VERSION

    # In the standard
    def __column_namespace__(self) -> Any:  # pragma: no cover
        return dataframe_api_compat.polars_standard

    @property
    def root_names(self) -> list[str]:
        return sorted(set(self._expr.meta.root_names()))

    @property
    def output_name(self) -> list[str]:
        return self._expr.meta.output_name()

    def len(self) -> PolarsColumn:
        return PolarsColumn(self._expr.len(), api_version=self._api_version)

    def get_rows(self, indices: PolarsColumn) -> PolarsColumn:
        return PolarsColumn(
            self._expr.take(indices._expr),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsColumn:
        if start is None:
            start = 0
        length = None if stop is None else stop - start
        if step is None:
            step = 1
        return PolarsColumn(
            self._expr.slice(start, length).take_every(step),
            api_version=self._api_version,
        )

    def filter(self, mask: PolarsColumn) -> PolarsColumn:
        return PolarsColumn(
            self._expr.filter(mask._expr), api_version=self._api_version  # type: ignore[arg-type]
        )

    def get_value(self, row: int) -> Any:
        return PolarsColumn(
            self._expr.take(row), api_version=self._api_version  # type: ignore[arg-type]
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsColumn) -> PolarsColumn:  # type: ignore[override]
        return PolarsColumn(
            self._expr.is_in(values._expr), api_version=self._api_version  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn:
        raise NotImplementedError()

    def is_null(self) -> PolarsColumn:
        return PolarsColumn(
            self._expr.is_null(),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsColumn:
        return PolarsColumn(
            self._expr.is_nan(),
            api_version=self._api_version,
        )

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        return PolarsColumn(self._expr.any(), api_version=self._api_version)

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        return PolarsColumn(self._expr.all(), api_version=self._api_version)

    def min(self, *, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.min(),
            api_version=self._api_version,
        )

    def max(self, *, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.max(),
            api_version=self._api_version,
        )

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.sum(),
            api_version=self._api_version,
        )

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.product(),
            api_version=self._api_version,
        )

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.mean(),
            api_version=self._api_version,
        )

    def median(self, *, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.median(),
            api_version=self._api_version,
        )

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.std(),
            api_version=self._api_version,
        )

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return PolarsColumn(
            self._expr.var(),
            api_version=self._api_version,
        )

    def __eq__(self, other: PolarsColumn | Any) -> PolarsColumn:  # type: ignore[override]
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr == other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr == other,
            api_version=self._api_version,
        )

    def __ne__(self, other: PolarsColumn | Any) -> PolarsColumn:  # type: ignore[override]
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr != other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr != other,
            api_version=self._api_version,
        )

    def __ge__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr >= other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr >= other,
            api_version=self._api_version,
        )

    def __gt__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr > other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr > other,
            api_version=self._api_version,
        )

    def __le__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr <= other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr <= other,
            api_version=self._api_version,
        )

    def __lt__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr < other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr < other,
            api_version=self._api_version,
        )

    def __mul__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            res = self._expr * other._expr
            return PolarsColumn(res, api_version=self._api_version)
        res = self._expr * other
        return PolarsColumn(res, api_version=self._api_version)

    def __floordiv__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr // other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr // other,
            api_version=self._api_version,
        )

    def __truediv__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            res = self._expr / other._expr
            return PolarsColumn(res, api_version=self._api_version)
        res = self._expr / other
        return PolarsColumn(res, api_version=self._api_version)

    def __pow__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            ret = self._expr**other._expr  # type: ignore[operator]
        else:
            ret = self._expr.pow(other)  # type: ignore[arg-type]
        return PolarsColumn(ret, api_version=self._api_version)

    def __mod__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr % other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr % other,
            api_version=self._api_version,
        )

    def __divmod__(
        self,
        other: PolarsColumn | Any,
    ) -> tuple[PolarsColumn, PolarsColumn]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: PolarsColumn | bool) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._expr & other._expr)
        return PolarsColumn(self._expr & other)  # type: ignore[operator]

    def __or__(self, other: PolarsColumn | bool) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._expr | other._expr)
        return PolarsColumn(self._expr | other)

    def __invert__(self) -> PolarsColumn:
        return PolarsColumn(~self._expr, api_version=self._api_version)

    def __add__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr + other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr + other,
            api_version=self._api_version,
        )

    def __sub__(self, other: PolarsColumn | Any) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self._expr - other._expr,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self._expr - other,
            api_version=self._api_version,
        )

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn:
        expr = self._expr.arg_sort(descending=not ascending)
        return PolarsColumn(
            expr,
            api_version=self._api_version,
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn:
        expr = self._expr.sort(descending=not ascending)
        return PolarsColumn(
            expr,
            api_version=self._api_version,
        )

    def fill_nan(self, value: float | NullType) -> PolarsColumn:
        return PolarsColumn(self._expr.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsColumn:
        return PolarsColumn(
            self._expr.fill_null(value),
            api_version=self._api_version,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return PolarsColumn(
            self._expr.cumsum(),
            api_version=self._api_version,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return PolarsColumn(
            self._expr.cumprod(),
            api_version=self._api_version,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return PolarsColumn(
            self._expr.cummax(),
            api_version=self._api_version,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return PolarsColumn(
            self._expr.cummin(),
            api_version=self._api_version,
        )

    def rename(self, name: str) -> PolarsColumn:
        return PolarsColumn(
            self._expr.alias(name),
            api_version=self._api_version,
        )


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.LazyFrame, api_version: str) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        if df is NotImplemented:
            raise NotImplementedError("operation not implemented")
        assert isinstance(df, pl.LazyFrame)
        self.df = df
        self._id = id(df)
        if api_version not in SUPPORTED_VERSIONS:
            raise AssertionError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.polars_standard.map_polars_dtype_to_standard_dtype(
                dtype
            )
            for column_name, dtype in self.dataframe.schema.items()
        }

    def __repr__(self) -> str:  # pragma: no cover
        return self.dataframe.__repr__()

    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns

    @property
    def dataframe(self) -> pl.LazyFrame:
        return self.df

    def group_by(self, *keys: str) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, list(keys), api_version=self._api_version)

    def select(self, *columns: str | Column | PermissiveColumn[Any]) -> PolarsDataFrame:
        resolved_names = []
        for name in columns:
            if isinstance(name, PolarsColumn):
                resolved_names.append(name._expr)
            elif isinstance(name, str):
                resolved_names.append(name)
            else:
                raise AssertionError(f"Expected str or PolarsColumn, got: {type(name)}")
        return PolarsDataFrame(
            self.df.select(resolved_names), api_version=self._api_version
        )

    def get_rows(self, indices: PolarsColumn) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices._expr)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self._api_version)

    def filter(self, mask: Column | PermissiveColumn[Any]) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask._expr), api_version=self._api_version)

    def assign(self, *columns: Column | PermissiveColumn[Any]) -> PolarsDataFrame:
        new_columns = []
        for col in columns:
            if isinstance(col, PolarsColumn):
                new_columns.append(col._expr)
            elif isinstance(col, PolarsPermissiveColumn):
                new_columns.append(col.column)
            else:
                raise AssertionError(
                    f"Expected PolarsColumn or PolarsPermissiveColumn, got: {type(col)}"
                )
        df = self.dataframe.with_columns(new_columns)
        return PolarsDataFrame(df, api_version=self._api_version)

    def drop_columns(self, *labels: str) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.drop(labels), api_version=self._api_version)

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsDataFrame(
            self.dataframe.rename(dict(mapping)), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:  # pragma: no cover
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
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
        ret = self.dataframe.select([pl.col(col).pow(other) for col in self.column_names])
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
        *keys: str | Column | PermissiveColumn[Any],
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsDataFrame:
        if not keys:
            keys = self.dataframe.columns
        # TODO: what if there's multiple `ascending`?
        return PolarsDataFrame(
            self.dataframe.sort(list(keys), descending=not ascending),
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
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
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
        assert isinstance(other, (PolarsDataFrame, PolarsPermissiveFrame))
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

    def collect(self) -> PolarsPermissiveFrame:
        return PolarsPermissiveFrame(
            self.dataframe.collect(), api_version=self._api_version
        )


class PolarsPermissiveFrame(PermissiveFrame):
    def __init__(self, df: pl.LazyFrame, api_version: str) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        if df is NotImplemented:
            raise NotImplementedError("operation not implemented")
        self.df = df
        self._id = id(df)
        if api_version not in SUPPORTED_VERSIONS:
            raise AssertionError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    def __repr__(self) -> str:  # pragma: no cover
        return self.dataframe.__repr__()

    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.polars_standard.map_polars_dtype_to_standard_dtype(
                dtype
            )
            for column_name, dtype in self.dataframe.schema.items()
        }

    @property
    def dataframe(self) -> pl.LazyFrame:
        return self.df

    def group_by(self, *keys: str) -> PolarsGroupBy:
        return PolarsGroupBy(self.df.lazy(), list(keys), api_version=self._api_version)

    def select(
        self, *columns: str | Column | PermissiveColumn[Any]
    ) -> PolarsPermissiveFrame:
        return self.relax().select(*columns).collect()

    def get_column_by_name(self, name) -> PolarsPermissiveColumn:
        return PolarsPermissiveColumn(
            self.dataframe.get_column(name), api_version=self._api_version
        )

    def get_rows(self, indices: PolarsColumn[Any]) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.all().take(indices.column)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.df[start:stop:step], api_version=self._api_version
        )

    def filter(self, mask: PolarsPermissiveColumn | PolarsColumn) -> PolarsDataFrame:
        if isinstance(mask, PolarsPermissiveColumn):
            mask = mask._to_expression()
        return PolarsPermissiveFrame(
            self.df.filter(mask._expr), api_version=self._api_version
        )

    def assign(self, *columns: PolarsColumn | PolarsColumn) -> PolarsDataFrame:
        return self.relax().assign(*columns).collect()

    def drop_columns(self, *labels: str) -> PolarsDataFrame:
        return self.relax().drop_columns(*labels).collect()

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsPermissiveFrame(
            self.dataframe.rename(dict(mapping)), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:
        return self.dataframe.columns

    def __eq__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__eq__(other)),
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
            api_version=self._api_version,
        )

    def __ge__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
            api_version=self._api_version,
        )

    def __gt__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
            api_version=self._api_version,
        )

    def __le__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
            api_version=self._api_version,
        )

    def __lt__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
            api_version=self._api_version,
        )

    def __and__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*") & other),
            api_version=self._api_version,
        )

    def __or__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self._api_version,
        )

    def __add__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self._api_version,
        )

    def __sub__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self._api_version,
        )

    def __mul__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self._api_version,
        )

    def __truediv__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self._api_version,
        )

    def __floordiv__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
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
        return PolarsPermissiveFrame(ret, api_version=self._api_version)

    def __mod__(self, other: Any) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
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
        return PolarsPermissiveFrame(
            quotient_df, api_version=self._api_version
        ), PolarsPermissiveFrame(remainder_df, api_version=self._api_version)

    def __invert__(self) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(~pl.col("*")), api_version=self._api_version
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.with_columns(pl.col("*").is_null()),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsPermissiveFrame(df, api_version=self._api_version)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").any()), api_version=self._api_version
        )

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").all()), api_version=self._api_version
        )

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").min()), api_version=self._api_version
        )

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").max()), api_version=self._api_version
        )

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").sum()), api_version=self._api_version
        )

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").product()), api_version=self._api_version
        )

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").mean()), api_version=self._api_version
        )

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").median()), api_version=self._api_version
        )

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").std()), api_version=self._api_version
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsPermissiveFrame(
            self.dataframe.select(pl.col("*").var()), api_version=self._api_version
        )

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsDataFrame:
        return (
            self.relax()
            .sort(*keys, ascending=ascending, nulls_position=nulls_position)
            .collect()
        )

    def fill_nan(
        self,
        value: float | NullType,
    ) -> PolarsDataFrame:
        if isinstance(value, Null):
            value = None
        return PolarsPermissiveFrame(self.dataframe.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

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
        return PolarsPermissiveFrame(df, api_version=self._api_version)

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.dataframe.to_numpy().astype(dtype)

    def join(
        self,
        other: PolarsPermissiveFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> PolarsDataFrame:
        return (
            self.relax()
            .join(other.relax(), left_on=left_on, right_on=right_on, how=how)
            .collect()
        )

    def relax(self) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.lazy(), api_version=self._api_version)
