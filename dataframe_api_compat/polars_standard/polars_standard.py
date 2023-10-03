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
        Expression,
        Bool,
        DataFrame,
        EagerFrame,
        EagerColumn,
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


LATEST_API_VERSION = "2023.09-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.08-beta"))


class PolarsColumn(EagerColumn[DType]):
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

    def _validate_column(self, column: PolarsColumn[Any] | EagerColumn[Any]) -> None:
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
        return len(self.column)

    @property
    def dtype(self) -> Any:
        return dataframe_api_compat.polars_standard.DTYPE_MAP[self._dtype]

    def get_rows(self, indices: EagerColumn[Any]) -> PolarsColumn[DType]:
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

    def filter(self, mask: Expression | EagerColumn[Any]) -> PolarsColumn[DType]:
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
        self, other: EagerColumn[DType] | Any
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
        self, other: EagerColumn[DType] | Any
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

    def __ge__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column >= other.column,
                api_version=self._api_version,
            )
        return PolarsColumn(
            self.column >= other,
            api_version=self._api_version,
        )

    def __gt__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Bool]:
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

    def __le__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Bool]:
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

    def __lt__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Bool]:
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

    def __mul__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            res = self.column * other.column
            return PolarsColumn(res, api_version=self._api_version)
        res = self.column * other
        return PolarsColumn(res, api_version=self._api_version)

    def __floordiv__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Any]:
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

    def __truediv__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res = self.column / other.column
            return PolarsColumn(res, api_version=self._api_version)
        res = self.column / other
        return PolarsColumn(res, api_version=self._api_version)

    def __pow__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            ret = self.column**other.column  # type: ignore[operator]
        else:
            ret = self.column.pow(other)  # type: ignore[arg-type]
        return PolarsColumn(ret, api_version=self._api_version)

    def __mod__(self, other: EagerColumn[DType] | Any) -> PolarsColumn[Any]:
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
        other: EagerColumn[DType] | Any,
    ) -> tuple[PolarsColumn[Any], PolarsColumn[Any]]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: EagerColumn[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column & other.column, api_version=self._api_version  # type: ignore[operator]
            )
        return PolarsColumn(self.column & other, api_version=self._api_version)  # type: ignore[operator]

    def __or__(self, other: EagerColumn[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column | other.column, api_version=self._api_version  # type: ignore[operator]
            )
        return PolarsColumn(self.column | other, api_version=self._api_version)  # type: ignore[operator]

    def __invert__(self) -> PolarsColumn[Bool]:
        return PolarsColumn(~self.column, api_version=self._api_version)

    def __add__(self, other: EagerColumn[Any] | Any) -> PolarsColumn[Any]:
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

    def __sub__(self, other: EagerColumn[Any] | Any) -> PolarsColumn[Any]:
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
        return PolarsColumn(
            self.column.rename(name),
            api_version=self._api_version,
        )

    def _to_expression(self) -> PolarsExpression:
        return PolarsExpression(pl.lit(self.column), api_version=self._api_version)


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
            self._expr = pl.col(expr)
        else:
            self._expr = expr
        # need to pass this down from namespace.col
        self._api_version = api_version or LATEST_API_VERSION

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def root_names(self) -> list[str]:
        return sorted(set(self._expr.meta.root_names()))

    @property
    def output_name(self) -> list[str]:
        return self._expr.meta.output_name()

    @property
    def name(self) -> str:
        if isinstance(self._expr, pl.Series):
            # TODO: can we avoid this completely?
            name = self._expr.name
        else:
            name = self._expr.meta.output_name()
        return name

    def len(self) -> PolarsExpression:
        return PolarsExpression(self._expr.len(), api_version=self._api_version)

    def get_rows(self, indices: PolarsExpression) -> PolarsExpression:
        return PolarsExpression(
            self._expr.take(indices._expr),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsExpression:
        if start is None:
            start = 0
        length = None if stop is None else stop - start
        if step is None:
            step = 1
        return PolarsExpression(
            self._expr.slice(start, length).take_every(step),
            api_version=self._api_version,
        )

    def filter(self, mask: PolarsExpression) -> PolarsExpression:
        return PolarsExpression(
            self._expr.filter(mask._expr), api_version=self._api_version  # type: ignore[arg-type]
        )

    def get_value(self, row: int) -> Any:
        return PolarsExpression(
            self._expr.take(row), api_version=self._api_version  # type: ignore[arg-type]
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsExpression) -> PolarsExpression:  # type: ignore[override]
        return PolarsExpression(
            self._expr.is_in(values._expr), api_version=self._api_version  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsExpression:
        raise NotImplementedError()

    def is_null(self) -> PolarsExpression:
        return PolarsExpression(
            self._expr.is_null(),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsExpression:
        return PolarsExpression(
            self._expr.is_nan(),
            api_version=self._api_version,
        )

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        return PolarsExpression(self._expr.any(), api_version=self._api_version)

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        return PolarsExpression(self._expr.all(), api_version=self._api_version)

    def min(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.min(),
            api_version=self._api_version,
        )

    def max(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.max(),
            api_version=self._api_version,
        )

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.sum(),
            api_version=self._api_version,
        )

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.product(),
            api_version=self._api_version,
        )

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.mean(),
            api_version=self._api_version,
        )

    def median(self, *, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.median(),
            api_version=self._api_version,
        )

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.std(),
            api_version=self._api_version,
        )

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return PolarsExpression(
            self._expr.var(),
            api_version=self._api_version,
        )

    def __eq__(  # type: ignore[override]
        self, other: PolarsExpression | Any
    ) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr == other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr == other,
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self, other: PolarsExpression | Any
    ) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr != other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr != other,
            api_version=self._api_version,
        )

    def __ge__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr >= other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr >= other,
            api_version=self._api_version,
        )

    def __gt__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr > other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr > other,
            api_version=self._api_version,
        )

    def __le__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr <= other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr <= other,
            api_version=self._api_version,
        )

    def __lt__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr < other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr < other,
            api_version=self._api_version,
        )

    def __mul__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res = self._expr * other._expr
            return PolarsExpression(res, api_version=self._api_version)
        res = self._expr * other
        return PolarsExpression(res, api_version=self._api_version)

    def __floordiv__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr // other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr // other,
            api_version=self._api_version,
        )

    def __truediv__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res = self._expr / other._expr
            return PolarsExpression(res, api_version=self._api_version)
        res = self._expr / other
        return PolarsExpression(res, api_version=self._api_version)

    def __pow__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            ret = self._expr**other._expr  # type: ignore[operator]
        else:
            ret = self._expr.pow(other)  # type: ignore[arg-type]
        return PolarsExpression(ret, api_version=self._api_version)

    def __mod__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr % other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr % other,
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
            return PolarsExpression(self._expr & other._expr)
        return PolarsExpression(self._expr & other)  # type: ignore[operator]

    def __or__(self, other: PolarsExpression | bool) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(self._expr | other._expr)
        return PolarsExpression(self._expr | other)

    def __invert__(self) -> PolarsExpression:
        return PolarsExpression(~self._expr, api_version=self._api_version)

    def __add__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr + other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr + other,
            api_version=self._api_version,
        )

    def __sub__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr - other._expr,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr - other,
            api_version=self._api_version,
        )

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsExpression:
        expr = self._expr.arg_sort(descending=not ascending)
        return PolarsExpression(
            expr,
            api_version=self._api_version,
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsExpression:
        expr = self._expr.sort(descending=not ascending)
        return PolarsExpression(
            expr,
            api_version=self._api_version,
        )

    def fill_nan(self, value: float | NullType) -> PolarsExpression:
        return PolarsExpression(self._expr.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsExpression:
        return PolarsExpression(
            self._expr.fill_null(value),
            api_version=self._api_version,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cumsum(),
            api_version=self._api_version,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cumprod(),
            api_version=self._api_version,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cummax(),
            api_version=self._api_version,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cummin(),
            api_version=self._api_version,
        )

    def rename(self, name: str) -> PolarsExpression:
        return PolarsExpression(
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
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.polars_standard.DTYPE_MAP[dtype]
            for column_name, dtype in self.dataframe.schema.items()
        }

    def __repr__(self) -> str:
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

    def select(self, *columns: str | Expression | EagerColumn[Any]) -> PolarsDataFrame:
        resolved_names = []
        for name in columns:
            if isinstance(name, PolarsExpression):
                resolved_names.append(name._expr)
            elif isinstance(name, str):
                resolved_names.append(name)
            else:
                raise AssertionError(
                    f"Expected str or PolarsExpression, got: {type(name)}"
                )
        return PolarsDataFrame(
            self.df.select(resolved_names), api_version=self._api_version
        )

    def get_rows(self, indices: PolarsExpression) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices._expr)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self._api_version)

    def filter(self, mask: Expression | EagerColumn[Any]) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask._expr), api_version=self._api_version)

    def assign(self, *columns: Expression | EagerColumn[Any]) -> PolarsDataFrame:
        new_columns = []
        for col in columns:
            if isinstance(col, PolarsExpression):
                new_columns.append(col._expr)
            elif isinstance(col, PolarsColumn):
                new_columns.append(col.column)
            else:
                raise TypeError(
                    f"Expected PolarsExpression or PolarsColumn, got: {type(col)}"
                )
        df = self.dataframe.with_columns(new_columns)
        return PolarsDataFrame(df, api_version=self._api_version)

    def update_columns(self, *columns: PolarsExpression | PolarsColumn) -> PolarsDataFrame:  # type: ignore[override]
        original_n_columns = len(self.get_column_names())
        new_columns = []
        for col in columns:
            if isinstance(col, PolarsExpression):
                new_columns.append(col)
            elif isinstance(col, PolarsColumn):
                new_columns.append(col.to_expression())
            else:
                raise TypeError(
                    f"Expected PolarsExpression or PolarsColumn, got: {type(col)}"
                )
        df = PolarsDataFrame(
            self.dataframe.with_columns([col._expr for col in new_columns]),
            api_version=self._api_version,
        )
        if len(df.get_column_names()) != original_n_columns:
            raise ValueError("tried inserting a new column, use insert_columns instead")
        return df

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
        *keys: str | Expression | EagerColumn[Any],
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
        assert isinstance(other, (PolarsDataFrame, PolarsEagerFrame))
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


class PolarsEagerFrame(EagerFrame):
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

    def __repr__(self) -> str:
        return self.dataframe.__repr__()

    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.polars_standard.DTYPE_MAP[dtype]
            for column_name, dtype in self.dataframe.schema.items()
        }

    @property
    def dataframe(self) -> pl.LazyFrame:
        return self.df

    def groupby(self, *keys: str) -> PolarsGroupBy:
        return PolarsGroupBy(self.df.lazy(), list(keys), api_version=self._api_version)

    def select(self, *columns: str | Expression | EagerColumn[Any]) -> PolarsEagerFrame:
        return self.relax().select(*columns).collect()

    def get_column(self, name) -> PolarsColumn:
        return PolarsColumn(
            self.dataframe.get_column(name), api_version=self._api_version
        )

    def get_rows(self, indices: PolarsColumn[Any]) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsEagerFrame(
            self.dataframe.select(pl.all().take(indices.column)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsEagerFrame(self.df[start:stop:step], api_version=self._api_version)

    def filter(self, mask: PolarsExpression | PolarsColumn) -> PolarsDataFrame:
        # todo: how to convert polars series to expression?
        if isinstance(mask, PolarsColumn):
            mask = mask._to_expression()
        return PolarsEagerFrame(self.df.filter(mask._expr), api_version=self._api_version)

    def insert(self, loc: int, label: str, value: PolarsExpression) -> PolarsDataFrame:
        if self._api_version != "2023.08-beta":
            raise NotImplementedError(
                "DataFrame.insert is only available for api version 2023.08-beta. "
                "Please use `DataFrame.insert_column` instead."
            )
        columns = self.dataframe.columns
        new_columns = columns[:loc] + [label] + columns[loc:]
        df = self.dataframe.with_columns(value._expr.alias(label)).select(new_columns)
        return PolarsEagerFrame(df, api_version=self._api_version)

    def assign(self, *columns: PolarsExpression | PolarsColumn) -> PolarsDataFrame:
        return self.relax().assign(*columns).collect()

    def drop_columns(self, *labels: str) -> PolarsDataFrame:
        return self.relax().drop_columns(*labels).collect()

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsEagerFrame(
            self.dataframe.rename(dict(mapping)), api_version=self._api_version
        )

    def get_column_names(self) -> list[str]:
        return self.dataframe.columns

    def __eq__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__eq__(other)),
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
            api_version=self._api_version,
        )

    def __ge__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
            api_version=self._api_version,
        )

    def __gt__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
            api_version=self._api_version,
        )

    def __le__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
            api_version=self._api_version,
        )

    def __lt__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
            api_version=self._api_version,
        )

    def __and__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*") & other),
            api_version=self._api_version,
        )

    def __or__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self._api_version,
        )

    def __add__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self._api_version,
        )

    def __sub__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self._api_version,
        )

    def __mul__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self._api_version,
        )

    def __truediv__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self._api_version,
        )

    def __floordiv__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
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
        return PolarsEagerFrame(ret, api_version=self._api_version)

    def __mod__(self, other: Any) -> PolarsDataFrame:
        return PolarsEagerFrame(
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
        return PolarsEagerFrame(
            quotient_df, api_version=self._api_version
        ), PolarsEagerFrame(remainder_df, api_version=self._api_version)

    def __invert__(self) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(~pl.col("*")), api_version=self._api_version
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.with_columns(pl.col("*").is_null()),
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsEagerFrame(df, api_version=self._api_version)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").any()), api_version=self._api_version
        )

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").all()), api_version=self._api_version
        )

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").min()), api_version=self._api_version
        )

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").max()), api_version=self._api_version
        )

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").sum()), api_version=self._api_version
        )

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").product()), api_version=self._api_version
        )

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").mean()), api_version=self._api_version
        )

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").median()), api_version=self._api_version
        )

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsEagerFrame(
            self.dataframe.select(pl.col("*").std()), api_version=self._api_version
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsEagerFrame(
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
        return PolarsEagerFrame(self.dataframe.fill_nan(value), api_version=self._api_version)  # type: ignore[arg-type]

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
        return PolarsEagerFrame(df, api_version=self._api_version)

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.dataframe.to_numpy().astype(dtype)

    def join(
        self,
        other: PolarsEagerFrame,
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
