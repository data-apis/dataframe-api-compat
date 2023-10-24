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
        expr: pl.Expr,
        *,
        id_: int,
        api_version: str | None = None,
    ) -> None:
        self._expr = expr
        self._id = id_
        self._api_version = api_version
        self._name = expr.meta.output_name()

    def _from_expr(self, expr):
        return self.__class__(expr, id_=self._id, api_version=self._api_version)

    # In the standard
    def __column_namespace__(self) -> Any:  # pragma: no cover
        return dataframe_api_compat.polars_standard

    @property
    def name(self):
        return self._name

    def len(self) -> PolarsColumn:
        return self._from_expr(self._expr.len())

    def get_rows(self, indices: PolarsColumn) -> PolarsColumn:
        return self._from_expr(self._expr.take(indices._expr))

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsColumn:
        if start is None:
            start = 0
        length = None if stop is None else stop - start
        if step is None:
            step = 1
        return self._from_expr(self._expr.slice(start, length).take_every(step))

    def filter(self, mask: PolarsColumn) -> PolarsColumn:
        return self._from_expr(self._expr.filter(mask._expr))

    def get_value(self, row: int) -> Any:
        raise NotImplementedError("can't get value out, use to_array instead")

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsColumn) -> PolarsColumn:  # type: ignore[override]
        return self._from_expr(self._expr.is_in(values._expr))

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn:
        raise NotImplementedError()

    def is_null(self) -> PolarsColumn:
        return self._from_expr(self._expr.is_null())

    def is_nan(self) -> PolarsColumn:
        return self._from_expr(self._expr.is_nan())

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        return self._from_expr(self._expr.any())

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        return self._from_expr(self._expr.all())

    def min(self, *, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.min())

    def max(self, *, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.max())

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.sum())

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.product())

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.mean())

    def median(self, *, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.median())

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.std())

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._from_expr(self._expr.var())

    def __eq__(self, other: PolarsColumn | Any) -> PolarsColumn:  # type: ignore[override]
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr == other)

    def __ne__(self, other: PolarsColumn | Any) -> PolarsColumn:  # type: ignore[override]
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr != other)

    def __ge__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr >= other)

    def __gt__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr > other)

    def __le__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr <= other)

    def __lt__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr < other)

    def __mul__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        res = self._expr * other
        return self._from_expr(res)

    def __floordiv__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr // other)

    def __truediv__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        res = self._expr / other
        return self._from_expr(res)

    def __pow__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        ret = self._expr.pow(other)  # type: ignore[arg-type]
        return self._from_expr(ret)

    def __mod__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr % other)

    def __divmod__(
        self,
        other: PolarsColumn | Any,
    ) -> tuple[PolarsColumn, PolarsColumn]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: PolarsColumn | bool) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr & other)  # type: ignore[operator]

    def __or__(self, other: PolarsColumn | bool) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr | other)

    def __invert__(self) -> PolarsColumn:
        return self._from_expr(~self._expr)

    def __add__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr + other)

    def __sub__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = other._expr if isinstance(other, PolarsColumn) else other
        return self._from_expr(self._expr - other)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn:
        expr = self._expr.arg_sort(descending=not ascending)
        return self._from_expr(expr)

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn:
        expr = self._expr.sort(descending=not ascending)
        return self._from_expr(expr)

    def fill_nan(self, value: float | NullType) -> PolarsColumn:
        return self._from_expr(self._expr.fill_nan(value))  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsColumn:
        return self._from_expr(self._expr.fill_null(value))

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self._expr.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self._expr.cumprod())

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self._expr.cummax())

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self._expr.cummin())

    def rename(self, name: str) -> PolarsColumn:
        return self._from_expr(self._expr.alias(name))

    @property
    def dt(self) -> ColumnDatetimeAccessor:
        """
        Return accessor with functions which work on temporal dtypes.
        """
        return ColumnDatetimeAccessor(self)


class ColumnDatetimeAccessor:
    def __init__(self, column: PolarsColumn) -> None:
        self.column = column
        self._api_version = column._api_version

    def _from_expr(self, expr):
        return self.column.__class__(
            expr, id_=self.column._id, api_version=self._api_version
        )

    def year(self) -> Column:
        return self._from_expr(self.column._expr.dt.year())

    def month(self) -> Column:
        return self._from_expr(self.column._expr.dt.month())

    def day(self) -> Column:
        return self._from_expr(self.column._expr.dt.day())

    def hour(self) -> Column:
        return self._from_expr(self.column._expr.dt.hour())

    def minute(self) -> Column:
        return self._from_expr(self.column._expr.dt.minute())

    def second(self) -> Column:
        return self._from_expr(self.column._expr.dt.second())

    def microsecond(self) -> Column:
        return self._from_expr(self.column._expr.dt.microsecond())

    def iso_weekday(self) -> Column:
        return self._from_expr(self.column._expr.dt.weekday())

    def floor(self, frequency: str) -> Column:
        frequency = (
            frequency.replace("day", "d")
            .replace("hour", "h")
            .replace("minute", "m")
            .replace("second", "s")
            .replace("millisecond", "ms")
            .replace("microsecond", "us")
            .replace("nanosecond", "ns")
        )
        return self._from_expr(self.column._expr.dt.truncate(frequency))

    def unix_timestamp(self) -> Column:
        return self._from_expr(self.column._expr.dt.timestamp("ms") // 1000)


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.LazyFrame | pl.DataFrame, api_version: str) -> None:
        self.df = df
        self._id = id(df)
        if api_version not in SUPPORTED_VERSIONS:
            raise AssertionError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    def col(self, value) -> PolarsColumn:
        return PolarsColumn(pl.col(value), id_=self._id, api_version=self._api_version)

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
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsGroupBy(
                self.dataframe, list(keys), api_version=self._api_version
            )
        return PolarsGroupBy(
            self.dataframe.lazy(), list(keys), api_version=self._api_version
        )

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
            else:
                raise AssertionError(f"Expected PolarsColumn, got: {type(col)}")
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

    def collect(self) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsDataFrame(
                self.dataframe.collect(), api_version=self._api_version
            )
        raise ValueError("DataFrame was already collected")

    def to_array(self, dtype) -> Any:
        if not isinstance(self.dataframe, pl.DataFrame):
            raise ValueError(
                "to_array() can only be called if the dataframe has already been collected.\n"
                "\n"
                "Try calling .collect() first.\n"
                "\n"
                "NOTE: `.collect()` forces materialisation. Only call it once per dataframe, and "
                "as late as possible in your pipeline!"
            )
        return self.dataframe.to_numpy()
