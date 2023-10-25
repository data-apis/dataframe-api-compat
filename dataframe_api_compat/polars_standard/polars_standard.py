from __future__ import annotations

import collections
from typing import Any
from typing import cast
from typing import Literal
from typing import NoReturn
from typing import TYPE_CHECKING

import polars as pl
from dataframe_api import DataFrame
from dataframe_api import GroupBy

import dataframe_api_compat.polars_standard

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

    from dataframe_api.typing import DType


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


class PolarsScalar:
    def __init__(self, value: pl.Expr, api_version: str, df: PolarsDataFrame):
        self.value = value
        self._api_version = api_version
        self.df = df

    def __bool__(self) -> bool:
        self.df.validate_is_collected("Scalar.__bool__")
        return self.df.materialise(self.value).item().__bool__()

    def __int__(self) -> int:
        self.df.validate_is_collected("Scalar.__int__")
        return self.df.materialise(self.value).item().__int__()

    def __float__(self) -> float:
        self.df.validate_is_collected("Scalar.__float__")
        return self.df.materialise(self.value).item().__float__()


class PolarsGroupBy(GroupBy):
    def __init__(self, df: pl.LazyFrame, keys: Sequence[str], api_version: str) -> None:
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

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.group_by(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").any())
        return PolarsDataFrame(result, api_version=self._api_version)

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.group_by(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").all())
        return PolarsDataFrame(result, api_version=self._api_version)

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result, api_version=self._api_version)

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result, api_version=self._api_version)

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result, api_version=self._api_version)

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result, api_version=self._api_version)

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result, api_version=self._api_version)

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result, api_version=self._api_version)

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result, api_version=self._api_version)

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result, api_version=self._api_version)


class PolarsColumn:
    def __init__(
        self,
        expr: pl.Expr,
        *,
        df: PolarsDataFrame,
        api_version: str,
    ) -> None:
        self.expr = expr
        self.df = df
        self.api_version = api_version
        self._name = expr.meta.output_name()

    def _from_expr(self, expr: pl.Expr):
        return self.__class__(expr, df=self.df, api_version=self.api_version)

    # In the standard
    def __column_namespace__(self) -> Any:  # pragma: no cover
        return dataframe_api_compat.polars_standard

    def _validate_comparand(self, other: PolarsColumn | Any) -> PolarsColumn | Any:
        if isinstance(other, PolarsScalar):
            if id(self.df) != id(other.df):
                raise ValueError("Columns/scalars are from different dataframes")
            return other.value
        if isinstance(other, PolarsColumn):
            if id(self.df) != id(other.df):
                raise ValueError("Columns are from different dataframes")
            return other.expr
        return other

    def _to_scalar(self, value: pl.Expr):
        return PolarsScalar(value, api_version=self.api_version, df=self.df)

    @property
    def name(self):
        return self._name

    def get_rows(self, indices: PolarsColumn) -> PolarsColumn:
        return self._from_expr(self.expr.take(indices.expr))

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsColumn:
        if start is None:
            start = 0
        length = None if stop is None else stop - start
        if step is None:
            step = 1
        return self._from_expr(self.expr.slice(start, length).take_every(step))

    def filter(self, mask: PolarsColumn) -> PolarsColumn:
        return self._from_expr(self.expr.filter(mask.expr))

    def get_value(self, row: int) -> Any:
        df = self.df.validate_is_collected("Column.get_value")
        return df.select(self.expr)[self.name][row]

    def to_array(self) -> Any:
        df = self.df.validate_is_collected("Column.to_array")
        return df.select(self.expr)[self.name].to_numpy()

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsColumn) -> PolarsColumn:  # type: ignore[override]
        return self._from_expr(self.expr.is_in(values.expr))

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn:
        raise NotImplementedError()

    def is_null(self) -> PolarsColumn:
        return self._from_expr(self.expr.is_null())

    def is_nan(self) -> PolarsColumn:
        return self._from_expr(self.expr.is_nan())

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        return self._to_scalar(self.expr.any())

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        return self._to_scalar(self.expr.all())

    def min(self, *, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.min())

    def max(self, *, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.max())

    def sum(self, *, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.sum())

    def prod(self, *, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.product())

    def mean(self, *, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.mean())

    def median(self, *, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.median())

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.std())

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        return self._to_scalar(self.expr.var())

    # Binary

    def __eq__(self, other: PolarsColumn | Any) -> PolarsColumn:  # type: ignore[override]
        other = self._validate_comparand(other)
        return self._from_expr(self.expr == other)

    def __ne__(self, other: PolarsColumn | Any) -> PolarsColumn:  # type: ignore[override]
        other = self._validate_comparand(other)
        return self._from_expr(self.expr != other)

    def __ge__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr >= other)

    def __gt__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr > other)

    def __le__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr <= other)

    def __lt__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr < other)

    def __mul__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        res = self.expr * other
        return self._from_expr(res)

    def __floordiv__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr // other)

    def __truediv__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        res = self.expr / other
        return self._from_expr(res)

    def __pow__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        ret = self.expr.pow(other)  # type: ignore[arg-type]
        return self._from_expr(ret)

    def __mod__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr % other)

    def __divmod__(
        self,
        other: PolarsColumn | Any,
    ) -> tuple[PolarsColumn, PolarsColumn]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: PolarsColumn | bool) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr & other)  # type: ignore[operator]

    def __or__(self, other: PolarsColumn | bool) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr | other)

    def __invert__(self) -> PolarsColumn:
        return self._from_expr(~self.expr)

    def __add__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr + other)

    def __sub__(self, other: PolarsColumn | Any) -> PolarsColumn:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr - other)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn:
        expr = self.expr.arg_sort(descending=not ascending)
        return self._from_expr(expr)

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn:
        expr = self.expr.sort(descending=not ascending)
        return self._from_expr(expr)

    def fill_nan(self, value: float | NullType) -> PolarsColumn:
        return self._from_expr(self.expr.fill_nan(value))  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsColumn:
        return self._from_expr(self.expr.fill_null(value))

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self.expr.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self.expr.cumprod())

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self.expr.cummax())

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return self._from_expr(self.expr.cummin())

    def rename(self, name: str) -> PolarsColumn:
        return self._from_expr(self.expr.alias(name))

    @property
    def dt(self) -> ColumnDatetimeAccessor:
        """
        Return accessor with functions which work on temporal dtypes.
        """
        return ColumnDatetimeAccessor(self)

    def __len__(self) -> int:
        df = self.df.validate_is_collected("Column.__len__")
        return len(df.select(self.expr)[self.name])


class ColumnDatetimeAccessor:
    def __init__(self, column: PolarsColumn) -> None:
        self.column = column
        self._api_version = column.api_version

    def from_expr(self, expr: pl.Expr) -> PolarsColumn:
        return self.column.__class__(
            expr, df=self.column.df, api_version=self._api_version
        )

    def year(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.year())

    def month(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.month())

    def day(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.day())

    def hour(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.hour())

    def minute(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.minute())

    def second(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.second())

    def microsecond(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.microsecond())

    def iso_weekday(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.weekday())

    def floor(self, frequency: str) -> PolarsColumn:
        frequency = (
            frequency.replace("day", "d")
            .replace("hour", "h")
            .replace("minute", "m")
            .replace("second", "s")
            .replace("millisecond", "ms")
            .replace("microsecond", "us")
            .replace("nanosecond", "ns")
        )
        return self.from_expr(self.column.expr.dt.truncate(frequency))

    def unix_timestamp(self) -> PolarsColumn:
        return self.from_expr(self.column.expr.dt.timestamp("ms") // 1000)


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.LazyFrame | pl.DataFrame, api_version: str) -> None:
        self.df = df
        if api_version not in SUPPORTED_VERSIONS:
            raise AssertionError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                "Try updating dataframe-api-compat?"
            )
        self.api_version = api_version

    @property
    def _is_collected(self):
        return isinstance(self.dataframe, pl.DataFrame)

    def materialise(self, expr: pl.Expr) -> pl.Series:
        df = cast(pl.DataFrame, self.dataframe)
        return df.select(expr).get_column(expr.meta.output_name())

    def validate_is_collected(self, method: str) -> pl.DataFrame:
        if not self._is_collected:
            raise ValueError(
                f"Method {method} requires you to call `.collect` first.\n"
                "\n"
                "Note: `.collect` forces materialisation in lazy libraries and "
                "so should be called as late as possible in your pipeline, and "
                "only once per dataframe."
            )
        return self.dataframe  # type: ignore

    def col(self, value: str) -> PolarsColumn:  # type: ignore[override]
        return PolarsColumn(pl.col(value), df=self, api_version=self.api_version)

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
    def dataframe(self) -> pl.LazyFrame | pl.DataFrame:
        return self.df

    def group_by(self, *keys: str) -> PolarsGroupBy:
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsGroupBy(self.dataframe, list(keys), api_version=self.api_version)
        return PolarsGroupBy(
            self.dataframe.lazy(), list(keys), api_version=self.api_version
        )

    def select(self, *columns: str) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.df.select(list(columns)), api_version=self.api_version
        )

    def get_rows(self, indices: PolarsColumn) -> PolarsDataFrame:  # type: ignore[override]
        self._validate_column(indices)
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices.expr)),
            api_version=self.api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self.api_version)

    def _validate_column(self, column: PolarsColumn) -> None:
        if id(self) != id(column.df):
            raise ValueError("Column is from a different dataframe")

    def filter(self, mask: PolarsColumn) -> PolarsDataFrame:  # type: ignore[override]
        self._validate_column(mask)
        return PolarsDataFrame(self.df.filter(mask.expr), api_version=self.api_version)

    def assign(self, *columns: PolarsColumn) -> PolarsDataFrame:  # type: ignore[override]
        new_columns: list[pl.Expr] = []
        for col in columns:
            self._validate_column(col)
            new_columns.append(col.expr)
        df = self.dataframe.with_columns(new_columns)
        return PolarsDataFrame(df, api_version=self.api_version)

    def drop_columns(self, *labels: str) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.drop(labels), api_version=self.api_version)

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):  # type: ignore
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsDataFrame(
            self.dataframe.rename(dict(mapping)), api_version=self.api_version
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
            api_version=self.api_version,
        )

    def __ne__(  # type: ignore[override]
        self,
        other: Any,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
            api_version=self.api_version,
        )

    def __ge__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
            api_version=self.api_version,
        )

    def __gt__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
            api_version=self.api_version,
        )

    def __le__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
            api_version=self.api_version,
        )

    def __lt__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
            api_version=self.api_version,
        )

    def __and__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") & other),
            api_version=self.api_version,
        )

    def __or__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self.api_version,
        )

    def __add__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self.api_version,
        )

    def __sub__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self.api_version,
        )

    def __mul__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self.api_version,
        )

    def __truediv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self.api_version,
        )

    def __floordiv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__floordiv__(other)),
            api_version=self.api_version,
        )

    def __pow__(self, other: Any) -> PolarsDataFrame:
        original_type = self.dataframe.schema
        ret = self.dataframe.select([pl.col(col).pow(other) for col in self.column_names])
        for column in self.dataframe.columns:
            if _is_integer_dtype(original_type[column]) and isinstance(other, int):
                if other < 0:  # pragma: no cover (todo)
                    raise ValueError("Cannot raise integer to negative power")
                ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        return PolarsDataFrame(ret, api_version=self.api_version)

    def __mod__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") % other),
            api_version=self.api_version,
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
            quotient_df, api_version=self.api_version
        ), PolarsDataFrame(remainder_df, api_version=self.api_version)

    def __invert__(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(~pl.col("*")), api_version=self.api_version
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").is_null()),
            api_version=self.api_version,
        )

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsDataFrame(df, api_version=self.api_version)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").any()), api_version=self.api_version
        )

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").all()), api_version=self.api_version
        )

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").min()), api_version=self.api_version
        )

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").max()), api_version=self.api_version
        )

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").sum()), api_version=self.api_version
        )

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").product()), api_version=self.api_version
        )

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").mean()), api_version=self.api_version
        )

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").median()), api_version=self.api_version
        )

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").std()), api_version=self.api_version
        )

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").var()), api_version=self.api_version
        )

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsDataFrame:
        if not keys:
            keys = tuple(self.dataframe.columns)
        # TODO: what if there's multiple `ascending`?
        return PolarsDataFrame(
            self.dataframe.sort(list(keys), descending=not ascending),
            api_version=self.api_version,
        )

    def fill_nan(
        self,
        value: float | NullType,
    ) -> PolarsDataFrame:
        if isinstance(value, Null):
            value = None  # type: ignore
        return PolarsDataFrame(self.dataframe.fill_nan(value), api_version=self.api_version)  # type: ignore[arg-type]

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
        return PolarsDataFrame(df, api_version=self.api_version)

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
            other_df,  # type: ignore
            left_on=left_on,
            right_on=right_on,
            how=how,
        )
        result = result.rename({f"{i}_tmp": i for i in extra_right_keys})

        return PolarsDataFrame(result, api_version=self.api_version)

    def collect(self) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsDataFrame(self.dataframe.collect(), api_version=self.api_version)
        raise ValueError("DataFrame was already collected")

    def to_array(self, dtype: DType) -> Any:
        df = self.validate_is_collected("DataFrame.to_array")
        return df.to_numpy()
