from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

import polars as pl

if TYPE_CHECKING:
    from dataframe_api import Column as ColumnT
    from dataframe_api.typing import DType
    from dataframe_api.typing import Namespace
    from dataframe_api.typing import NullType
    from typing_extensions import Self

    from dataframe_api_compat.polars_standard.dataframe_object import DataFrame
    from dataframe_api_compat.polars_standard.scalar_object import Scalar
else:
    ColumnT = object


class Column(ColumnT):
    def __init__(
        self,
        expr: pl.Expr,
        *,
        df: DataFrame | None,
        api_version: str,
    ) -> None:
        self.expr = expr
        self.df = df
        self.api_version = api_version
        try:
            self._name = expr.meta.output_name()
        except pl.ComputeError:  # pragma: no cover
            # can remove if/when requiring polars >= 0.19.13
            if df is not None:
                # Unexpected error. Just let it raise.
                raise
            self._name = ""

    def __repr__(self) -> str:  # pragma: no cover
        column = self.materialise("Column.__repr__")
        return column.__repr__()

    def _from_expr(self, expr: pl.Expr) -> Self:
        return self.__class__(expr, df=self.df, api_version=self.api_version)

    # In the standard
    def __column_namespace__(self) -> Namespace:  # pragma: no cover
        import dataframe_api_compat

        return dataframe_api_compat.polars_standard.Namespace(
            api_version=self.api_version,
        )

    def _validate_comparand(self, other: Self | Any) -> Self | Any:
        from dataframe_api_compat.polars_standard.scalar_object import Scalar

        if isinstance(other, Scalar):
            if id(self.df) != id(other.df):
                msg = "Columns/scalars are from different dataframes"
                raise ValueError(msg)
            return other.value
        if isinstance(other, Column):
            if id(self.df) != id(other.df):
                msg = "Columns are from different dataframes"
                raise ValueError(msg)
            return other.expr
        return other

    def materialise(self, method: str) -> pl.Series:
        if self.df is not None:
            ser = self.df.materialise_expression(self.expr)
        else:
            df = pl.select(self.expr)
            ser = df.get_column(df.columns[0])
        return ser

    def _to_scalar(self, value: pl.Expr) -> Scalar:
        from dataframe_api_compat.polars_standard.scalar_object import Scalar

        return Scalar(value, api_version=self.api_version, df=self.df)

    @property
    def name(self) -> str:
        return self._name

    @property
    def column(self) -> pl.Expr | pl.Series:
        if self.df is None:
            # self-standing column
            df = pl.select(self.expr)
            return df.get_column(df.columns[0])
        elif self.df.is_persisted:
            return self.df.materialise_expression(self.expr)
        return self.expr  # pragma: no cover (probably unneeded?)

    @property
    def dtype(self) -> DType:
        from dataframe_api_compat.polars_standard import (
            map_polars_dtype_to_standard_dtype,
        )

        if self.df is not None:
            dtype = self.df.dataframe.select(self.expr).schema[self.name]
        else:
            dtype = pl.select(self.expr).schema[self.name]
        return map_polars_dtype_to_standard_dtype(dtype)

    @property
    def parent_dataframe(self) -> DataFrame | None:
        return self.df

    def get_rows(self, indices: Column) -> Column:
        return self._from_expr(self.expr.take(indices.expr))

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> Column:
        if start is None:
            start = 0
        length = None if stop is None else stop - start
        if step is None:
            step = 1
        return self._from_expr(self.expr.slice(start, length).take_every(step))

    def filter(self, mask: Column) -> Column:
        return self._from_expr(self.expr.filter(mask.expr))

    def get_value(self, row_number: int) -> Any:
        ser = self.materialise("Column.get_value")
        return ser[row_number]

    def to_array(self) -> Any:
        ser = self.materialise("Column.to_array")
        return ser.to_numpy()

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    def is_in(self, values: Self) -> Self:
        return self._from_expr(self.expr.is_in(values.expr))

    def unique_indices(self, *, skip_nulls: bool = True) -> Self:
        raise NotImplementedError

    def is_null(self) -> Self:
        return self._from_expr(self.expr.is_null())

    def is_nan(self) -> Column:
        return self._from_expr(self.expr.is_nan())

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> Scalar:  # type: ignore[override]  # todo, fix
        return self._to_scalar(self.expr.any())

    def all(self, *, skip_nulls: bool = True) -> Scalar:  # type: ignore[override]  # todo fix
        return self._to_scalar(self.expr.all())

    def min(self, *, skip_nulls: bool = True) -> Scalar:
        return self._to_scalar(self.expr.min())

    def max(self, *, skip_nulls: bool = True) -> Scalar:
        return self._to_scalar(self.expr.max())

    def sum(self, *, skip_nulls: bool = True) -> Scalar:
        return self._to_scalar(self.expr.sum())

    def prod(self, *, skip_nulls: bool = True) -> Scalar:
        return self._to_scalar(self.expr.product())

    def mean(self, *, skip_nulls: bool = True) -> Scalar:
        return self._to_scalar(self.expr.mean())

    def median(self, *, skip_nulls: bool = True) -> Scalar:
        return self._to_scalar(self.expr.median())

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> Scalar:
        return self._to_scalar(self.expr.std())

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> Scalar:
        return self._to_scalar(self.expr.var())

    # Binary

    def __add__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr + other)

    def __radd__(self, other: Column | Any) -> Column:
        return self.__add__(other)

    def __sub__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr - other)

    def __rsub__(self, other: Column | Any) -> Column:
        return -1 * self.__sub__(other)

    def __eq__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = self._validate_comparand(other)
        return self._from_expr(self.expr == other)

    def __ne__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = self._validate_comparand(other)
        return self._from_expr(self.expr != other)

    def __ge__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr >= other)

    def __gt__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr > other)

    def __le__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr <= other)

    def __lt__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr < other)

    def __mul__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        res = self.expr * other
        return self._from_expr(res)

    def __rmul__(self, other: Column | Any) -> Column:
        return self.__mul__(other)

    def __floordiv__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr // other)

    def __rfloordiv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __truediv__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        res = self.expr / other
        return self._from_expr(res)

    def __rtruediv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __pow__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        ret = self.expr.pow(other)  # type: ignore[arg-type]
        return self._from_expr(ret)

    def __rpow__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Column | Any) -> Column:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr % other)

    def __rmod__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __divmod__(
        self,
        other: Column | Any,
    ) -> tuple[Column, Column]:
        # validation happens in the deferred calls anyway
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: Self | bool) -> Self:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr & other)  # type: ignore[arg-type]

    def __rand__(self, other: Column | Any) -> Column:
        return self.__and__(other)

    def __or__(self, other: Self | bool) -> Self:
        other = self._validate_comparand(other)
        return self._from_expr(self.expr | other)  # type: ignore[arg-type]

    def __ror__(self, other: Column | Any) -> Column:
        return self.__or__(other)

    def __invert__(self) -> Column:
        return self._from_expr(~self.expr)

    def sorted_indices(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        expr = self.expr.arg_sort(descending=not ascending)
        return self._from_expr(expr)

    def sort(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        expr = self.expr.sort(descending=not ascending)
        return self._from_expr(expr)

    def fill_nan(self, value: float | NullType) -> Column:
        if isinstance(value, self.__column_namespace__().NullType):
            return self._from_expr(self.expr.fill_nan(pl.lit(None)))
        return self._from_expr(self.expr.fill_nan(value))

    def fill_null(self, value: Any) -> Column:
        value = self._validate_comparand(value)
        return self._from_expr(self.expr.fill_null(value))

    def cumulative_sum(self, *, skip_nulls: bool = True) -> Column:
        return self._from_expr(self.expr.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool = True) -> Column:
        return self._from_expr(self.expr.cumprod())

    def cumulative_max(self, *, skip_nulls: bool = True) -> Column:
        return self._from_expr(self.expr.cummax())

    def cumulative_min(self, *, skip_nulls: bool = True) -> Column:
        return self._from_expr(self.expr.cummin())

    def rename(self, name: str) -> Column:
        return self._from_expr(self.expr.alias(name))

    def __len__(self) -> int:
        ser = self.materialise("Column.__len__")
        return len(ser)

    def shift(self, offset: int) -> Column:
        return self._from_expr(self.expr.shift(offset))

    # --- temporal methods ---

    def year(self) -> Column:
        return self._from_expr(self.expr.dt.year())

    def month(self) -> Column:
        return self._from_expr(self.expr.dt.month())

    def day(self) -> Column:
        return self._from_expr(self.expr.dt.day())

    def hour(self) -> Column:
        return self._from_expr(self.expr.dt.hour())

    def minute(self) -> Column:
        return self._from_expr(self.expr.dt.minute())

    def second(self) -> Column:
        return self._from_expr(self.expr.dt.second())

    def microsecond(self) -> Column:
        return self._from_expr(self.expr.dt.microsecond())

    def nanosecond(self) -> Column:
        return self._from_expr(self.expr.dt.nanosecond())

    def iso_weekday(self) -> Column:
        return self._from_expr(self.expr.dt.weekday())

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
        return self._from_expr(self.expr.dt.truncate(frequency))

    def unix_timestamp(
        self,
        *,
        time_unit: Literal["s", "ms", "us"] = "s",
    ) -> Column:
        if time_unit != "s":
            return self._from_expr(self.expr.dt.timestamp(time_unit=time_unit))
        return self._from_expr(self.expr.dt.timestamp(time_unit="ms") // 1000)
