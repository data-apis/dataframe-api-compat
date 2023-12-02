from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

import polars as pl

from dataframe_api_compat.utils import validate_comparand

POLARS_VERSION = tuple(int(v) for v in pl.__version__.split("."))

if TYPE_CHECKING:
    from dataframe_api import Column as ColumnT
    from dataframe_api.typing import DType
    from dataframe_api.typing import Namespace
    from dataframe_api.typing import NullType
    from dataframe_api.typing import Scalar
    from typing_extensions import Self

    from dataframe_api_compat.polars_standard.dataframe_object import DataFrame
else:
    ColumnT = object


def _extract_name(expr: pl.Expr | pl.Series, df: DataFrame | None) -> str:
    if isinstance(expr, pl.Expr):
        try:
            return expr.meta.output_name()
        except pl.ComputeError:  # pragma: no cover
            # can remove if/when requiring polars >= 0.19.13
            if df is not None:
                # Unexpected error. Just let it raise.
                raise
            return ""
    return expr.name


class Column(ColumnT):
    def __init__(
        self,
        expr: pl.Expr | pl.Series,
        *,
        df: DataFrame | None,
        api_version: str,
        is_persisted: bool = False,
    ) -> None:
        self._expr = expr
        self._name = _extract_name(expr, df)
        self._df = df
        self._api_version = api_version
        self._is_persisted = is_persisted
        assert is_persisted ^ (df is not None)
        assert is_persisted ^ isinstance(expr, pl.Expr)

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard Column (api_version={self._api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.column` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    def _from_expr(self, expr: pl.Expr | pl.Series) -> Self:
        return self.__class__(
            expr,
            df=self._df,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    def _materialise(self) -> pl.Series:
        if not self._is_persisted:
            msg = "Column is not persisted, please call `.persist()` first.\nNote: `persist` forces computation, use it with care, only when you need to,\nand as late and little as possible."
            raise RuntimeError(
                msg,
            )
        return self._expr  # type: ignore[return-value]

    # In the standard
    def __column_namespace__(self) -> Namespace:  # pragma: no cover
        import dataframe_api_compat

        return dataframe_api_compat.polars_standard.Namespace(
            api_version=self._api_version,
        )

    def _to_scalar(self, value: Any) -> Scalar:
        from dataframe_api_compat.polars_standard.scalar_object import Scalar

        return Scalar(
            value,
            api_version=self._api_version,
            df=self._df,
            is_persisted=self._is_persisted,
        )

    def persist(self) -> Column:
        if self._df is not None:
            assert isinstance(self._df.dataframe, pl.LazyFrame)  # help mypy
            df = self._df.dataframe.select(self._expr).collect()
        else:
            warnings.warn(
                "Calling `.persist` on Column that was already persisted",
                UserWarning,
                stacklevel=2,
            )
            df = pl.select(self._expr)
        column = df.get_column(df.columns[0])
        return Column(
            column,
            df=None,
            api_version=self._api_version,
            is_persisted=True,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def column(self) -> pl.Expr | pl.Series:
        return self._expr

    @property
    def dtype(self) -> DType:
        from dataframe_api_compat.polars_standard import (
            map_polars_dtype_to_standard_dtype,
        )

        if self._df is not None:
            dtype = self._df.dataframe.select(self._expr).schema[self.name]
        else:
            dtype = pl.select(self._expr).schema[self.name]
        return map_polars_dtype_to_standard_dtype(dtype)

    @property
    def parent_dataframe(self) -> DataFrame | None:
        return self._df

    def get_rows(self, indices: Column) -> Column:
        if POLARS_VERSION < (0, 19, 14):
            return self._from_expr(self._expr.take(indices._expr))
        return self._from_expr(self._expr.gather(indices._expr))

    def filter(self, mask: Column) -> Column:
        return self._from_expr(self._expr.filter(mask._expr))  # type: ignore[arg-type]

    def get_value(self, row_number: int) -> Any:
        if POLARS_VERSION < (0, 19, 14):
            result = self._expr.take(row_number)
        else:
            result = self._expr.gather(row_number)
        if isinstance(result, pl.Series):
            return self._to_scalar(result.item())
        return self._to_scalar(result)

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
        if POLARS_VERSION < (0, 19, 14):
            return self._from_expr(self._expr.slice(start, length).take_every(step))
        return self._from_expr(self._expr.slice(start, length).gather_every(step))

    # Binary comparisons

    def __eq__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = validate_comparand(self, other)
        return self._from_expr(self._expr == other)

    def __ne__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = validate_comparand(self, other)
        return self._from_expr(self._expr != other)

    def __ge__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr >= other)

    def __gt__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr > other)

    def __le__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr <= other)

    def __lt__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr < other)

    def __mul__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        res = self._expr * other
        return self._from_expr(res)

    def __rmul__(self, other: Column | Any) -> Column:
        return self.__mul__(other)

    def __floordiv__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr // other)

    def __rfloordiv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __truediv__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        res = self._expr / other
        return self._from_expr(res)

    def __rtruediv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __pow__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        ret = self._expr.pow(other)
        return self._from_expr(ret)

    def __rpow__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr % other)

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

    def __and__(
        self,
        other: Self | bool | Scalar,
    ) -> Self:
        _other = validate_comparand(self, other)
        return self._from_expr(self._expr & _other)

    def __rand__(
        self,
        other: Column | Any | Scalar,
    ) -> Column:
        return self.__and__(other)

    def __or__(
        self,
        other: Self | bool | Scalar,
    ) -> Self:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr | other)  # type: ignore[operator, arg-type]

    def __ror__(self, other: Column | Any | Scalar) -> Column:
        return self.__or__(other)

    def __add__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr + other)

    def __radd__(self, other: Column | Any) -> Column:
        return self.__add__(other)

    def __sub__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        return self._from_expr(self._expr - other)

    def __rsub__(self, other: Column | Any) -> Column:
        return -1 * self.__sub__(other)

    # Unary

    def __invert__(self) -> Column:
        return self._from_expr(~self._expr)

    # Reductions

    def any(self, *, skip_nulls: bool | Scalar = True) -> Scalar:
        return self._to_scalar(self._expr.any())

    def all(self, *, skip_nulls: bool | Scalar = True) -> Scalar:
        return self._to_scalar(self._expr.all())

    def min(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.min())

    def max(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.max())

    def sum(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.sum())

    def prod(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.product())

    def mean(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.mean())

    def median(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.median())

    def std(
        self,
        *,
        correction: float | Scalar = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.std())

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> Scalar:
        return self._to_scalar(self._expr.var())

    def __len__(self) -> int:
        ser = self._materialise()
        return len(ser)

    # Transformations

    def is_null(self) -> Self:
        return self._from_expr(self._expr.is_null())

    def is_nan(self) -> Column:
        return self._from_expr(self._expr.is_nan())

    def sort(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        expr = self._expr.sort(descending=not ascending)
        return self._from_expr(expr)

    def is_in(self, values: Self) -> Self:
        return self._from_expr(self._expr.is_in(values._expr))  # type: ignore[arg-type]

    def sorted_indices(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        expr = self._expr.arg_sort(descending=not ascending)
        return self._from_expr(expr)

    def unique_indices(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Self:
        raise NotImplementedError

    def fill_nan(
        self,
        value: float | NullType | Scalar,
    ) -> Column:
        _value = validate_comparand(self, value)
        if isinstance(_value, self.__column_namespace__().NullType):
            return self._from_expr(self._expr.fill_nan(pl.lit(None)))
        return self._from_expr(self._expr.fill_nan(_value))

    def fill_null(self, value: Any) -> Column:
        value = validate_comparand(self, value)
        return self._from_expr(self._expr.fill_null(value))

    def cumulative_sum(self, *, skip_nulls: bool | Scalar = True) -> Column:
        if POLARS_VERSION < (0, 19, 14):
            return self._from_expr(self._expr.cumsum())
        return self._from_expr(self._expr.cum_sum())

    def cumulative_prod(self, *, skip_nulls: bool | Scalar = True) -> Column:
        if POLARS_VERSION < (0, 19, 14):
            return self._from_expr(self._expr.cumprod())
        return self._from_expr(self._expr.cum_prod())

    def cumulative_max(self, *, skip_nulls: bool | Scalar = True) -> Column:
        if POLARS_VERSION < (0, 19, 14):
            return self._from_expr(self._expr.cummax())
        return self._from_expr(self._expr.cum_max())

    def cumulative_min(self, *, skip_nulls: bool | Scalar = True) -> Column:
        if POLARS_VERSION < (0, 19, 14):
            return self._from_expr(self._expr.cummin())
        return self._from_expr(self._expr.cum_min())

    def rename(self, name: str | Scalar) -> Column:
        _name = validate_comparand(self, name)
        return self._from_expr(self._expr.alias(_name))

    def shift(self, offset: int | Scalar) -> Column:
        _offset = validate_comparand(self, offset)
        return self._from_expr(self._expr.shift(_offset))

    # Conversions

    def to_array(self) -> Any:
        ser = self._materialise()
        return ser.to_numpy()

    # --- temporal methods ---

    def year(self) -> Column:
        return self._from_expr(self._expr.dt.year())

    def month(self) -> Column:
        return self._from_expr(self._expr.dt.month())

    def day(self) -> Column:
        return self._from_expr(self._expr.dt.day())

    def hour(self) -> Column:
        return self._from_expr(self._expr.dt.hour())

    def minute(self) -> Column:
        return self._from_expr(self._expr.dt.minute())

    def second(self) -> Column:
        return self._from_expr(self._expr.dt.second())

    def microsecond(self) -> Column:
        return self._from_expr(self._expr.dt.microsecond())

    def nanosecond(self) -> Column:
        return self._from_expr(self._expr.dt.nanosecond())

    def iso_weekday(self) -> Column:
        return self._from_expr(self._expr.dt.weekday())

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
        return self._from_expr(self._expr.dt.truncate(frequency))

    def unix_timestamp(
        self,
        *,
        time_unit: str | Scalar = "s",
    ) -> Column:
        _time_unit = validate_comparand(self, time_unit)
        if _time_unit != "s":
            return self._from_expr(self._expr.dt.timestamp(time_unit=_time_unit))
        return self._from_expr(self._expr.dt.timestamp(time_unit="ms") // 1000)
