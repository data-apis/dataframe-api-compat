from __future__ import annotations

import collections
import secrets
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


null = None
NullType = Type[None]


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


class PolarsColumn(Column[DType]):
    def __init__(self, column: pl.Series, *, hash: str | None = None) -> None:
        # _df only necessary in the lazy case
        # keep track of which dataframe the column came from
        self._series = column
        self._hash = hash

    # In the standard
    def __column_namespace__(self, *, api_version: str | None = None) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def name(self) -> str:
        return self.column.name

    @property
    def column(self) -> pl.Series:
        return self._series

    def __len__(self) -> int:
        return len(self.column)

    @property
    def dtype(self) -> Any:
        return dataframe_api_compat.polars_standard.DTYPE_MAP[
            self.column.dtype  # type: ignore[index]
        ]

    def get_rows(self, indices: Column[Any]) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.take(indices.column))

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsColumn[DType]:
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.column)
        if step is None:
            step = 1
        return PolarsColumn(self.column[start:stop:step])

    def get_rows_by_mask(self, mask: Column[Bool]) -> PolarsColumn[DType]:
        name = self.column.name
        return PolarsColumn(self.column.to_frame().filter(mask.column)[name])

    def get_value(self, row: int) -> Any:
        return self.column[row]

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: Column[DType]) -> PolarsColumn[Bool]:
        if values.dtype != self.dtype:
            raise ValueError(f"`value` has dtype {values.dtype}, expected {self.dtype}")
        return PolarsColumn(self.column.is_in(values.column))

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn[Any]:
        df = self.column.to_frame()
        keys = df.columns
        return PolarsColumn(df.with_row_count().unique(keys).get_column("row_nr"))

    def is_null(self) -> PolarsColumn[Bool]:
        return PolarsColumn(self.column.is_null())

    def is_nan(self) -> PolarsColumn[Bool]:
        return PolarsColumn(self.column.is_nan())

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        # todo: this is wrong!
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
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column == other.column, hash=self._hash)
        return PolarsColumn(self.column == other, hash=self._hash)

    def __ne__(  # type: ignore[override]
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column != other.column, hash=self._hash)
        return PolarsColumn(self.column != other, hash=self._hash)

    def __ge__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column >= other.column, hash=self._hash)
        return PolarsColumn(self.column >= other, hash=self._hash)

    def __gt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            # todo: validate other column's ._df
            return PolarsColumn(self.column > other.column, hash=self._hash)
        return PolarsColumn(self.column > other, hash=self._hash)

    def __le__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column <= other.column, hash=self._hash)
        return PolarsColumn(self.column <= other, hash=self._hash)

    def __lt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column < other.column, hash=self._hash)
        return PolarsColumn(self.column < other, hash=self._hash)

    def __mul__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column * other.column, hash=self._hash)
        return PolarsColumn(self.column * other, hash=self._hash)

    def __floordiv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column // other.column, hash=self._hash)
        return PolarsColumn(self.column // other, hash=self._hash)

    def __truediv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column / other.column, hash=self._hash)
        return PolarsColumn(self.column / other, hash=self._hash)

    def __pow__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        original_type = self.column.dtype
        if isinstance(other, PolarsColumn):
            ret = self.column.pow(other.column)
            if _is_integer_dtype(original_type) and _is_integer_dtype(other.column.dtype):
                if (other.column < 0).any():
                    raise ValueError("Cannot raise integer to negative power")
                ret = ret.cast(original_type)
        else:
            ret = self.column.pow(other)  # type: ignore[arg-type]
            if _is_integer_dtype(original_type) and isinstance(other, int):
                if other < 0:
                    raise ValueError("Cannot raise integer to negative power")
                ret = ret.cast(original_type)
        return PolarsColumn(ret)

    def __mod__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column % other.column)
        return PolarsColumn(self.column % other)

    def __divmod__(
        self,
        other: Column[DType] | Any,
    ) -> tuple[PolarsColumn[Any], PolarsColumn[Any]]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __and__(self, other: Column[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column & other.column, hash=self._hash)
        return PolarsColumn(self.column & other, hash=self._hash)  # type: ignore[operator]

    def __or__(self, other: Column[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column | other.column, hash=self._hash)
        return PolarsColumn(self.column | other, hash=self._hash)  # type: ignore[operator]

    def __invert__(self) -> PolarsColumn[Bool]:
        return PolarsColumn(~self.column, hash=self._hash)

    def __add__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column + other.column, hash=self._hash)
        return PolarsColumn(self.column + other, hash=self._hash)

    def __sub__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column - other.column, hash=self._hash)
        return PolarsColumn(self.column - other, hash=self._hash)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn[Any]:
        df = self.column.to_frame()
        keys = df.columns
        if ascending:
            return PolarsColumn(
                df.with_row_count().sort(keys, descending=False).get_column("row_nr")
            )
        return PolarsColumn(
            df.with_row_count().sort(keys, descending=False).get_column("row_nr")[::-1]
        )

    def fill_nan(self, value: float | NullType) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.fill_nan(value))  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.fill_null(value))

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cumprod())

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cummax())

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cummin())

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy().astype(dtype)

    def rename(self, name: str | None) -> PolarsColumn[DType]:
        if isinstance(self.column, pl.Series):
            return PolarsColumn(self.column.rename(name or ""))
        return PolarsColumn(self.column.alias(name or ""), hash=self._hash)


class PolarsGroupBy(GroupBy):
    def __init__(self, df: pl.DataFrame | pl.LazyFrame, keys: Sequence[str]) -> None:
        for key in keys:
            if key not in df.columns:
                raise KeyError(f"key {key} not present in DataFrame's columns")
        self.df = df
        self.keys = keys

    def size(self) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).count().rename({"count": "size"})
        return PolarsDataFrame(result)

    def any(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").any())
        return PolarsDataFrame(result)

    def all(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").all())
        return PolarsDataFrame(result)

    def min(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result)

    def max(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result)

    def sum(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result)

    def prod(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result)

    def median(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result)

    def mean(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result)

    def std(
        self, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result)

    def var(
        self, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result)


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df
        self._hash = secrets.token_hex(3)

    def _validate_column(self, column) -> None:
        if isinstance(column.column, pl.Expr) and column._hash != self._hash:
            raise ValueError(
                "Column was created from a different dataframe!",
                column._hash,
                self._hash,
            )

    def __dataframe_namespace__(self, *, api_version: str | None = None) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def dataframe(self) -> pl.DataFrame | pl.LazyFrame:
        return self.df

    def shape(self) -> tuple[int, int]:
        return self.df.shape  # type: ignore[union-attr]

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys)

    def get_column_by_name(self, name: str) -> PolarsColumn[DType]:
        # todo: make single-column df so it can work with lazyframe?
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(pl.col(name), hash=self._hash)
        return PolarsColumn(self.df.get_column(name))  # type: ignore[union-attr]

    def get_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names))

    def get_rows(self, indices: Column[Any]) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe[indices.column])

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step])

    def get_rows_by_mask(self, mask: Column[Bool]) -> PolarsDataFrame:
        self._validate_column(mask)
        return PolarsDataFrame(self.df.filter(mask.column))

    def insert(self, loc: int, label: str, value: Column[Any]) -> PolarsDataFrame:
        columns = self.dataframe.columns
        new_columns = columns[:loc] + [label] + columns[loc:]
        df = self.dataframe.with_columns(value.column.alias(label)).select(new_columns)
        return PolarsDataFrame(df)

    def drop_column(self, label: str) -> PolarsDataFrame:
        if not isinstance(label, str):
            raise TypeError(f"Expected str, got: {type(label)}")
        return PolarsDataFrame(self.dataframe.drop(label))

    def rename_columns(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PolarsDataFrame(self.dataframe.rename(dict(mapping)))

    def get_column_names(self) -> Sequence[str]:
        return self.dataframe.columns

    def __eq__(  # type: ignore[override]
        self,
        other: DataFrame | Any,
    ) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__eq__(other.dataframe))  # type: ignore[arg-type]
        return PolarsDataFrame(self.dataframe.__eq__(other))  # type: ignore[arg-type]

    def __ne__(  # type: ignore[override]
        self,
        other: DataFrame,
    ) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__ne__(other.dataframe))  # type: ignore[arg-type]
        return PolarsDataFrame(self.dataframe.__ne__(other))  # type: ignore[arg-type]

    def __ge__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__ge__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__ge__(other))  # type: ignore[operator]

    def __gt__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__gt__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__gt__(other))  # type: ignore[operator]

    def __le__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__le__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__le__(other))  # type: ignore[operator]

    def __lt__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__lt__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__lt__(other))  # type: ignore[operator]

    def __and__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.with_columns(
                    self.dataframe.get_column(col) & other.dataframe.get_column(col)  # type: ignore[union-attr]
                    for col in self.dataframe.columns
                )
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(
                self.dataframe.get_column(col) & other  # type: ignore[operator,union-attr]
                for col in self.dataframe.columns
            )
        )

    def __or__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.with_columns(
                    self.dataframe.get_column(col) | other.dataframe.get_column(col)  # type: ignore[union-attr]
                    for col in self.dataframe.columns
                )
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(
                self.dataframe.get_column(col) | other  # type: ignore[operator, union-attr]
                for col in self.dataframe.columns
            )
        )

    def __add__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__add__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__add__(other))  # type: ignore[operator]

    def __sub__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__sub__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__sub__(other))  # type: ignore[operator]

    def __mul__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__mul__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.__mul__(other))  # type: ignore[operator]

    def __truediv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__truediv__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.__truediv__(other)  # type: ignore[operator]
        )

    def __floordiv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__floordiv__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.__floordiv__(other)  # type: ignore[operator]
        )

    def __pow__(self, other: DataFrame | Any) -> PolarsDataFrame:
        original_type = self.dataframe.schema
        if isinstance(other, PolarsDataFrame):
            ret = self.dataframe.select(
                [pl.col(col).pow(other.dataframe.get_column(col)) for col in self.get_column_names()]  # type: ignore[union-attr]
            )
            for column in self.dataframe.columns:
                if _is_integer_dtype(original_type[column]) and _is_integer_dtype(
                    other.dataframe.get_column(column).dtype  # type: ignore[union-attr]
                ):
                    if (other.dataframe.get_column(column) < 0).any():  # type: ignore[union-attr]
                        raise ValueError("Cannot raise integer to negative power")
                    ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        else:
            ret = self.dataframe.select(
                [
                    pl.col(col).pow(other)  # type: ignore[arg-type]
                    for col in self.get_column_names()
                ]
            )
            for column in self.dataframe.columns:
                if _is_integer_dtype(original_type[column]) and isinstance(other, int):
                    if other < 0:
                        raise ValueError("Cannot raise integer to negative power")
                    ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        return PolarsDataFrame(ret)

    def __mod__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            assert isinstance(self.dataframe, pl.DataFrame) and isinstance(
                other.dataframe, pl.DataFrame
            )
            return PolarsDataFrame(self.dataframe.__mod__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__mod__(other))  # type: ignore[operator]

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PolarsDataFrame, PolarsDataFrame]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    def __invert__(self) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(~pl.col("*")))

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        result = {}
        for column in self.dataframe.columns:
            result[column] = self.dataframe.get_column(column).is_null()  # type: ignore[union-attr]
        return PolarsDataFrame(pl.DataFrame(result))

    def is_nan(self) -> PolarsDataFrame:
        result = {}
        for column in self.dataframe.columns:
            result[column] = self.dataframe.get_column(column).is_nan()  # type: ignore[union-attr]
        return PolarsDataFrame(pl.DataFrame(result))

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").any()))

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").all()))

    def any_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        expr = pl.any_horizontal(pl.col("*"))
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(expr, hash=self._hash)
        return PolarsColumn(self.dataframe.select(expr).get_column("any"))  # type: ignore[union-attr]

    def all_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        expr = pl.all_horizontal(pl.col("*"))
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(expr, hash=self._hash)
        return PolarsColumn(self.dataframe.select(expr).get_column("all"))  # type: ignore[union-attr]

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").min()))

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").max()))

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").sum()))

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").product()))

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").mean()))

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").median()))

    def std(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").std()))

    def var(
        self, *, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").var()))

    def sorted_indices(
        self,
        keys: Sequence[Any] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsColumn[Any]:
        if keys is None:
            keys = self.dataframe.columns
        df = self.dataframe.select(keys)
        if ascending:
            return PolarsColumn(
                df.with_row_count().sort(keys, descending=False).get_column("row_nr")  # type: ignore[union-attr]
            )
        return PolarsColumn(
            df.with_row_count().sort(keys, descending=False).get_column("row_nr")[::-1]  # type: ignore[union-attr]
        )

    def unique_indices(
        self, keys: Sequence[str] | None = None, *, skip_nulls: bool = True
    ) -> PolarsColumn[Any]:
        df = self.dataframe
        if keys is None:
            keys = df.columns
        # TODO support lazyframe
        return PolarsColumn(df.with_row_count().unique(keys).get_column("row_nr"))  # type: ignore[union-attr]

    def fill_nan(
        self,
        value: float | NullType,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.fill_nan(value))  # type: ignore[arg-type]

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
        return PolarsDataFrame(df)

    def to_array_object(self, dtype: str) -> Any:
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        if isinstance(self.dataframe, pl.LazyFrame):
            # todo - remove this? should it raise?
            return self.dataframe.collect().to_numpy().astype(dtype)
        return self.dataframe.to_numpy().astype(dtype)
