from __future__ import annotations

import collections
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
)

import polars as pl

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
NullType = type[None]


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
    def __init__(self, column: pl.Series, *, name: str | None = None) -> None:
        self._series = column
        self._name = name

    # In the standard
    def __column_namespace__(self, *, api_version: str | None = None) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def name(self) -> str | None:
        return self._name

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
        return PolarsColumn(df.with_row_count().unique(keys)["row_nr"])

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
            return PolarsColumn(self.column == other.column)
        return PolarsColumn(self.column == other)

    def __ne__(  # type: ignore[override]
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column != other.column)
        return PolarsColumn(self.column != other)

    def __ge__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column >= other.column)
        return PolarsColumn(self.column >= other)

    def __gt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column > other.column)
        return PolarsColumn(self.column > other)

    def __le__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column <= other.column)
        return PolarsColumn(self.column <= other)

    def __lt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column < other.column)
        return PolarsColumn(self.column < other)

    def __mul__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column * other.column)
        return PolarsColumn(self.column * other)

    def __floordiv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column // other.column)
        return PolarsColumn(self.column // other)

    def __truediv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column / other.column)
        return PolarsColumn(self.column / other)

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
            return PolarsColumn(self.column & other.column)
        return PolarsColumn(self.column & other)  # type: ignore[operator]

    def __or__(self, other: Column[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column | other.column)
        return PolarsColumn(self.column | other)  # type: ignore[operator]

    def __invert__(self) -> PolarsColumn[Bool]:
        return PolarsColumn(~self.column)

    def __add__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column + other.column)
        return PolarsColumn(self.column + other)

    def __sub__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column - other.column)
        return PolarsColumn(self.column - other)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn[Any]:
        df = self.column.to_frame()
        keys = df.columns
        if ascending:
            return PolarsColumn(
                df.with_row_count().sort(keys, descending=False)["row_nr"]
            )
        return PolarsColumn(
            df.with_row_count().sort(keys, descending=False)["row_nr"][::-1]
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


class PolarsGroupBy(GroupBy):
    def __init__(self, df: pl.DataFrame, keys: Sequence[str]) -> None:
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
    def __init__(self, df: pl.DataFrame) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df

    def __dataframe_namespace__(self, *, api_version: str | None = None) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def dataframe(self) -> pl.DataFrame:
        return self.df

    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys)

    def get_column_by_name(self, name: str) -> PolarsColumn[DType]:
        return PolarsColumn(self.df[name])

    def get_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names))

    def get_rows(self, indices: Column[Any]) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe[indices.column])

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.dataframe)
        if step is None:
            step = 1
        return PolarsDataFrame(self.df[start:stop:step])

    def get_rows_by_mask(self, mask: Column[Bool]) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask.column))

    def insert(self, loc: int, label: str, value: Column[Any]) -> PolarsDataFrame:
        df = self.df.clone()
        df.insert_at_idx(loc, pl.Series(label, value.column))
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
            return PolarsDataFrame(self.dataframe.__eq__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__eq__(other))

    def __ne__(  # type: ignore[override]
        self,
        other: DataFrame,
    ) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__ne__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__ne__(other))

    def __ge__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__ge__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__ge__(other))

    def __gt__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__gt__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__gt__(other))

    def __le__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__le__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__le__(other))

    def __lt__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__lt__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__lt__(other))

    def __and__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.with_columns(
                    self.dataframe[col] & other.dataframe[col]
                    for col in self.dataframe.columns
                )
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(
                self.dataframe[col] & other  # type: ignore[operator]
                for col in self.dataframe.columns
            )
        )

    def __or__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.with_columns(
                    self.dataframe[col] | other.dataframe[col]
                    for col in self.dataframe.columns
                )
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(
                self.dataframe[col] | other  # type: ignore[operator]
                for col in self.dataframe.columns
            )
        )

    def __add__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__add__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__add__(other))  # type: ignore[operator]

    def __sub__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__sub__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__sub__(other))  # type: ignore[operator]

    def __mul__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__mul__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__mul__(other))  # type: ignore[operator]

    def __truediv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__truediv__(other.dataframe))
        return PolarsDataFrame(
            self.dataframe.__truediv__(other)  # type: ignore[operator]
        )

    def __floordiv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__floordiv__(other.dataframe))
        return PolarsDataFrame(
            self.dataframe.__floordiv__(other)  # type: ignore[operator]
        )

    def __pow__(self, other: DataFrame | Any) -> PolarsDataFrame:
        original_type = self.dataframe.schema
        if isinstance(other, PolarsDataFrame):
            ret = self.dataframe.select(
                [pl.col(col).pow(other.dataframe[col]) for col in self.get_column_names()]
            )
            for column in self.dataframe.columns:
                if _is_integer_dtype(original_type[column]) and _is_integer_dtype(
                    other.dataframe[column].dtype
                ):
                    if (other.dataframe[column] < 0).any():
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
            result[column] = self.dataframe[column].is_null()
        return PolarsDataFrame(pl.DataFrame(result))

    def is_nan(self) -> PolarsDataFrame:
        result = {}
        for column in self.dataframe.columns:
            result[column] = self.dataframe[column].is_nan()
        return PolarsDataFrame(pl.DataFrame(result))

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").any()))

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").all()))

    def any_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        return PolarsColumn(self.dataframe.select(pl.any_horizontal(pl.col("*")))["any"])

    def all_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        return PolarsColumn(self.dataframe.select(pl.all_horizontal(pl.col("*")))["all"])

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
                df.with_row_count().sort(keys, descending=False)["row_nr"]
            )
        return PolarsColumn(
            df.with_row_count().sort(keys, descending=False)["row_nr"][::-1]
        )

    def unique_indices(
        self, keys: Sequence[str] | None = None, *, skip_nulls: bool = True
    ) -> PolarsColumn[Any]:
        df = self.dataframe
        if keys is None:
            keys = df.columns
        return PolarsColumn(df.with_row_count().unique(keys)["row_nr"])

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
        return self.dataframe.to_numpy().astype(dtype)
