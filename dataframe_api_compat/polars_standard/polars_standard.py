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


class PolarsColumn(Column[DType]):
    def __init__(
        self,
        column: pl.Series | pl.Expr,
        *,
        dtype: Any,
        id_: int | None,  # | None = None,
        method: str | None = None,
    ) -> None:
        if column is NotImplemented:
            raise NotImplementedError("operation not implemented")
        self._series = column
        self._dtype = dtype
        # keep track of which dataframe the column came from
        self._id = id_
        # keep track of which method this was called from
        self._method = method
        if isinstance(column, pl.Series):
            # just helps with defensiveness
            assert column.dtype == dtype

    def _validate_column(self, column: PolarsColumn[Any]) -> None:
        if isinstance(column.column, pl.Expr) and column._id != self._id:
            raise ValueError(
                "Column was created from a different dataframe!",
                column._id,
                self._id,
            )

    # In the standard
    def __column_namespace__(self, *, api_version: str | None = None) -> Any:
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
            self.column.take(indices.column), dtype=self._dtype, id_=self._id
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
        return PolarsColumn(self.column[start:stop:step], dtype=self._dtype, id_=self._id)

    def get_rows_by_mask(self, mask: PolarsColumn[Bool]) -> PolarsColumn[DType]:  # type: ignore[override]
        self._validate_column(mask)
        return PolarsColumn(
            self.column.filter(mask.column), dtype=self._dtype, id_=self._id  # type: ignore[arg-type]
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
            self.column.is_in(values.column), dtype=pl.Boolean(), id_=self._id  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn[Any]:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("unique_indices not implemented for lazy columns")
        df = self.column.to_frame()
        keys = df.columns
        return PolarsColumn(
            df.with_row_count().unique(keys).get_column("row_nr"),
            dtype=pl.UInt32(),
            id_=self._id,
        )

    def is_null(self) -> PolarsColumn[Bool]:
        return PolarsColumn(self.column.is_null(), dtype=pl.Boolean(), id_=self._id)

    def is_nan(self) -> PolarsColumn[Bool]:
        return PolarsColumn(self.column.is_nan(), dtype=pl.Boolean(), id_=self._id)

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
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").min())
                .schema["a"]
            )
            return PolarsColumn(self.column.min(), id_=self._id, dtype=res_dtype)
        return self.column.min()

    def max(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").max())
                .schema["a"]
            )
            return PolarsColumn(self.column.max(), id_=self._id, dtype=res_dtype)
        return self.column.max()

    def sum(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").sum())
                .schema["a"]
            )
            return PolarsColumn(self.column.sum(), id_=self._id, dtype=res_dtype)
        return self.column.sum()

    def prod(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").product())
                .schema["a"]
            )
            return PolarsColumn(self.column.product(), id_=self._id, dtype=res_dtype)
        return self.column.product()

    def mean(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").mean())
                .schema["a"]
            )
            return PolarsColumn(self.column.mean(), id_=self._id, dtype=res_dtype)
        return self.column.mean()

    def median(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").median())
                .schema["a"]
            )
            return PolarsColumn(self.column.median(), id_=self._id, dtype=res_dtype)
        return self.column.median()

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").std())
                .schema["a"]
            )
            return PolarsColumn(self.column.std(), id_=self._id, dtype=res_dtype)
        return self.column.std()

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        if isinstance(self.column, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").var())
                .schema["a"]
            )
            return PolarsColumn(self.column.var(), id_=self._id, dtype=res_dtype)
        return self.column.var()

    def __eq__(  # type: ignore[override]
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column == other.column, dtype=pl.Boolean(), id_=self._id
            )
        return PolarsColumn(self.column == other, dtype=pl.Boolean(), id_=self._id)

    def __ne__(  # type: ignore[override]
        self, other: Column[DType] | Any
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column != other.column, dtype=pl.Boolean(), id_=self._id
            )
        return PolarsColumn(self.column != other, dtype=pl.Boolean(), id_=self._id)

    def __ge__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(
                self.column >= other.column, dtype=pl.Boolean(), id_=self._id
            )
        return PolarsColumn(self.column >= other, dtype=pl.Boolean(), id_=self._id)

    def __gt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column > other.column, id_=self._id, dtype=pl.Boolean()
            )
        return PolarsColumn(self.column > other, id_=self._id, dtype=pl.Boolean())

    def __le__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column <= other.column, id_=self._id, dtype=pl.Boolean()
            )
        return PolarsColumn(self.column <= other, id_=self._id, dtype=pl.Boolean())

    def __lt__(self, other: Column[DType] | Any) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column < other.column, id_=self._id, dtype=pl.Boolean()
            )
        return PolarsColumn(self.column < other, id_=self._id, dtype=pl.Boolean())

    def __mul__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            res = self.column * other.column
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") * pl.col("b"))
                .schema["result"]
            )
            return PolarsColumn(res, dtype=res_dtype, id_=self._id)
        res = self.column * other
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") * other)
            .schema["result"]
        )
        return PolarsColumn(res, dtype=res_dtype, id_=self._id)

    def __floordiv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") // other.column)
                .schema["result"]
            )
            return PolarsColumn(
                self.column // other.column, dtype=res_dtype, id_=self._id
            )
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") // other)
            .schema["result"]
        )
        return PolarsColumn(self.column // other, dtype=res_dtype, id_=self._id)

    def __truediv__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res = self.column / other.column
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") / pl.col("b"))
                .schema["result"]
            )
            return PolarsColumn(res, dtype=res_dtype, id_=self._id)
        res = self.column / other
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") / other)
            .schema["result"]
        )
        return PolarsColumn(res, dtype=res_dtype, id_=self._id)

    def __pow__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        original_type = self._dtype
        if isinstance(other, PolarsColumn):
            ret = self.column**other.column  # type: ignore[operator]
            ret_type = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": original_type, "b": other._dtype}
                )
                .select(result=pl.col("a") ** pl.col("b"))
                .schema["result"]
            )
            if _is_integer_dtype(original_type) and _is_integer_dtype(other._dtype):
                ret_type = original_type
                ret = ret.cast(ret_type)
        else:
            ret = self.column.pow(other)  # type: ignore[arg-type]
            ret_type = (
                pl.DataFrame({"a": [1]}, schema={"a": original_type})
                .select(result=pl.col("a") ** other)  # type: ignore[operator]
                .schema["result"]
            )
            if _is_integer_dtype(original_type) and isinstance(other, int):
                ret_type = original_type
                ret = ret.cast(ret_type)
        return PolarsColumn(ret, dtype=ret_type, id_=self._id)

    def __mod__(self, other: Column[DType] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") % other.column)
                .schema["result"]
            )
            return PolarsColumn(self.column % other.column, dtype=res_dtype, id_=self._id)
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") % other)
            .schema["result"]
        )
        return PolarsColumn(self.column % other, dtype=res_dtype, id_=self._id)

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
                self.column & other.column, dtype=self._dtype, id_=self._id  # type: ignore[operator]
            )
        return PolarsColumn(self.column & other, dtype=self._dtype, id_=self._id)  # type: ignore[operator]

    def __or__(self, other: Column[Bool] | bool) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            return PolarsColumn(
                self.column | other.column, dtype=self._dtype, id_=self._id  # type: ignore[operator]
            )
        return PolarsColumn(self.column | other, dtype=self._dtype, id_=self._id)  # type: ignore[operator]

    def __invert__(self) -> PolarsColumn[Bool]:
        return PolarsColumn(~self.column, id_=self._id, dtype=self._dtype)

    def __add__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") + pl.col("b"))
                .schema["result"]
            )
            return PolarsColumn(self.column + other.column, dtype=res_dtype, id_=self._id)
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") + other)
            .schema["result"]
        )
        return PolarsColumn(self.column + other, dtype=res_dtype, id_=self._id)

    def __sub__(self, other: Column[Any] | Any) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            self._validate_column(other)
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") - pl.col("b"))
                .schema["result"]
            )
            return PolarsColumn(self.column - other.column, dtype=res_dtype, id_=self._id)
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") - other)
            .schema["result"]
        )
        return PolarsColumn(self.column - other, dtype=res_dtype, id_=self._id)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn[Any]:
        # if isinstance(self.column, pl.Expr):
        #     raise NotImplementedError("sorted_indices not implemented for lazy columns")
        expr = self.column.arg_sort(descending=not ascending)
        return PolarsColumn(
            expr,
            id_=self._id,
            dtype=pl.UInt32(),
            method=f"Column.{ascending}_sorted_indices",
        )

    def fill_nan(self, value: float | NullType) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.fill_nan(value), dtype=self._dtype, id_=self._id)  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.fill_null(value), dtype=self._dtype, id_=self._id)

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cumsum(), dtype=self._dtype, id_=self._id)

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cumprod(), dtype=self._dtype, id_=self._id)

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cummax(), dtype=self._dtype, id_=self._id)

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsColumn[DType]:
        return PolarsColumn(self.column.cummin(), dtype=self._dtype, id_=self._id)

    def to_array_object(self, dtype: str) -> Any:
        if isinstance(self.column, pl.Expr):
            raise NotImplementedError("to_array_object not implemented for lazy columns")
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self.column.to_numpy().astype(dtype)

    def rename(self, name: str) -> PolarsColumn[DType]:
        if isinstance(self.column, pl.Series):
            return PolarsColumn(self.column.rename(name), id_=self._id, dtype=self._dtype)
        return PolarsColumn(self.column.alias(name), id_=self._id, dtype=self._dtype)


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
        grp = self.df.groupby(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").any())
        return PolarsDataFrame(result)

    def all(self, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.df.groupby(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").all())
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
        if df is NotImplemented:
            raise NotImplementedError("operation not implemented")
        self.df = df
        self._id = id(df)

    def _validate_column(self, column: PolarsColumn[Any]) -> None:
        if isinstance(column.column, pl.Expr) and column._id != self._id:
            raise ValueError(
                "Column was created from a different dataframe!",
                column._id,
                self._id,
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
        dtype = self.dataframe.schema[name]
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(pl.col(name), dtype=dtype, id_=self._id)
        return PolarsColumn(self.dataframe.get_column(name), dtype=dtype, id_=self._id)

    def get_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names))

    def get_rows(self, indices: PolarsColumn[Any]) -> PolarsDataFrame:  # type: ignore[override]
        assert "idx" not in self.dataframe.columns
        self._validate_column(indices)
        if isinstance(self.dataframe, pl.LazyFrame) and isinstance(
            indices.column, pl.Expr
        ):
            if indices._method is not None and indices._method.endswith("sorted_indices"):
                if "True" in indices._method:
                    return PolarsDataFrame(
                        self.dataframe.sort(indices.column.meta.root_names())
                    )
                return PolarsDataFrame(
                    self.dataframe.sort(indices.column.meta.root_names(), descending=True)
                )
            raise NotImplementedError(
                "get_rows only supported for lazyframes if called right after:\n"
                "- DataFrame.sorted_indices\n"
            )
        assert isinstance(indices.column, pl.Series)
        assert isinstance(self.dataframe, pl.DataFrame)
        return PolarsDataFrame(self.dataframe[indices.column])

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step])

    def get_rows_by_mask(self, mask: Column[Bool]) -> PolarsDataFrame:
        self._validate_column(mask)  # type: ignore[arg-type]
        return PolarsDataFrame(self.df.filter(mask.column))

    def insert(self, loc: int, label: str, value: Column[Any]) -> PolarsDataFrame:
        self._validate_column(value)  # type: ignore[arg-type]
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
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            return PolarsDataFrame(self.dataframe.__eq__(other.dataframe))
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__eq__(other)))

    def __ne__(  # type: ignore[override]
        self,
        other: DataFrame,
    ) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            return PolarsDataFrame(self.dataframe.__ne__(other.dataframe))
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__ne__(other)))

    def __ge__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__ge__(other.dataframe)
            return PolarsDataFrame(res)
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__ge__(other)))

    def __gt__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__gt__(other.dataframe)
            return PolarsDataFrame(res)
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__gt__(other)))

    def __le__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__le__(other.dataframe)
            return PolarsDataFrame(res)
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__le__(other)))

    def __lt__(self, other: PolarsDataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__lt__(other.dataframe)
            return PolarsDataFrame(res)
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__lt__(other)))

    def __and__(self, other: PolarsDataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.with_columns(
                    self.dataframe.get_column(col) & other.dataframe.get_column(col)  # type: ignore[union-attr]
                    for col in self.dataframe.columns
                )
            )
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*") & other))

    def __or__(self, other: PolarsDataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.with_columns(
                    self.dataframe.get_column(col) | other.dataframe.get_column(col)  # type: ignore[union-attr]
                    for col in self.dataframe.columns
                )
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            )
        )

    def __add__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__add__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__add__(other)))

    def __sub__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__sub__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__sub__(other)))

    def __mul__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__mul__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").__mul__(other)))

    def __truediv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__truediv__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other))
        )

    def __floordiv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__floordiv__(other.dataframe))  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__floordiv__(other))
        )

    def __pow__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
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
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            res = self.dataframe.__mod__(other.dataframe)  # type: ignore[operator]
            if res is NotImplemented:
                raise NotImplementedError("operation not supported for lazyframes")
            return PolarsDataFrame(res)
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*") % other))

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PolarsDataFrame, PolarsDataFrame]:
        # quotient = self // other
        # remainder = self - quotient * other
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            quotient = self // other
            remainder = self - quotient * other
            return quotient, remainder
        quotient_df = self.dataframe.with_columns(pl.col("*") // other)
        remainder_df = self.dataframe.with_columns(
            pl.col("*") - (pl.col("*") // other) * other
        )
        return PolarsDataFrame(quotient_df), PolarsDataFrame(remainder_df)

    def __invert__(self) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(~pl.col("*")))

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_null(self) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.with_columns(pl.col("*").is_null()))

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsDataFrame(df)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").any()))

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").all()))

    def any_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        expr = pl.any_horizontal(pl.col("*"))
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(expr, id_=self._id, dtype=pl.Boolean())
        return PolarsColumn(
            self.dataframe.select(expr).get_column("any"),
            dtype=pl.Boolean(),
            id_=self._id,
        )

    def all_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        expr = pl.all_horizontal(pl.col("*"))
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(expr, id_=self._id, dtype=pl.Boolean())
        return PolarsColumn(
            self.dataframe.select(expr).get_column("all"),
            dtype=pl.Boolean(),
            id_=self._id,
        )

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
        expr = pl.arg_sort_by(keys, descending=not ascending)
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsColumn(
                expr,
                dtype=pl.UInt32(),
                id_=self._id,
                method=f"DataFrame.{ascending}_sorted_indices",
            )
        return PolarsColumn(
            self.dataframe.select(expr.alias("idx"))["idx"],
            dtype=pl.UInt32(),
            id_=self._id,
        )

    def unique_indices(
        self, keys: Sequence[str] | None = None, *, skip_nulls: bool = True
    ) -> PolarsColumn[Any]:
        df = self.dataframe
        if keys is None:
            keys = df.columns
        if isinstance(df, pl.LazyFrame):
            raise NotImplementedError(
                "unique_indices is not yet supported for lazyframes"
            )
        return PolarsColumn(
            df.with_row_count().unique(keys).get_column("row_nr"),
            dtype=pl.UInt32(),
            id_=self._id,
        )

    def fill_nan(
        self,
        value: float | NullType,
    ) -> PolarsDataFrame:
        if isinstance(value, Null):
            value = None
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
            # todo - document this in the spec?
            return self.dataframe.collect().to_numpy().astype(dtype)
        return self.dataframe.to_numpy().astype(dtype)
