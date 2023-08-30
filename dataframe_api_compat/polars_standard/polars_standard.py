from __future__ import annotations

import collections
from typing import Any
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
        DataFrame,
        GroupBy,
    )
else:

    class DataFrame:
        ...

    class Expression:
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


LATEST_API_VERSION = "2023.08-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.09-beta"))


class PolarsExpression:
    def __init__(
        self,
        expr: pl.Series | pl.Expr,
        *,
        dtype: Any = None,
        id_: int | None = None,  # | None = None,
        api_version: str = LATEST_API_VERSION,
    ) -> None:
        if expr is NotImplemented:
            raise NotImplementedError("operation not implemented")
        if isinstance(expr, str):
            self._expr = pl.col(expr)
        else:
            self._expr = expr
        self._dtype = dtype
        # keep track of which dataframe the column came from
        self._id = id_
        if isinstance(expr, pl.Series):
            # just helps with defensiveness
            assert expr.dtype == dtype
        if api_version not in SUPPORTED_VERSIONS:
            raise ValueError(
                "Unsupported API version, expected one of: "
                f"{SUPPORTED_VERSIONS}. "
                f"Got: {api_version}."
                "Try updating dataframe-api-compat?"
            )
        self._api_version = api_version

    # In the standard
    def __column_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard

    @property
    def name(self) -> str:
        if isinstance(self._expr, pl.Series):
            # TODO: can we avoid this completely?
            name = self._expr.name
        else:
            name = self._expr.meta.output_name()
        return name

    # def __len__(self) -> int:
    #     if isinstance(self._expr, pl.Series):
    #         return len(self._expr)
    #     raise NotImplementedError(
    #         "__len__ intentionally not implemented for lazy columns"
    #     )

    # @property
    # def dtype(self) -> Any:
    #     return dataframe_api_compat.polars_standard.DTYPE_MAP[self._dtype]

    def get_rows(self, indices: PolarsExpression) -> PolarsExpression:
        return PolarsExpression(
            self._expr.take(indices.column),
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsExpression:
        if isinstance(self._expr, pl.Expr):
            raise NotImplementedError("slice_rows not implemented for lazy columns")
        if start is None:
            start = 0
        if stop is None:
            stop = len(self._expr)
        if step is None:
            step = 1
        return PolarsExpression(
            self._expr[start:stop:step],
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def get_rows_by_mask(self, mask: PolarsExpression) -> PolarsExpression:
        return PolarsExpression(
            self._expr.filter(mask.column), dtype=self._dtype, id_=self._id, api_version=self._api_version  # type: ignore[arg-type]
        )

    def get_value(self, row: int) -> Any:
        if isinstance(self._expr, pl.Expr):
            raise NotImplementedError("get_value not implemented for lazy columns")
        return self.Expression

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: PolarsExpression) -> PolarsExpression:  # type: ignore[override]
        if values.dtype != self.dtype:
            raise ValueError(f"`value` has dtype {values.dtype}, expected {self.dtype}")
        return PolarsExpression(
            self._expr.is_in(values.column), dtype=pl.Boolean(), id_=self._id, api_version=self._api_version  # type: ignore[arg-type]
        )

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsExpression:
        if isinstance(self._expr, pl.Expr):
            raise NotImplementedError("unique_indices not implemented for lazy columns")
        df = self._expr.to_frame()
        keys = df.columns
        return PolarsExpression(
            df.with_row_count().unique(keys).get_column("row_nr"),
            dtype=pl.UInt32(),
            id_=self._id,
            api_version=self._api_version,
        )

    def is_null(self) -> PolarsExpression:
        return PolarsExpression(
            self._expr.is_null(),
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
        )

    def is_nan(self) -> PolarsExpression:
        return PolarsExpression(
            self._expr.is_nan(),
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
        )

    def any(self, *, skip_nulls: bool = True) -> bool | None:
        if isinstance(self._expr, pl.Expr):
            raise NotImplementedError("any not implemented for lazy columns")
        return self._expr.any()

    def all(self, *, skip_nulls: bool = True) -> bool | None:
        if isinstance(self._expr, pl.Expr):
            raise NotImplementedError("all not implemented for lazy columns")
        return self._expr.all()

    def min(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").min())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.min(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.min()

    def max(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").max())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.max(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.max()

    def sum(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").sum())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.sum(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.sum()

    def prod(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").product())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.product(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.product()

    def mean(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").mean())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.mean(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.mean()

    def median(self, *, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").median())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.median(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.median()

    def std(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").std())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.std(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.std()

    def var(self, *, correction: int | float = 1.0, skip_nulls: bool = True) -> Any:
        if isinstance(self._expr, pl.Expr):
            res_dtype = (
                pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
                .select(pl.col("a").var())
                .schema["a"]
            )
            return PolarsExpression(
                self._expr.var(),
                id_=self._id,
                dtype=res_dtype,
                api_version=self._api_version,
            )
        return self._expr.var()

    def __eq__(  # type: ignore[override]
        self, other: PolarsExpression | Any
    ) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr == other._expr,
                dtype=pl.Boolean(),
                id_=self._id,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr == other,
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self, other: PolarsExpression | Any
    ) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr != other._expr,
                dtype=pl.Boolean(),
                id_=self._id,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr != other,
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
        )

    def __ge__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr >= other._expr,
                dtype=pl.Boolean(),
                id_=self._id,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr >= other,
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
        )

    def __gt__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr > other._expr,
                id_=self._id,
                dtype=pl.Boolean(),
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr > other,
            id_=self._id,
            dtype=pl.Boolean(),
            api_version=self._api_version,
        )

    def __le__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr <= other._expr,
                id_=self._id,
                dtype=pl.Boolean(),
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr <= other,
            id_=self._id,
            dtype=pl.Boolean(),
            api_version=self._api_version,
        )

    def __lt__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr < other._expr,
                id_=self._id,
                dtype=pl.Boolean(),
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr < other,
            id_=self._id,
            dtype=pl.Boolean(),
            api_version=self._api_version,
        )

    def __mul__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res = self._expr * other._expr
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") * pl.col("b"))
                .schema["result"]
            )
            return PolarsExpression(
                res, dtype=res_dtype, id_=self._id, api_version=self._api_version
            )
        res = self._expr * other
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") * other)
            .schema["result"]
        )
        return PolarsExpression(
            res, dtype=res_dtype, id_=self._id, api_version=self._api_version
        )

    def __floordiv__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") // other._expr)
                .schema["result"]
            )
            return PolarsExpression(
                self._expr // other._expr,
                dtype=res_dtype,
                id_=self._id,
                api_version=self._api_version,
            )
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") // other)
            .schema["result"]
        )
        return PolarsExpression(
            self._expr // other,
            dtype=res_dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def __truediv__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res = self._expr / other._expr
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") / pl.col("b"))
                .schema["result"]
            )
            return PolarsExpression(
                res, dtype=res_dtype, id_=self._id, api_version=self._api_version
            )
        res = self._expr / other
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") / other)
            .schema["result"]
        )
        return PolarsExpression(
            res, dtype=res_dtype, id_=self._id, api_version=self._api_version
        )

    def __pow__(self, other: PolarsExpression | Any) -> PolarsExpression:
        original_type = self._dtype
        if isinstance(other, PolarsExpression):
            ret = self._expr**other._expr  # type: ignore[operator]
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
            ret = self._expr.pow(other)  # type: ignore[arg-type]
            ret_type = (
                pl.DataFrame({"a": [1]}, schema={"a": original_type})
                .select(result=pl.col("a") ** other)  # type: ignore[operator]
                .schema["result"]
            )
            if _is_integer_dtype(original_type) and isinstance(other, int):
                ret_type = original_type
                ret = ret.cast(ret_type)
        return PolarsExpression(
            ret, dtype=ret_type, id_=self._id, api_version=self._api_version
        )

    def __mod__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") % other._expr)
                .schema["result"]
            )
            return PolarsExpression(
                self._expr % other._expr,
                dtype=res_dtype,
                id_=self._id,
                api_version=self._api_version,
            )
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") % other)
            .schema["result"]
        )
        return PolarsExpression(
            self._expr % other,
            dtype=res_dtype,
            id_=self._id,
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
            return PolarsExpression(
                self._expr & other._expr, dtype=self._dtype, id_=self._id, api_version=self._api_version  # type: ignore[operator]
            )
        return PolarsExpression(self._expr & other, dtype=self._dtype, id_=self._id, api_version=self._api_version)  # type: ignore[operator]

    def __or__(self, other: PolarsExpression | bool) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            return PolarsExpression(
                self._expr | other._expr, dtype=self._dtype, id_=self._id, api_version=self._api_version  # type: ignore[operator]
            )
        return PolarsExpression(self._expr | other, dtype=self._dtype, id_=self._id, api_version=self._api_version)  # type: ignore[operator]

    def __invert__(self) -> PolarsExpression:
        return PolarsExpression(
            ~self._expr, id_=self._id, dtype=self._dtype, api_version=self._api_version
        )

    def __add__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") + pl.col("b"))
                .schema["result"]
            )
            return PolarsExpression(
                self._expr + other._expr,
                dtype=res_dtype,
                id_=self._id,
                api_version=self._api_version,
            )
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") + other)
            .schema["result"]
        )
        return PolarsExpression(
            self._expr + other,
            dtype=res_dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def __sub__(self, other: PolarsExpression | Any) -> PolarsExpression:
        if isinstance(other, PolarsExpression):
            res_dtype = (
                pl.DataFrame(
                    {"a": [1], "b": [1]}, schema={"a": self._dtype, "b": other._dtype}
                )
                .select(result=pl.col("a") - pl.col("b"))
                .schema["result"]
            )
            return PolarsExpression(
                self._expr - other._expr,
                dtype=res_dtype,
                id_=self._id,
                api_version=self._api_version,
            )
        res_dtype = (
            pl.DataFrame({"a": [1]}, schema={"a": self._dtype})
            .select(result=pl.col("a") - other)
            .schema["result"]
        )
        return PolarsExpression(
            self._expr - other,
            dtype=res_dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsExpression:
        # if isinstance(self._expr, pl.Expr):
        #     raise NotImplementedError("sorted_indices not implemented for lazy columns")
        expr = self._expr.arg_sort(descending=not ascending)
        return PolarsExpression(
            expr,
            id_=self._id,
            dtype=pl.UInt32(),
            api_version=self._api_version,
        )

    def sort(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsExpression:
        expr = self._expr.sort(descending=not ascending)
        return PolarsExpression(
            expr,
            id_=self._id,
            dtype=self._dtype,
            api_version=self._api_version,
        )

    def fill_nan(self, value: float | NullType) -> PolarsExpression:
        return PolarsExpression(self._expr.fill_nan(value), dtype=self._dtype, id_=self._id, api_version=self._api_version)  # type: ignore[arg-type]

    def fill_null(self, value: Any) -> PolarsExpression:
        return PolarsExpression(
            self._expr.fill_null(value),
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def cumulative_sum(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cumsum(),
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def cumulative_prod(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cumprod(),
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def cumulative_max(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cummax(),
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def cumulative_min(self, *, skip_nulls: bool = True) -> PolarsExpression:
        return PolarsExpression(
            self._expr.cummin(),
            dtype=self._dtype,
            id_=self._id,
            api_version=self._api_version,
        )

    def to_array_object(self, dtype: str) -> Any:
        if isinstance(self._expr, pl.Expr):
            raise NotImplementedError("to_array_object not implemented for lazy columns")
        if dtype not in _ARRAY_API_DTYPES:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected one of {_ARRAY_API_DTYPES}"
            )
        return self._expr.to_numpy().astype(dtype)

    def rename(self, name: str) -> PolarsExpression:
        if isinstance(self._expr, pl.Series):
            return PolarsExpression(
                self._expr.rename(name),
                id_=self._id,
                dtype=self._dtype,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self._expr.alias(name),
            id_=self._id,
            dtype=self._dtype,
            api_version=self._api_version,
        )


class PolarsGroupBy(GroupBy):
    def __init__(
        self, df: pl.DataFrame | pl.LazyFrame, keys: Sequence[str], api_version: str
    ) -> None:
        for key in keys:
            if key not in df.columns:
                raise KeyError(f"key {key} not present in DataFrame's columns")
        self.df = df
        self.keys = keys
        self._api_version = api_version

    def size(self) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).count().rename({"count": "size"})
        return PolarsDataFrame(result, api_version=self._api_version)

    def any(self, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.df.groupby(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").any())
        return PolarsDataFrame(result, api_version=self._api_version)

    def all(self, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.df.groupby(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            raise ValueError("Expected all boolean columns")
        result = grp.agg(pl.col("*").all())
        return PolarsDataFrame(result, api_version=self._api_version)

    def min(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result, api_version=self._api_version)

    def max(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result, api_version=self._api_version)

    def sum(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result, api_version=self._api_version)

    def prod(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result, api_version=self._api_version)

    def median(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result, api_version=self._api_version)

    def mean(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result, api_version=self._api_version)

    def std(
        self, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result, api_version=self._api_version)

    def var(
        self, correction: int | float = 1.0, skip_nulls: bool = True
    ) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result, api_version=self._api_version)


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.DataFrame | pl.LazyFrame, api_version: str) -> None:
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
    def dataframe(self) -> pl.DataFrame | pl.LazyFrame:
        return self.df

    def shape(self) -> tuple[int, int]:
        return self.df.shape  # type: ignore[union-attr]

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys, api_version=self._api_version)

    def get_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names), api_version=self._api_version)

    def get_rows(self, indices: PolarsExpression) -> PolarsDataFrame:  # type: ignore[override]
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices._expr)),
            api_version=self._api_version,
        )

    def slice_rows(
        self, start: int | None, stop: int | None, step: int | None
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self._api_version)

    def get_rows_by_mask(self, mask: PolarsExpression) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask._expr), api_version=self._api_version)

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
        df = self.dataframe.with_columns(value._expr).select(new_columns)
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
            self.dataframe.with_columns([col._expr for col in columns]),
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
        other: DataFrame | Any,
    ) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            return PolarsDataFrame(
                self.dataframe.__eq__(other.dataframe), api_version=self._api_version
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__eq__(other)),
            api_version=self._api_version,
        )

    def __ne__(  # type: ignore[override]
        self,
        other: DataFrame,
    ) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            return PolarsDataFrame(
                self.dataframe.__ne__(other.dataframe), api_version=self._api_version
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
            api_version=self._api_version,
        )

    def __ge__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__ge__(other.dataframe)
            return PolarsDataFrame(res, api_version=self._api_version)
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
            api_version=self._api_version,
        )

    def __gt__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__gt__(other.dataframe)
            return PolarsDataFrame(res, api_version=self._api_version)
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
            api_version=self._api_version,
        )

    def __le__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__le__(other.dataframe)
            return PolarsDataFrame(res, api_version=self._api_version)
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
            api_version=self._api_version,
        )

    def __lt__(self, other: PolarsDataFrame | Any) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            if isinstance(self.dataframe, pl.LazyFrame) or isinstance(
                other.dataframe, pl.LazyFrame
            ):
                raise NotImplementedError("operation not supported for lazyframes")
            res = self.dataframe.__lt__(other.dataframe)
            return PolarsDataFrame(res, api_version=self._api_version)
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
            api_version=self._api_version,
        )

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
                ),
                api_version=self._api_version,
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") & other),
            api_version=self._api_version,
        )

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
                ),
                api_version=self._api_version,
            )
        return PolarsDataFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self._api_version,
        )

    def __add__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__add__(other.dataframe), api_version=self._api_version)  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self._api_version,
        )

    def __sub__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__sub__(other.dataframe), api_version=self._api_version)  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self._api_version,
        )

    def __mul__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__mul__(other.dataframe), api_version=self._api_version)  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self._api_version,
        )

    def __truediv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__truediv__(other.dataframe), api_version=self._api_version)  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self._api_version,
        )

    def __floordiv__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__floordiv__(other.dataframe), api_version=self._api_version)  # type: ignore[operator]
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__floordiv__(other)),
            api_version=self._api_version,
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
        return PolarsDataFrame(ret, api_version=self._api_version)

    def __mod__(self, other: DataFrame | Any) -> PolarsDataFrame:
        if isinstance(self.dataframe, pl.LazyFrame) and (
            isinstance(other, DataFrame) and isinstance(other.dataframe, pl.LazyFrame)
        ):
            raise NotImplementedError("operation not supported for lazyframes")
        if isinstance(other, PolarsDataFrame):
            res = self.dataframe.__mod__(other.dataframe)  # type: ignore[operator]
            if res is NotImplemented:
                raise NotImplementedError("operation not supported for lazyframes")
            return PolarsDataFrame(res, api_version=self._api_version)
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") % other),
            api_version=self._api_version,
        )

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

    def any_rowwise(self, *, skip_nulls: bool = True) -> PolarsExpression:
        expr = pl.any_horizontal(pl.col("*"))
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsExpression(
                expr, id_=self._id, dtype=pl.Boolean(), api_version=self._api_version
            )
        return PolarsExpression(
            self.dataframe.select(expr).get_column("any"),
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
        )

    def all_rowwise(self, *, skip_nulls: bool = True) -> PolarsExpression:
        expr = pl.all_horizontal(pl.col("*"))
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsExpression(
                expr, id_=self._id, dtype=pl.Boolean(), api_version=self._api_version
            )
        return PolarsExpression(
            self.dataframe.select(expr).get_column("all"),
            dtype=pl.Boolean(),
            id_=self._id,
            api_version=self._api_version,
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

    def sorted_indices(
        self,
        keys: Sequence[Any] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsExpression:
        if keys is None:
            keys = self.dataframe.columns
        expr = pl.arg_sort_by(keys, descending=not ascending)
        if isinstance(self.dataframe, pl.LazyFrame):
            return PolarsExpression(
                expr,
                dtype=pl.UInt32(),
                id_=self._id,
                api_version=self._api_version,
            )
        return PolarsExpression(
            self.dataframe.select(expr.alias("idx"))["idx"],
            dtype=pl.UInt32(),
            id_=self._id,
            api_version=self._api_version,
        )

    def sort(
        self,
        keys: Sequence[Any] | None = None,
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsDataFrame:
        if self._api_version == "2023.08-beta":
            raise NotImplementedError("dataframe.sort only available after 2023.08-beta")
        if keys is None:
            keys = self.dataframe.columns
        # TODO: what if there's multiple `ascending`?
        return PolarsDataFrame(
            self.dataframe.sort(keys, descending=not ascending),
            api_version=self._api_version,
        )

    def unique_indices(
        self, keys: Sequence[str] | None = None, *, skip_nulls: bool = True
    ) -> PolarsExpression:
        df = self.dataframe
        if keys is None:
            keys = df.columns
        if isinstance(df, pl.LazyFrame):
            raise NotImplementedError(
                "unique_indices is not yet supported for lazyframes"
            )
        return PolarsExpression(
            df.with_row_count().unique(keys).get_column("row_nr"),
            dtype=pl.UInt32(),
            id_=self._id,
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
        if isinstance(self.dataframe, pl.LazyFrame):
            # todo - document this in the spec?
            return self.dataframe.collect().to_numpy().astype(dtype)
        return self.dataframe.to_numpy().astype(dtype)
