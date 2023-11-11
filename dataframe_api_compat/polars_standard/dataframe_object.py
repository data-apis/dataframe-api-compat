from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

import polars as pl

import dataframe_api_compat
from dataframe_api_compat.polars_standard.column_object import PolarsColumn

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from dataframe_api import Column
    from dataframe_api import DataFrame
    from dataframe_api import GroupBy
    from dataframe_api.typing import DType

    from dataframe_api_compat.polars_standard.group_by_object import PolarsGroupBy

else:
    Column = object
    DataFrame = object
    GroupBy = object


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


class PolarsDataFrame(DataFrame):
    def __init__(
        self,
        df: pl.LazyFrame,
        api_version: str,
        *,
        is_persisted: bool = False,
    ) -> None:
        self.df = df
        self.api_version = api_version
        self.is_persisted = is_persisted

    def materialise_expression(self, expr: pl.Expr) -> pl.Series:
        if not self.is_persisted:
            msg = "Cannot materialise a lazy dataframe, please call `persist` first"
            raise ValueError(msg)  # todo better err message
        df = self.dataframe.collect().select(expr)
        return df.get_column(df.columns[0])

    def validate_is_persisted(self, method: str) -> pl.DataFrame:
        if not self.is_persisted:
            msg = f"Method {method} requires you to call `.persist` first on the parent dataframe.\n\nNote: `.persist` forces materialisation in lazy libraries and so should be called as late as possible in your pipeline, and only once per dataframe."
            raise ValueError(
                msg,
            )
        return self.dataframe.collect()

    def _validate_booleanness(self) -> None:
        if not all(v == pl.Boolean for v in self.dataframe.schema.values()):
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def col(self, value: str) -> PolarsColumn:
        return PolarsColumn(pl.col(value), df=self, api_version=self.api_version)

    @property
    def schema(self) -> dict[str, DType]:
        return {
            column_name: dataframe_api_compat.polars_standard.map_polars_dtype_to_standard_dtype(
                dtype,
            )
            for column_name, dtype in self.dataframe.schema.items()
        }

    def shape(self) -> tuple[int, int]:
        df = self.validate_is_persisted("shape")
        return df.shape

    def __repr__(self) -> str:  # pragma: no cover
        return self.dataframe.__repr__()

    def __dataframe_namespace__(self) -> Any:
        return dataframe_api_compat.polars_standard.PolarsNamespace(
            api_version=self.api_version,
        )

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns

    @property
    def dataframe(self) -> pl.LazyFrame:
        return self.df

    def group_by(self, *keys: str) -> PolarsGroupBy:
        from dataframe_api_compat.polars_standard.group_by_object import PolarsGroupBy

        return PolarsGroupBy(self.dataframe, list(keys), api_version=self.api_version)

    def select(self, *columns: str) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.df.select(list(columns)),
            api_version=self.api_version,
        )

    def get_rows(self, indices: PolarsColumn) -> PolarsDataFrame:  # type: ignore[override]
        self._validate_column(indices)
        return PolarsDataFrame(
            self.dataframe.select(pl.all().take(indices.expr)),
            api_version=self.api_version,
        )

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.df[start:stop:step], api_version=self.api_version)

    def _validate_column(self, column: PolarsColumn) -> None:
        if id(self) != id(column.df):
            msg = "Column is from a different dataframe"
            raise ValueError(msg)

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
        if not isinstance(mapping, collections.abc.Mapping):
            msg = f"Expected Mapping, got: {type(mapping)}"
            raise TypeError(msg)
        return PolarsDataFrame(
            self.dataframe.rename(dict(mapping)),
            api_version=self.api_version,
        )

    def get_column_names(self) -> list[str]:  # pragma: no cover
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
        return self.dataframe.columns

    # Binary

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

    def __rand__(self, other: Any) -> PolarsDataFrame:
        return self.__and__(other)

    def __or__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(
                (pl.col(col) | other).alias(col) for col in self.dataframe.columns
            ),
            api_version=self.api_version,
        )

    def __ror__(self, other: Any) -> PolarsDataFrame:
        return self.__or__(other)

    def __add__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__add__(other)),
            api_version=self.api_version,
        )

    def __radd__(self, other: Any) -> PolarsDataFrame:
        return self.__add__(other)

    def __sub__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__sub__(other)),
            api_version=self.api_version,
        )

    def __rsub__(self, other: Any) -> PolarsDataFrame:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__mul__(other)),
            api_version=self.api_version,
        )

    def __rmul__(self, other: Any) -> PolarsDataFrame:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__truediv__(other)),
            api_version=self.api_version,
        )

    def __rtruediv__(self, other: Any) -> PolarsDataFrame:  # pragma: no cover
        raise NotImplementedError

    def __floordiv__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").__floordiv__(other)),
            api_version=self.api_version,
        )

    def __rfloordiv__(self, other: Any) -> PolarsDataFrame:
        raise NotImplementedError

    def __pow__(self, other: Any) -> PolarsDataFrame:
        original_type = self.dataframe.schema
        ret = self.dataframe.select([pl.col(col).pow(other) for col in self.column_names])
        for column in self.dataframe.columns:
            if _is_integer_dtype(original_type[column]) and isinstance(other, int):
                if other < 0:  # pragma: no cover (todo)
                    msg = "Cannot raise integer to negative power"
                    raise ValueError(msg)
                ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        return PolarsDataFrame(ret, api_version=self.api_version)

    def __rpow__(self, other: Any) -> PolarsDataFrame:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Any) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*") % other),
            api_version=self.api_version,
        )

    def __rmod__(self, other: Any) -> PolarsDataFrame:
        raise NotImplementedError

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[PolarsDataFrame, PolarsDataFrame]:
        quotient_df = self.dataframe.with_columns(pl.col("*") // other)
        remainder_df = self.dataframe.with_columns(
            pl.col("*") - (pl.col("*") // other) * other,
        )
        return PolarsDataFrame(
            quotient_df,
            api_version=self.api_version,
        ), PolarsDataFrame(remainder_df, api_version=self.api_version)

    def __invert__(self) -> PolarsDataFrame:
        self._validate_booleanness()
        return PolarsDataFrame(
            self.dataframe.select(~pl.col("*")),
            api_version=self.api_version,
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    def is_null(self) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.with_columns(pl.col("*").is_null()),
            api_version=self.api_version,
        )

    def is_nan(self) -> PolarsDataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return PolarsDataFrame(df, api_version=self.api_version)

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").any()),
            api_version=self.api_version,
        )

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").all()),
            api_version=self.api_version,
        )

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").min()),
            api_version=self.api_version,
        )

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").max()),
            api_version=self.api_version,
        )

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").sum()),
            api_version=self.api_version,
        )

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").product()),
            api_version=self.api_version,
        )

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").mean()),
            api_version=self.api_version,
        )

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").median()),
            api_version=self.api_version,
        )

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").std()),
            api_version=self.api_version,
        )

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.dataframe.select(pl.col("*").var()),
            api_version=self.api_version,
        )

    # Horizontal reductions

    def all_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return PolarsColumn(
            pl.all_horizontal(self.column_names).alias("all"),
            api_version=self.api_version,
            df=self,
        )

    def any_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn:
        return PolarsColumn(
            pl.any_horizontal(self.column_names).alias("all"),
            api_version=self.api_version,
            df=self,
        )

    def sorted_indices(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsColumn:
        raise NotImplementedError

    def unique_indices(self, *keys: str, skip_nulls: bool = True) -> PolarsColumn:
        raise NotImplementedError

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
        value: float | None,
    ) -> PolarsDataFrame:
        if isinstance(value, dataframe_api_compat.polars_standard.PolarsNamespace.Null):
            value = None
        return PolarsDataFrame(
            self.dataframe.fill_nan(value),
            api_version=self.api_version,
        )

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
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
            raise ValueError(msg)

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        # need to do some extra work to preserve all names
        # https://github.com/pola-rs/polars/issues/9335
        extra_right_keys = set(right_on).difference(left_on)
        assert isinstance(other, PolarsDataFrame)
        other_df = other.dataframe
        # TODO: make more robust
        other_df = other_df.with_columns(
            [pl.col(i).alias(f"{i}_tmp") for i in extra_right_keys],
        )
        result = self.dataframe.join(
            other_df,
            left_on=left_on,
            right_on=right_on,
            how=how,
        )
        result = result.rename({f"{i}_tmp": i for i in extra_right_keys})

        return PolarsDataFrame(result, api_version=self.api_version)

    def persist(self) -> PolarsDataFrame:
        if not self.is_persisted:
            return PolarsDataFrame(
                self.dataframe.collect().lazy(),
                api_version=self.api_version,
                is_persisted=True,
            )
        msg = "DataFrame was already persisted"  # is this really necessary?
        raise ValueError(msg)

    def to_array(self, dtype: DType) -> Any:
        dtype = dtype  # todo
        df = self.validate_is_persisted("DataFrame.to_array")
        return df.to_numpy()
