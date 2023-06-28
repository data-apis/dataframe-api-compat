from __future__ import annotations
import polars_standard
import collections

from typing import (
    Any,
    Sequence,
    Mapping,
    NoReturn,
    TYPE_CHECKING,
    Generic,
    TypeVar,
    Literal,
)
import polars as pl

if TYPE_CHECKING:
    from dataframe_api import (
        DataFrame,
        DTypeT,
        IntDType,
        Bool,
        Column,
        Scalar,
        GroupBy,
        null,
        DType,
    )
else:
    DTypeT = TypeVar("DTypeT")

    class DataFrame(Generic[DTypeT]):
        ...

    class Column(Generic[DTypeT]):
        ...

    class GroupBy:
        ...


class PolarsColumn(Column[DTypeT]):
    def __init__(self, column: pl.Series) -> None:
        self._series = column

    # In the standard
    def __column_namespace__(self, *, api_version: str | None = None) -> Any:
        return polars_standard

    @property
    def column(self) -> pl.Series:
        return self._series

    def __len__(self) -> int:
        return len(self.column)

    @property
    def dtype(self) -> DType:
        return polars_standard.DTYPE_MAP[self.column.dtype]  # type: ignore[index]

    def get_rows(self, indices: Column[IntDType]) -> PolarsColumn[DTypeT]:
        return PolarsColumn(self.column.take(indices.column))

    def get_value(self, row: int) -> Scalar:
        return self.column[row]  # type: ignore[no-any-return]

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def is_in(self, values: Column[DTypeT]) -> PolarsColumn[Bool]:
        if values.dtype != self.dtype:
            raise ValueError(f"`value` has dtype {values.dtype}, expected {self.dtype}")
        return PolarsColumn(self.column.is_in(values.column))

    def unique_indices(self, *, skip_nulls: bool = True) -> PolarsColumn[IntDType]:
        df = self.column.to_frame()
        keys = df.columns
        return PolarsColumn(df.with_row_count().unique(keys)["row_nr"])

    def is_null(self) -> PolarsColumn[Bool]:
        return PolarsColumn(self.column.is_null())

    def is_nan(self) -> PolarsColumn[Bool]:
        return PolarsColumn(self.column.is_nan())

    def any(self, *, skip_nulls: bool = True) -> bool:
        return self.column.any()

    def all(self, *, skip_nulls: bool = True) -> bool:
        return self.column.all()

    def min(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.min()  # type: ignore[return-value]

    def max(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.max()  # type: ignore[return-value]

    def sum(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.sum()  # type: ignore[return-value]

    def prod(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.product()  # type: ignore[return-value]

    def mean(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.mean()  # type: ignore[return-value]

    def median(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.median()  # type: ignore[return-value]

    def std(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.std()  # type: ignore[return-value]

    def var(self, *, skip_nulls: bool = True) -> Scalar:
        return self.column.var()  # type: ignore[return-value]

    def __eq__(  # type: ignore[override]
        self, other: Column[DTypeT] | Scalar
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column == other.column)
        return PolarsColumn(self.column == other)

    def __ne__(  # type: ignore[override]
        self, other: Column[DTypeT] | Scalar
    ) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column != other.column)
        return PolarsColumn(self.column != other)

    def __ge__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column >= other.column)
        return PolarsColumn(self.column >= other)

    def __gt__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column > other.column)
        return PolarsColumn(self.column > other)

    def __le__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column <= other.column)
        return PolarsColumn(self.column <= other)

    def __lt__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Bool]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column < other.column)
        return PolarsColumn(self.column < other)

    def __mul__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column * other.column)
        return PolarsColumn(self.column * other)

    def __floordiv__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column // other.column)
        return PolarsColumn(self.column // other)

    def __truediv__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column / other.column)
        return PolarsColumn(self.column / other)

    def __pow__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column.pow(other.column))
        return PolarsColumn(self.column.pow(other))  # type: ignore[arg-type]

    def __mod__(self, other: Column[DTypeT] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column % other.column)
        return PolarsColumn(self.column % other)

    def __divmod__(
        self,
        other: Column[DTypeT] | Scalar,
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

    def __add__(self, other: Column[Any] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column + other.column)
        return PolarsColumn(self.column + other)

    def __sub__(self, other: Column[Any] | Scalar) -> PolarsColumn[Any]:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self.column - other.column)
        return PolarsColumn(self.column - other)

    def sorted_indices(
        self, *, ascending: bool = True, nulls_position: Literal["first", "last"] = "last"
    ) -> PolarsColumn[IntDType]:
        df = self.column.to_frame()
        keys = df.columns
        return PolarsColumn(df.with_row_count().sort(keys, descending=False)["row_nr"])

    def fill_nan(self, value: float | null) -> PolarsColumn[DTypeT]:
        return PolarsColumn(self.column.fill_nan(value))  # type: ignore[arg-type]


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

    def std(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result)

    def var(self, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result)


class PolarsDataFrame(DataFrame):
    def __init__(self, df: pl.DataFrame) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df

    def __dataframe_namespace__(self, *, api_version: str | None = None) -> Any:
        return polars_standard

    @property
    def dataframe(self) -> pl.DataFrame:
        return self.df

    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys)

    def get_column_by_name(self, name: str) -> PolarsColumn[DTypeT]:
        return PolarsColumn(self.df[name])

    def get_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrame:
        if isinstance(names, str):
            raise TypeError(f"Expected sequence of str, got {type(names)}")
        return PolarsDataFrame(self.df.select(names))

    def get_rows(self, indices: Column[IntDType]) -> PolarsDataFrame:
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
        return PolarsDataFrame(
            self.df.with_row_count("idx")
            .filter(pl.col("idx").is_in(range(start, stop, step)))
            .drop("idx")
        )

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
        other: DataFrame | Scalar,
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

    def __ge__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__ge__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__ge__(other))

    def __gt__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__gt__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__gt__(other))

    def __le__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__le__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__le__(other))

    def __lt__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__lt__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__lt__(other))

    def __add__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__add__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__add__(other))  # type: ignore[operator]

    def __sub__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__sub__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__sub__(other))  # type: ignore[operator]

    def __mul__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__mul__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__mul__(other))  # type: ignore[operator]

    def __truediv__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__truediv__(other.dataframe))
        return PolarsDataFrame(
            self.dataframe.__truediv__(other)  # type: ignore[operator]
        )

    def __floordiv__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__floordiv__(other.dataframe))
        return PolarsDataFrame(
            self.dataframe.__floordiv__(other)  # type: ignore[operator]
        )

    def __pow__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(
                self.dataframe.select(
                    [
                        pl.col(col).pow(other.dataframe[col])
                        for col in self.get_column_names()
                    ]
                )
            )
        return PolarsDataFrame(
            self.dataframe.select(
                pl.col(col).pow(other)  # type: ignore[arg-type]
                for col in self.get_column_names()
            )
        )

    def __mod__(self, other: DataFrame | Scalar) -> PolarsDataFrame:
        if isinstance(other, PolarsDataFrame):
            return PolarsDataFrame(self.dataframe.__mod__(other.dataframe))
        return PolarsDataFrame(self.dataframe.__mod__(other))  # type: ignore[operator]

    def __divmod__(
        self,
        other: DataFrame | Scalar,
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
        return PolarsColumn(self.dataframe.select(pl.any(pl.col("*")))["any"])

    def all_rowwise(self, *, skip_nulls: bool = True) -> PolarsColumn[Bool]:
        return PolarsColumn(self.dataframe.select(pl.all(pl.col("*")))["all"])

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

    def std(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").std()))

    def var(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.select(pl.col("*").var()))

    def sorted_indices(
        self,
        keys: Sequence[Any],
        *,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> PolarsColumn[IntDType]:
        df = self.dataframe.select(keys)
        return PolarsColumn(df.with_row_count().sort(keys, descending=False)["row_nr"])

    def fill_nan(
        self,
        value: float | null,
    ) -> PolarsDataFrame:
        return PolarsDataFrame(self.dataframe.fill_nan(value))  # type: ignore[arg-type]
