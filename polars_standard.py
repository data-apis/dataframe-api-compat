from __future__ import annotations

from typing import Sequence, NoReturn, Any, Type, Mapping
import polars as pl
import polars


def dataframe_standard(df: pl.DataFrame) -> PolarsDataFrame:
    return PolarsDataFrame(df)


polars.DataFrame.__dataframe_consortium__ = (  # type: ignore[attr-defined]
    dataframe_standard
)


class PolarsNamespace:
    @classmethod
    def concat(cls, dataframes: Sequence[PolarsDataFrame]) -> PolarsDataFrame:
        dfs = []
        for _df in dataframes:
            dfs.append(_df.dataframe)
        return PolarsDataFrame(pl.concat(dfs))

    @classmethod
    def column_class(cls) -> Type[PolarsColumn]:
        return PolarsColumn


class PolarsColumn:
    def __init__(self, column: pl.Series) -> None:
        self._series = column

    def __len__(self) -> int:
        return len(self._series)

    def __getitem__(self, row: int) -> object:
        return self._series[row]

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def unique(self) -> PolarsColumn:
        return PolarsColumn(self._series.unique())

    def mean(self) -> object:
        return self._series.mean()

    @classmethod
    def from_sequence(cls, array: Sequence[object], dtype: str) -> PolarsColumn:
        # TODO: pending agreement on how to specify dtypes
        dtype_map = {
            "int64": pl.Int64,
            "float64": pl.Float64,
        }
        return cls(pl.Series(array, dtype=dtype_map[dtype]))

    def isnull(self) -> PolarsColumn:
        return PolarsColumn(self._series.is_null())

    def isnan(self) -> PolarsColumn:
        return PolarsColumn(self._series.is_nan())

    def any(self) -> bool:
        return self._series.any()

    def all(self) -> bool:
        return self._series.all()

    def __eq__(  # type: ignore[override]
        self, other: PolarsColumn | object
    ) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series == other._series)
        return PolarsColumn(self._series == other)

    def __and__(self, other: PolarsColumn | object) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series & other._series)
        return PolarsColumn(self._series & other)  # type: ignore[operator]

    def __invert__(self) -> PolarsColumn:
        return PolarsColumn(~self._series)

    def max(self) -> object:
        return self._series.max()

    def std(self) -> object:
        return self._series.std()

    def __gt__(self, other: PolarsColumn) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series > other._series)
        return PolarsColumn(self._series > other)

    def __lt__(self, other: PolarsColumn) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series < other._series)
        return PolarsColumn(self._series < other)

    def __add__(self, other: PolarsColumn) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series + other._series)
        return PolarsColumn(self._series + other)

    def __sub__(self, other: PolarsColumn) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series - other._series)
        return PolarsColumn(self._series - other)

    def __truediv__(self, other: PolarsColumn) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series / other._series)
        return PolarsColumn(self._series / other)


class PolarsGroupBy:
    def __init__(self, df: pl.DataFrame, keys: Sequence[str]) -> None:
        self.df = df
        self.keys = keys

    def size(self) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).count().rename({"count": "size"})
        return PolarsDataFrame(result)

    def any(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").any())
        return PolarsDataFrame(result)

    def all(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").all())
        return PolarsDataFrame(result)

    def min(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result)

    def max(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result)

    def sum(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result)

    def prod(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result)

    def median(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result)

    def mean(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result)

    def std(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result)

    def var(self, skipna: bool = True) -> PolarsDataFrame:
        result = self.df.groupby(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result)


class PolarsDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df

    def __dataframe_namespace__(self, *, api_version: str | None = None) -> Any:
        return PolarsNamespace

    def __len__(self) -> int:
        return len(self.df)

    @property
    def dataframe(self) -> pl.DataFrame:
        return self.df

    def shape(self) -> tuple[int, int]:
        return self.df.shape

    @classmethod
    def from_dict(cls, data: dict[str, PolarsColumn]) -> PolarsDataFrame:
        return cls(
            pl.DataFrame({label: column._series for label, column in data.items()})
        )

    def groupby(self, keys: Sequence[str]) -> PolarsGroupBy:
        return PolarsGroupBy(self.df, keys)

    def get_column_by_name(self, name: str) -> PolarsColumn:
        return PolarsColumn(self.df[name])

    def get_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.select(names))

    def get_rows(self, indices: PolarsColumn) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.df.with_row_count("idx")
            .filter(pl.col("idx").is_in(indices._series))
            .drop("idx")
        )

    def slice_rows(self, start: int, stop: int, step: int) -> PolarsDataFrame:
        return PolarsDataFrame(
            self.df.with_row_count("idx")
            .filter(pl.col("idx").is_in(range(start, stop, step)))
            .drop("idx")
        )

    def get_rows_by_mask(self, mask: PolarsColumn) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.filter(mask._series))

    def insert(self, loc: int, label: str, value: PolarsColumn) -> PolarsDataFrame:
        df = self.df.clone()
        if len(df) > 0:
            df.insert_at_idx(loc, pl.Series(label, value._series))
            return PolarsDataFrame(df)
        return PolarsDataFrame(pl.DataFrame({label: value._series}))

    def drop_column(self, label: str) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.drop(label))

    def rename(self, mapping: Mapping[str, str]) -> PolarsDataFrame:
        return PolarsDataFrame(self.df.rename(dict(mapping)))

    def get_column_names(self) -> Sequence[str]:
        return self.df.columns

    def isnull(self) -> PolarsDataFrame:
        result = {}
        for column in self.dataframe.columns:
            result[column] = self.dataframe[column].is_null()
        return PolarsDataFrame(pl.DataFrame(result))

    def any_rowwise(self) -> PolarsColumn:
        # self._validate_booleanness()
        return PolarsColumn(self.dataframe.select(pl.any(pl.col("*")))["any"])

    def sorted_indices(self, keys: Sequence[str]) -> PolarsColumn:
        df = self.dataframe.select(keys)
        return PolarsColumn(df.with_row_count().sort(keys)["row_nr"])
