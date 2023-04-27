from __future__ import annotations

from typing import NoReturn
import polars as pl


class PolarsColumn:
    def __init__(self, column):
        self._series = column

    def __len__(self) -> int:
        return len(self._series)

    def __getitem__(self, row: int) -> object:
        return self._series[row]

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def unique(self) -> PolarsColumn:
        return PolarsColumn(self._series.unique())

    def mean(self) -> float:
        return self.series.mean()

    @classmethod
    def from_array(cls, array, dtype) -> PolarsColumn:
        dtype_map = {"int": pl.Int64, "float": pl.Float64}
        return cls(pl.Series(array, dtype=dtype_map[dtype]))

    def isnull(self):
        return PolarsColumn(self._series.is_null())

    def isnan(self) -> PolarsColumn:
        return PolarsColumn(self._series.is_nan())

    def any(self):
        return self._series.any()

    def all(self):
        return self._series.all()

    def __eq__(self, other) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series == other._series)
        return PolarsColumn(self._series == other)

    def __and__(self, other) -> PolarsColumn:
        if isinstance(other, PolarsColumn):
            return PolarsColumn(self._series & other._series)
        return PolarsColumn(self._series & other)

    def __invert__(self, other) -> PolarsColumn:
        return PolarsColumn(~self._series)

    def max(self):
        return self._series.max()


class PolarsGroupBy:
    def __init__(self, df, keys):
        self.df = df
        self.keys = keys

    def any(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").any())
        return PolarsDataFrame(result)

    def all(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").all())
        return PolarsDataFrame(result)

    def min(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result)

    def max(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result)

    def sum(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result)

    def prod(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result)

    def median(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result)

    def mean(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result)

    def std(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result)

    def var(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result)


class PolarsDataFrame:
    def __init__(self, df):
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df

    @property
    def dataframe(self):
        return self.df

    @property
    def column_class(self):
        return PolarsColumn

    def shape(self):
        return self.df.shape

    @classmethod
    def from_dict(cls, data: dict[str, PolarsColumn]) -> PolarsDataFrame:
        return cls(
            pl.DataFrame({label: column._series for label, column in data.items()})
        )

    def groupby(self, keys):
        return PolarsGroupBy(self.df, keys)

    def __iter__(self):
        yield from self.column_names

    def get_column_by_name(self, name):
        return PolarsColumn(self.df[name])

    def get_columns_by_name(self, names):
        return PolarsDataFrame(self.df.select(names))

    def get_rows(self, indices):
        return PolarsDataFrame(
            self.df.with_row_count("idx").filter(pl.col("idx").is_in(indices)).drop("idx")
        )

    def slice_rows(self, start, stop, step):
        return PolarsDataFrame(
            self.df.with_row_count("idx")
            .filter(pl.col("idx").is_in(range(start, stop, step)))
            .drop("idx")
        )

    def get_rows_by_mask(self, mask):
        return PolarsDataFrame(self.df.filter(mask._series))

    def insert(self, loc, label, value):
        df = self.df.clone()
        df.insert_at_idx(loc, pl.Series(label, value._series))
        return PolarsDataFrame(df)

    def drop_column(self, label):
        return PolarsDataFrame(self.df.drop(label))

    def set_column(self, label, value):
        columns = self.df.columns
        if label in columns:
            idx = self.df.columns.index(idx)
            return self.drop_column(label).insert(idx, label, value)
        return PolarsDataFrame(self.df.with_columns(label=pl.Series(value)))

    def rename(self, mapping):
        return PolarsDataFrame(self.df.rename(mapping))

    def get_column_names(self):
        return self.df.columns
