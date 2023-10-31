from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import polars as pl

from dataframe_api_compat.polars_standard.dataframe_object import PolarsDataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api import Column
    from dataframe_api import DataFrame
    from dataframe_api import GroupBy
else:
    Column = object
    DataFrame = object
    GroupBy = object


class PolarsGroupBy(GroupBy):
    def __init__(self, df: pl.LazyFrame, keys: Sequence[str], api_version: str) -> None:
        for key in keys:
            if key not in df.columns:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        self.df = df
        self.keys = keys
        self._api_version = api_version
        self.group_by = (
            self.df.group_by if pl.__version__ < "0.19.0" else self.df.group_by
        )

    def size(self) -> PolarsDataFrame:
        result = self.group_by(self.keys).count().rename({"count": "size"})
        return PolarsDataFrame(result, api_version=self._api_version)

    def any(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.group_by(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            msg = "Expected all boolean columns"
            raise TypeError(msg)
        result = grp.agg(pl.col("*").any())
        return PolarsDataFrame(result, api_version=self._api_version)

    def all(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        grp = self.group_by(self.keys)
        if not all(
            self.df.schema[col] is pl.Boolean
            for col in self.df.columns
            if col not in self.keys
        ):
            msg = "Expected all boolean columns"
            raise TypeError(msg)
        result = grp.agg(pl.col("*").all())
        return PolarsDataFrame(result, api_version=self._api_version)

    def min(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").min())
        return PolarsDataFrame(result, api_version=self._api_version)

    def max(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").max())
        return PolarsDataFrame(result, api_version=self._api_version)

    def sum(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").sum())
        return PolarsDataFrame(result, api_version=self._api_version)

    def prod(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").product())
        return PolarsDataFrame(result, api_version=self._api_version)

    def median(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").median())
        return PolarsDataFrame(result, api_version=self._api_version)

    def mean(self, *, skip_nulls: bool = True) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").mean())
        return PolarsDataFrame(result, api_version=self._api_version)

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").std())
        return PolarsDataFrame(result, api_version=self._api_version)

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> PolarsDataFrame:
        result = self.group_by(self.keys).agg(pl.col("*").var())
        return PolarsDataFrame(result, api_version=self._api_version)

    def aggregate(self, *args: Any) -> Any:
        raise NotImplementedError  # todo
