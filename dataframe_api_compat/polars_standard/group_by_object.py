from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import polars as pl

from dataframe_api_compat.polars_standard import Namespace
from dataframe_api_compat.polars_standard.dataframe_object import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api import GroupBy as GroupByT
    from dataframe_api.groupby_object import Aggregation as AggregationT
    from dataframe_api.typing import NullType
    from dataframe_api.typing import Scalar

else:
    GroupByT = object


class GroupBy(GroupByT):
    def __init__(
        self,
        df: DataFrame,
        keys: Sequence[str],
        api_version: str,
    ) -> None:
        for key in keys:
            if key not in df.column_names:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        self._df = df.dataframe
        self._is_persisted = df._is_persisted
        self._keys = keys
        self._api_version = api_version
        self._grouped = (
            self._df.groupby(self._keys)
            if pl.__version__ < "0.19.0"
            else self._df.group_by(self._keys)
        )

    def _validate_booleanness(self) -> None:
        if not all(
            self._df.schema[col] is pl.Boolean
            for col in self._df.columns
            if col not in self._keys
        ):
            msg = "Expected all boolean columns"
            raise TypeError(msg)

    def _to_dataframe(self, result: pl.DataFrame | pl.LazyFrame) -> DataFrame:
        return DataFrame(
            result,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    def size(self) -> DataFrame:
        result = self._grouped.count().rename({"count": "size"})
        return self._to_dataframe(result)

    def any(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        result = self._grouped.agg(pl.col("*").any())
        return self._to_dataframe(result)

    def all(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        result = self._grouped.agg(pl.col("*").all())
        return self._to_dataframe(result)

    def min(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.agg(pl.col("*").min())
        return self._to_dataframe(result)

    def max(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.agg(pl.col("*").max())
        return self._to_dataframe(result)

    def sum(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.agg(pl.col("*").sum())
        return self._to_dataframe(result)

    def prod(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.agg(pl.col("*").product())
        return self._to_dataframe(result)

    def median(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.agg(pl.col("*").median())
        return self._to_dataframe(result)

    def mean(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.agg(pl.col("*").mean())
        return self._to_dataframe(result)

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        result = self._grouped.agg(pl.col("*").std())
        return self._to_dataframe(result)

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        result = self._grouped.agg(pl.col("*").var())
        return self._to_dataframe(result)

    def aggregate(
        self,
        *aggregations: AggregationT,
    ) -> DataFrame:
        return self._to_dataframe(
            self._grouped.agg(
                *[resolve_aggregation(aggregation) for aggregation in aggregations],
            ),
        )


def resolve_aggregation(aggregation: AggregationT) -> pl.Expr:
    aggregation = cast(Namespace.Aggregation, aggregation)
    if aggregation.aggregation == "count":
        return pl.count().alias(aggregation.output_name)
    return getattr(  # type: ignore[no-any-return]
        pl.col(aggregation.column_name),
        aggregation.aggregation,
    )().alias(
        aggregation.output_name,
    )
