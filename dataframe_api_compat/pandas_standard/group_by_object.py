from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from dataframe_api_compat.pandas_standard.dataframe_object import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api import GroupBy as GroupByT

    import dataframe_api_compat
else:
    GroupByT = object


class GroupBy(GroupByT):
    def __init__(self, df: pd.DataFrame, keys: Sequence[str], api_version: str) -> None:
        self.df = df
        self.grouped = df.groupby(list(keys), sort=False, as_index=False)
        self.keys = list(keys)
        self._api_version = api_version

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self.df.columns.difference(result.columns)
        if len(failed_columns) > 0:  # pragma: no cover
            msg = "Groupby operation could not be performed on columns "
            f"{failed_columns}. Please drop them before calling group_by."
            raise AssertionError(
                msg,
            )

    def size(self) -> DataFrame:
        return DataFrame(self.grouped.size(), api_version=self._api_version)

    def _validate_booleanness(self) -> None:
        if not (
            (self.df.drop(columns=self.keys).dtypes == "bool")
            | (self.df.drop(columns=self.keys).dtypes == "boolean")
        ).all():
            msg = "'function' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def any(self, *, skip_nulls: bool = True) -> DataFrame:
        self._validate_booleanness()
        result = self.grouped.any()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def all(self, *, skip_nulls: bool = True) -> DataFrame:
        self._validate_booleanness()
        result = self.grouped.all()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def min(self, *, skip_nulls: bool = True) -> DataFrame:
        result = self.grouped.min()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def max(self, *, skip_nulls: bool = True) -> DataFrame:
        result = self.grouped.max()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def sum(self, *, skip_nulls: bool = True) -> DataFrame:
        result = self.grouped.sum()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def prod(self, *, skip_nulls: bool = True) -> DataFrame:
        result = self.grouped.prod()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def median(self, *, skip_nulls: bool = True) -> DataFrame:
        result = self.grouped.median()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def mean(self, *, skip_nulls: bool = True) -> DataFrame:
        result = self.grouped.mean()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> DataFrame:
        result = self.grouped.std()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> DataFrame:
        result = self.grouped.var()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def aggregate(  # type: ignore[override]
        self,
        *aggregations: dataframe_api_compat.pandas_standard.Namespace.Aggregation,
    ) -> DataFrame:  # pragma: no cover
        output_names = [aggregation.output_name for aggregation in aggregations]

        include_size = False
        size_output_name = None
        column_aggregations: list[
            dataframe_api_compat.pandas_standard.Namespace.Aggregation
        ] = []
        for aggregation in aggregations:
            if aggregation.aggregation == "size":
                include_size = True
                size_output_name = aggregation.output_name
            else:
                column_aggregations.append(aggregation)

        agg = {
            aggregation.column_name: aggregation.aggregation
            for aggregation in column_aggregations
        }
        if agg:
            aggregated = self.grouped.agg(agg).rename(
                {
                    aggregation.column_name: aggregation.output_name
                    for aggregation in column_aggregations
                },
                axis=1,
            )

        if include_size:
            size = self.grouped.size().drop(self.keys, axis=1)
            assert len(size.columns) == 1
            size = size.rename(columns={size.columns[0]: size_output_name})

        if agg and include_size:
            df = pd.concat([aggregated, size], axis=1)
        elif agg:
            df = aggregated
        elif include_size:
            df = size
        else:
            msg = "No aggregations specified"
            raise ValueError(msg)
        return DataFrame(
            df.loc[:, output_names],
            api_version=self._api_version,
            is_persisted=False,
        )
