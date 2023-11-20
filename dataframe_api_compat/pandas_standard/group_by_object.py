from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from dataframe_api_compat.pandas_standard.dataframe_object import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api import GroupBy as GroupByT
    from dataframe_api.typing import NullType
    from dataframe_api.typing import Scalar

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

    def any(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        result = self.grouped.any()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def all(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        result = self.grouped.all()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def min(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self.grouped.min()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def max(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self.grouped.max()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def sum(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self.grouped.sum()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def prod(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self.grouped.prod()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def median(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self.grouped.median()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def mean(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self.grouped.mean()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        result = self.grouped.std()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        result = self.grouped.var()
        self._validate_result(result)
        return DataFrame(result, api_version=self._api_version)

    def aggregate(  # type: ignore[override]
        self,
        *aggregations: dataframe_api_compat.pandas_standard.Namespace.Aggregation,
    ) -> DataFrame:
        [aggregation.output_name for aggregation in aggregations]

        include_size = False
        size_output_name = None
        column_aggregations: dict[
            str,
            dataframe_api_compat.pandas_standard.Namespace.Aggregation,
        ] = {}
        for aggregation in aggregations:
            if aggregation.aggregation == "size":
                include_size = True
                size_output_name = aggregation.output_name
            else:
                column_aggregations[aggregation.output_name] = pd.NamedAgg(
                    column=aggregation.column_name,
                    aggfunc=aggregation.aggregation,
                )
        if column_aggregations:
            aggregated = self.grouped.agg(**column_aggregations)

        if include_size:
            size = self.grouped.size()
            assert len(size.columns) == 1 + len(self.keys)
            size_name = size.columns.difference(self.keys)[0]
            size = size.rename(columns={size_name: size_output_name})

        if column_aggregations and include_size:
            df = pd.concat([aggregated, size.drop(self.keys, axis=1)], axis=1)
        elif column_aggregations:
            df = aggregated
        elif include_size:
            df = size
        else:
            msg = "No aggregations specified"
            raise ValueError(msg)
        return DataFrame(
            df,
            api_version=self._api_version,
            is_persisted=False,
        )
