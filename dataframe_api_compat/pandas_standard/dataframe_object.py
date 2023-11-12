from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype

import dataframe_api_compat

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from dataframe_api import DataFrame as DataFrameT
    from dataframe_api.typing import DType

    from dataframe_api_compat.pandas_standard.column_object import Column
    from dataframe_api_compat.pandas_standard.group_by_object import GroupBy
else:
    DataFrameT = object


class DataFrame(DataFrameT):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        api_version: str,
        is_collected: bool = False,
    ) -> None:
        self.is_persisted = is_collected
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe.reset_index(drop=True)
        self.api_version = api_version

    def validate_is_persisted(self) -> pd.DataFrame:
        if not self.is_persisted:
            msg = "Method requires you to call `.persist` first on the parent dataframe.\n\nNote: `.persist` forces materialisation in lazy libraries and so should be called as late as possible in your pipeline, and only once per dataframe."
            raise ValueError(
                msg,
            )
        return self.dataframe

    def __repr__(self) -> str:  # pragma: no cover
        return self.dataframe.__repr__()  # type: ignore[no-any-return]

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                msg = f"Expected unique column names, got {col} {count} time(s)"
                raise ValueError(
                    msg,
                )

    def _validate_booleanness(self) -> None:
        if not (
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def _validate_column(self, column: Column) -> None:
        if id(self) != id(column.df):
            msg = "cannot compare columns from different dataframes"
            raise ValueError(msg)

    # In the Standard

    def col(self, name: str) -> Column:
        from dataframe_api_compat.pandas_standard.column_object import Column

        return Column(
            self.dataframe.loc[:, name],
            df=self,
            api_version=self.api_version,
        )

    def shape(self) -> tuple[int, int]:
        df = self.validate_is_persisted()
        return df.shape  # type: ignore[no-any-return]

    @property
    def schema(self) -> dict[str, Any]:
        return {
            column_name: dataframe_api_compat.pandas_standard.map_pandas_dtype_to_standard_dtype(
                dtype.name,
            )
            for column_name, dtype in self.dataframe.dtypes.items()
        }

    def __dataframe_namespace__(
        self,
    ) -> dataframe_api_compat.pandas_standard.Namespace:
        return dataframe_api_compat.pandas_standard.Namespace(
            api_version=self.api_version,
        )

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> DataFrame:
        return DataFrame(
            self.dataframe.iloc[start:stop:step],
            api_version=self.api_version,
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def group_by(self, *keys: str) -> GroupBy:
        from dataframe_api_compat.pandas_standard.group_by_object import GroupBy

        for key in keys:
            if key not in self.column_names:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        return GroupBy(self.dataframe, keys, api_version=self.api_version)

    def select(self, *columns: str) -> DataFrame:
        return DataFrame(
            self.dataframe.loc[:, list(columns)],
            api_version=self.api_version,
        )

    def get_rows(
        self,
        indices: Column,  # type: ignore[override]
    ) -> DataFrame:
        self._validate_column(indices)
        return DataFrame(
            self.dataframe.iloc[indices.column, :],
            api_version=self.api_version,
        )

    def filter(
        self,
        mask: Column,  # type: ignore[override]
    ) -> DataFrame:
        self._validate_column(mask)
        df = self.dataframe
        df = df.loc[mask.column]
        return DataFrame(df, api_version=self.api_version)

    def assign(
        self,
        *columns: Column,  # type: ignore[override]
    ) -> DataFrame:
        df = self.dataframe.copy()  # TODO: remove defensive copy with CoW?
        for column in columns:
            self._validate_column(column)
            df[column.name] = column.column
        return DataFrame(df, api_version=self.api_version)

    def drop_columns(self, *labels: str) -> DataFrame:
        return DataFrame(
            self.dataframe.drop(list(labels), axis=1),
            api_version=self.api_version,
        )

    def rename_columns(self, mapping: Mapping[str, str]) -> DataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            msg = f"Expected Mapping, got: {type(mapping)}"
            raise TypeError(msg)
        return DataFrame(
            self.dataframe.rename(columns=mapping),
            api_version=self.api_version,
        )

    def get_column_names(self) -> list[str]:  # pragma: no cover
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> DataFrame:
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return DataFrame(
            df.sort_values(list(keys), ascending=ascending),
            api_version=self.api_version,
        )

    # Binary operations

    def __eq__(self, other: Any) -> DataFrame:  # type: ignore[override]
        return DataFrame(self.dataframe.__eq__(other), api_version=self.api_version)

    def __ne__(self, other: Any) -> DataFrame:  # type: ignore[override]
        return DataFrame(self.dataframe.__ne__(other), api_version=self.api_version)

    def __ge__(self, other: Any) -> DataFrame:
        return DataFrame(self.dataframe.__ge__(other), api_version=self.api_version)

    def __gt__(self, other: Any) -> DataFrame:
        return DataFrame(self.dataframe.__gt__(other), api_version=self.api_version)

    def __le__(self, other: Any) -> DataFrame:
        return DataFrame(self.dataframe.__le__(other), api_version=self.api_version)

    def __lt__(self, other: Any) -> DataFrame:
        return DataFrame(self.dataframe.__lt__(other), api_version=self.api_version)

    def __and__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__and__(other),
            api_version=self.api_version,
        )

    def __rand__(self, other: Column | Any) -> DataFrame:
        return self.__and__(other)

    def __or__(self, other: Any) -> DataFrame:
        return DataFrame(self.dataframe.__or__(other), api_version=self.api_version)

    def __ror__(self, other: Column | Any) -> DataFrame:
        return self.__or__(other)

    def __add__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__add__(other),
            api_version=self.api_version,
        )

    def __radd__(self, other: Column | Any) -> DataFrame:
        return self.__add__(other)

    def __sub__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__sub__(other),
            api_version=self.api_version,
        )

    def __rsub__(self, other: Column | Any) -> DataFrame:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__mul__(other),
            api_version=self.api_version,
        )

    def __rmul__(self, other: Column | Any) -> DataFrame:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__truediv__(other),
            api_version=self.api_version,
        )

    def __rtruediv__(self, other: Column | Any) -> DataFrame:  # pragma: no cover
        raise NotImplementedError

    def __floordiv__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__floordiv__(other),
            api_version=self.api_version,
        )

    def __rfloordiv__(self, other: Column | Any) -> DataFrame:  # pragma: no cover
        raise NotImplementedError

    def __pow__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__pow__(other),
            api_version=self.api_version,
        )

    def __rpow__(self, other: Column | Any) -> DataFrame:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Any) -> DataFrame:
        return DataFrame(
            self.dataframe.__mod__(other),
            api_version=self.api_version,
        )

    def __rmod__(self, other: Column | Any) -> DataFrame:  # pragma: no cover
        raise NotImplementedError

    def __divmod__(
        self,
        other: DataFrame | Any,
    ) -> tuple[DataFrame, DataFrame]:
        quotient, remainder = self.dataframe.__divmod__(other)
        return DataFrame(quotient, api_version=self.api_version), DataFrame(
            remainder,
            api_version=self.api_version,
        )

    # Unary

    def __invert__(self) -> DataFrame:
        self._validate_booleanness()
        return DataFrame(self.dataframe.__invert__(), api_version=self.api_version)

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    # Reductions

    def any(self, *, skip_nulls: bool = True) -> DataFrame:
        self._validate_booleanness()
        return DataFrame(
            self.dataframe.any().to_frame().T,
            api_version=self.api_version,
        )

    def all(self, *, skip_nulls: bool = True) -> DataFrame:
        self._validate_booleanness()
        return DataFrame(
            self.dataframe.all().to_frame().T,
            api_version=self.api_version,
        )

    def min(self, *, skip_nulls: bool = True) -> DataFrame:
        return DataFrame(
            self.dataframe.min().to_frame().T,
            api_version=self.api_version,
        )

    def max(self, *, skip_nulls: bool = True) -> DataFrame:
        return DataFrame(
            self.dataframe.max().to_frame().T,
            api_version=self.api_version,
        )

    def sum(self, *, skip_nulls: bool = True) -> DataFrame:
        return DataFrame(
            self.dataframe.sum().to_frame().T,
            api_version=self.api_version,
        )

    def prod(self, *, skip_nulls: bool = True) -> DataFrame:
        return DataFrame(
            self.dataframe.prod().to_frame().T,
            api_version=self.api_version,
        )

    def median(self, *, skip_nulls: bool = True) -> DataFrame:
        return DataFrame(
            self.dataframe.median().to_frame().T,
            api_version=self.api_version,
        )

    def mean(self, *, skip_nulls: bool = True) -> DataFrame:
        return DataFrame(
            self.dataframe.mean().to_frame().T,
            api_version=self.api_version,
        )

    def std(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> DataFrame:
        return DataFrame(
            self.dataframe.std().to_frame().T,
            api_version=self.api_version,
        )

    def var(
        self,
        *,
        correction: int | float = 1.0,
        skip_nulls: bool = True,
    ) -> DataFrame:
        return DataFrame(
            self.dataframe.var().to_frame().T,
            api_version=self.api_version,
        )

    # Horizontal reductions

    def all_rowwise(self, *, skip_nulls: bool = True) -> Column:  # pragma: no cover
        msg = "Please use `__dataframe_namespace__().all_rowwise` instead"
        raise NotImplementedError(msg)

    def any_rowwise(self, *, skip_nulls: bool = True) -> Column:  # pragma: no cover
        msg = "Please use `__dataframe_namespace__().any` instead"
        raise NotImplementedError(msg)

    def sorted_indices(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:  # pragma: no cover
        msg = "Please use `__dataframe_namespace__().sorted_indices` instead"
        raise NotImplementedError(msg)

    def unique_indices(
        self,
        *keys: str,
        skip_nulls: bool = True,
    ) -> Column:  # pragma: no cover
        msg = "Please use `__dataframe_namespace__().unique_indices` instead"
        raise NotImplementedError(msg)

    # Transformations

    def is_null(self, *, skip_nulls: bool = True) -> DataFrame:
        result: list[pd.Series] = []
        for column in self.dataframe.columns:
            result.append(self.dataframe[column].isna())
        return DataFrame(pd.concat(result, axis=1), api_version=self.api_version)

    def is_nan(self) -> DataFrame:
        result: list[pd.Series] = []
        for column in self.dataframe.columns:
            if is_extension_array_dtype(self.dataframe[column].dtype):
                result.append(
                    np.isnan(self.dataframe[column]).replace(pd.NA, False).astype(bool),
                )
            else:
                result.append(self.dataframe[column].isna())
        return DataFrame(pd.concat(result, axis=1), api_version=self.api_version)

    def fill_nan(self, value: float | pd.NAType) -> DataFrame:
        new_cols = {}
        df = self.dataframe
        for col in df.columns:
            ser = df[col].copy()
            if is_extension_array_dtype(ser.dtype):
                if value is dataframe_api_compat.pandas_standard.Namespace.null:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = pd.NA
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
            else:
                if value is dataframe_api_compat.pandas_standard.Namespace.null:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = np.nan
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
            new_cols[col] = ser
        df = pd.DataFrame(new_cols)
        return DataFrame(df, api_version=self.api_version)

    def fill_null(
        self,
        value: Any,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        if column_names is None:
            column_names = self.dataframe.columns.tolist()
        assert isinstance(column_names, list)  # help type checkers
        df = self.dataframe.copy()
        for column in column_names:
            col = df[column]
            if is_extension_array_dtype(col.dtype):
                # crazy hack to preserve nan...
                num = pd.Series(
                    np.where(np.isnan(col).fillna(False), 0, col.fillna(value)),
                    dtype=col.dtype,
                )
                other = pd.Series(
                    np.where(np.isnan(col).fillna(False), 0, 1),
                    dtype=col.dtype,
                )
                col = num / other
            else:
                col = col.fillna(value)
            df[column] = col
        return DataFrame(df, api_version=self.api_version)

    def drop_nulls(
        self,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        namespace = self.__dataframe_namespace__()
        mask = ~namespace.any_rowwise(
            *[
                self.col(col_name).is_null()
                for col_name in column_names or self.column_names
            ],
        )
        return self.filter(mask)

    # Other

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        if how not in ["left", "inner", "outer"]:
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
            raise ValueError(msg)
        assert isinstance(other, DataFrame)
        return DataFrame(
            self.dataframe.merge(
                other.dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
            ),
            api_version=self.api_version,
        )

    def persist(self) -> DataFrame:
        if self.is_persisted:
            msg = "Dataframe is already collected"
            raise ValueError(msg)
        return DataFrame(
            self.dataframe,
            api_version=self.api_version,
            is_collected=True,
        )

    def to_array(self, dtype: DType) -> Any:
        self.validate_is_persisted()
        return self.dataframe.to_numpy(dtype)
