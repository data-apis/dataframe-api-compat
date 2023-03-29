from __future__ import annotations

import pandas as pd
import collections
from typing import Sequence, Mapping, NoReturn


class PandasColumn:
    def __init__(self, column: pd.Series) -> None:  # type: ignore[type-arg]
        self._series = column

    def __dlpack__(self) -> object:
        arr = self._series.to_numpy()
        return arr.__dlpack__()

    def isnull(self) -> PandasColumn:
        return PandasColumn(self._series.isna())

    def notnull(self) -> PandasColumn:
        return PandasColumn(self._series.notna())

    def any(self) -> bool:
        return self._series.any()

    def all(self) -> bool:
        return self._series.all()

    def __len__(self) -> int:
        return len(self._series)


class PandasGroupBy:
    def __init__(self, df: pd.DataFrame, keys: Sequence[str]) -> None:
        self.df = df
        self.keys = list(keys)

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self.df.columns.difference(result.columns)
        if len(failed_columns) > 0:
            raise RuntimeError(
                "Groupby operation could not be performed on columns "
                f"{failed_columns}. Please drop them before calling groupby."
            )

    def any(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).any()
        if not (self.df.drop(columns=self.keys).dtypes == "bool").all():
            raise ValueError("Expected boolean types")
        self._validate_result(result)
        return PandasDataFrame(result)

    def all(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).all()
        if not (self.df.drop(columns=self.keys).dtypes == "bool").all():
            raise ValueError("Expected boolean types")
        self._validate_result(result)
        return PandasDataFrame(result)

    def min(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).min()
        self._validate_result(result)
        return PandasDataFrame(result)

    def max(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).max()
        self._validate_result(result)
        return PandasDataFrame(result)

    def sum(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).sum()
        self._validate_result(result)
        return PandasDataFrame(result)

    def prod(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).prod()
        self._validate_result(result)
        return PandasDataFrame(result)

    def median(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).median()
        self._validate_result(result)
        return PandasDataFrame(result)

    def mean(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).mean()
        self._validate_result(result)
        return PandasDataFrame(result)

    def std(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).std()
        self._validate_result(result)
        return PandasDataFrame(result)

    def var(self, skipna: bool = True) -> PandasDataFrame:
        result = self.df.groupby(self.keys, as_index=False).var()
        self._validate_result(result)
        return PandasDataFrame(result)


class PandasDataFrame:
    # Not technically part of the standard

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._validate_columns(dataframe.columns)  # type: ignore[arg-type]
        self.dataframe = dataframe

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                raise ValueError(
                    f"Expected unique column names, got {col} {count} time(s)"
                )
        for col in columns:
            if not isinstance(col, str):
                raise TypeError(
                    f"Expected column names to be of type str, got {col} "
                    f"of type {type(col)}"
                )

    def _validate_index(self, index: pd.Index) -> None:
        pd.testing.assert_index_equal(self.dataframe.index, index)

    def _validate_comparand(self, other: PandasDataFrame) -> None:
        if isinstance(other, PandasDataFrame) and not (
            self.dataframe.index.equals(other.dataframe.index)
            and self.dataframe.shape == other.dataframe.shape
            and self.dataframe.columns.equals(other.dataframe.columns)
        ):
            raise ValueError(
                "Expected DataFrame with same length, matching columns, "
                "and matching index."
            )

    def _validate_booleanness(self) -> None:
        if not (self.dataframe.dtypes == "bool").all():
            raise NotImplementedError(
                "'any' can only be called on DataFrame " "where all dtypes are 'bool'"
            )

    # In the standard

    @classmethod
    def from_dict(cls, data: dict[str, PandasColumn]) -> PandasDataFrame:
        return cls(
            pd.DataFrame({label: column._series for label, column in data.items()})
        )

    def groupby(self, keys: Sequence[str]) -> PandasGroupBy:
        if not isinstance(keys, collections.abc.Sequence):
            raise TypeError(f"Expected sequence of strings, got: {type(keys)}")
        for key in keys:
            if key not in self.get_column_names():
                raise KeyError(f"key {key} not present in DataFrame's columns")
        return PandasGroupBy(self.dataframe, keys)

    def get_column_by_name(self, name: str) -> PandasColumn:
        if not isinstance(name, str):
            raise TypeError(f"Expected str, got: {type(name)}")
        return PandasColumn(self.dataframe.loc[:, name])

    def get_columns_by_name(self, names: Sequence[str]) -> PandasDataFrame:
        self._validate_columns(names)
        return PandasDataFrame(self.dataframe.loc[:, list(names)])

    def get_rows(self, indices: Sequence[int]) -> PandasDataFrame:
        if not isinstance(indices, collections.abc.Sequence):
            raise TypeError(f"Expected Sequence of int, got {type(indices)}")
        return PandasDataFrame(
            self.dataframe.iloc[list(indices), :].reset_index(drop=True)
        )

    def slice_rows(self, start: int, stop: int, step: int) -> PandasDataFrame:
        return PandasDataFrame(
            self.dataframe.iloc[start:stop:step].reset_index(drop=True)
        )

    def get_rows_by_mask(self, mask: PandasColumn) -> PandasDataFrame:
        series = mask._series
        self._validate_index(series.index)
        return PandasDataFrame(self.dataframe.loc[series, :].reset_index(drop=True))

    def insert(self, loc: int, label: str, value: PandasColumn) -> PandasDataFrame:
        series = value._series
        self._validate_index(series.index)
        before = self.dataframe.iloc[:, :loc]
        after = self.dataframe.iloc[:, loc:]
        to_insert = value._series.rename(label)
        return PandasDataFrame(pd.concat([before, to_insert, after], axis=1))

    def drop_column(self, label: str) -> PandasDataFrame:
        if not isinstance(label, str):
            raise TypeError(f"Expected str, got: {type(label)}")
        return PandasDataFrame(self.dataframe.drop(label, axis=1))

    def set_column(self, label: str, value: PandasColumn) -> PandasDataFrame:
        columns = self.get_column_names()
        if label in columns:
            idx: int = columns.index(label)
            return self.drop_column(label).insert(idx, label, value)
        return self.insert(len(columns), label, value)

    def rename_columns(self, mapping: Mapping[str, str]) -> PandasDataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError(f"Expected Mapping, got: {type(mapping)}")
        return PandasDataFrame(self.dataframe.rename(columns=mapping))

    def get_column_names(self) -> Sequence[str]:
        return self.dataframe.columns.tolist()

    def __iter__(self) -> NoReturn:
        raise NotImplementedError()

    def __eq__(self, other: PandasDataFrame) -> PandasDataFrame:  # type: ignore[override]
        self._validate_comparand(other)
        return PandasDataFrame(self.dataframe.__eq__(other.dataframe))

    def __ne__(self, other: PandasDataFrame) -> PandasDataFrame:  # type: ignore[override]
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__ne__(other.dataframe)))

    def __ge__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__ge__(other.dataframe)))

    def __gt__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__gt__(other.dataframe)))

    def __le__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__le__(other.dataframe)))

    def __lt__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__lt__(other.dataframe)))

    def __add__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__add__(other.dataframe)))

    def __sub__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__sub__(other.dataframe)))

    def __mul__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__mul__(other.dataframe)))

    def __truediv__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__truediv__(other.dataframe)))

    def __floordiv__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__floordiv__(other.dataframe)))

    def __pow__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__pow__(other.dataframe)))

    def __mod__(self, other: PandasDataFrame) -> PandasDataFrame:
        self._validate_comparand(other)
        return PandasDataFrame((self.dataframe.__mod__(other.dataframe)))

    def __divmod__(
        self, other: PandasDataFrame
    ) -> tuple[PandasDataFrame, PandasDataFrame]:
        self._validate_comparand(other)
        quotient, remainder = self.dataframe.__divmod__(other.dataframe)
        return PandasDataFrame(quotient), PandasDataFrame(remainder)

    def any(self) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(self.dataframe.any().to_frame().T)

    def all(self) -> PandasDataFrame:
        self._validate_booleanness()
        return PandasDataFrame(self.dataframe.all().to_frame().T)

    def isnull(self) -> PandasDataFrame:
        raise NotImplementedError()

    def isnan(self) -> PandasDataFrame:
        return PandasDataFrame(self.dataframe.isna())
