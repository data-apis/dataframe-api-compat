from __future__ import annotations
from dataframe_api_compat.pandas_standard.pandas_standard import (
    PandasDataFrame,
    PandasColumn,
)
import pandas as pd

from typing import (
    Any,
    Sequence,
)


class Int64:
    ...


class Int32:
    ...


class Float64:
    ...


class Float32:
    ...


class Bool:
    ...


DTYPE_MAP = {
    "int64": Int64(),
    "Int64": Int64(),
    "int32": Int32(),
    "Int32": Int32(),
    "float64": Float64(),
    "Float64": Float64(),
    "float32": Float32(),
    "Float32": Float32(),
    "bool": Bool(),
    "boolean": Bool(),
}


def map_standard_dtype_to_pandas_dtype(dtype: Any) -> Any:
    if isinstance(dtype, Int64):
        return pd.Int64Dtype()
    if isinstance(dtype, Int32):
        return pd.Int32Dtype()
    if isinstance(dtype, Float64):
        return pd.Float64Dtype()
    if isinstance(dtype, Float32):
        return pd.Float32Dtype()
    if isinstance(dtype, Bool):
        return pd.BooleanDtype()
    raise AssertionError(f"Unknown dtype: {dtype}")


def convert_to_standard_compliant_dataframe(df: pd.DataFrame) -> PandasDataFrame:
    return PandasDataFrame(df)


def convert_to_standard_compliant_column(df: pd.Series[Any]) -> PandasColumn[Any]:
    return PandasColumn(df)


def concat(dataframes: Sequence[PandasDataFrame]) -> PandasDataFrame:
    dtypes = dataframes[0].dataframe.dtypes
    dfs = []
    for _df in dataframes:
        try:
            pd.testing.assert_series_equal(_df.dataframe.dtypes, dtypes)
        except Exception as exc:
            raise ValueError("Expected matching columns") from exc
        else:
            dfs.append(_df.dataframe)
    return PandasDataFrame(
        pd.concat(
            dfs,
            axis=0,
            ignore_index=True,
        )
    )


def column_from_sequence(
    sequence: Sequence[Any], *, dtype: Any, name: str | None = None
) -> PandasColumn[Any]:
    ser = pd.Series(sequence, dtype=map_standard_dtype_to_pandas_dtype(dtype))
    return PandasColumn(ser, name=name)


def dataframe_from_dict(data: dict[str, PandasColumn[Any]]) -> PandasDataFrame:
    for col_name, col in data.items():
        if not isinstance(col, PandasColumn):  # pragma: no cover
            raise TypeError(f"Expected PandasColumn, got {type(col)}")
        if col.name is not None and col_name != col.name:
            raise ValueError(f"Expected column name {col_name}, got {col.name}")
    return PandasDataFrame(
        pd.DataFrame({label: column.column for label, column in data.items()})
    )
