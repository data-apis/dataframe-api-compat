from __future__ import annotations
from pandas_standard.pandas_standard import PandasDataFrame, PandasColumn
import pandas as pd

from typing import (
    Any,
    Sequence,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from dataframe_api import (
        DTypeT,
        DType,
    )


class Int64:
    ...


class Float64:
    ...


class Bool:
    ...


def _map_standard_dtype_to_pandas_dtype(dtype: DType) -> Any:
    if isinstance(dtype, Int64):
        return pd.Int64Dtype()
    if isinstance(dtype, Float64):
        return pd.Float64Dtype()
    if isinstance(dtype, Bool):
        return pd.BooleanDtype()
    raise AssertionError(f"Unknown dtype: {dtype}")


def convert_to_standard_compliant_dataframe(df: pd.DataFrame) -> PandasDataFrame:
    return PandasDataFrame(df)


class PandasInt64(Int64):
    ...


class PandasFloat64(Float64):
    ...


class PandasBool(Bool):
    ...


DTYPE_MAP = {
    "int64": PandasInt64(),
    "Int64": PandasInt64(),
    "float64": PandasFloat64(),
    "Float64": PandasFloat64(),
    "bool": PandasBool(),
}


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
    sequence: Sequence[DTypeT], dtype: DTypeT
) -> PandasColumn[DTypeT]:
    ser = pd.Series(sequence, dtype=_map_standard_dtype_to_pandas_dtype(dtype))
    return PandasColumn(ser)


def dataframe_from_dict(data: dict[str, PandasColumn[Any]]) -> PandasDataFrame:
    return PandasDataFrame(
        pd.DataFrame({label: column.column for label, column in data.items()})
    )
