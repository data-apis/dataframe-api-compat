from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import pandas as pd

from dataframe_api_compat.pandas_standard.pandas_standard import PandasColumn
from dataframe_api_compat.pandas_standard.pandas_standard import PandasDataFrame
from dataframe_api_compat.pandas_standard.pandas_standard import PandasGroupBy

if TYPE_CHECKING:
    from collections.abc import Sequence

Column = PandasColumn
DataFrame = PandasDataFrame
GroupBy = PandasGroupBy


class Int64:
    ...


class Int32:
    ...


class Int16:
    ...


class Int8:
    ...


class UInt64:
    ...


class UInt32:
    ...


class UInt16:
    ...


class UInt8:
    ...


class Float64:
    ...


class Float32:
    ...


class Bool:
    ...


null = pd.NA

DTYPE_MAP = {
    "int64": Int64(),
    "Int64": Int64(),
    "int32": Int32(),
    "Int32": Int32(),
    "int16": Int16(),
    "Int16": Int16(),
    "int8": Int8(),
    "Int8": Int8(),
    "uint64": UInt64(),
    "UInt64": UInt64(),
    "uint32": UInt32(),
    "UInt32": UInt32(),
    "uint16": UInt16(),
    "UInt16": UInt16(),
    "uint8": UInt8(),
    "UInt8": UInt8(),
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


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame, api_version: str | None = None
) -> PandasDataFrame:
    if api_version is None:
        api_version = "2023.08"
    if api_version != "2023.08":  # pragma: no cover
        raise ValueError(
            f"Unknown api_version: {api_version}. Expected: '2023.08', or None"
        )
    return PandasDataFrame(df)


def convert_to_standard_compliant_column(
    df: pd.Series[Any], api_version: str | None = None
) -> PandasColumn[Any]:
    if api_version is None:
        api_version = "2023.08"
    if api_version != "2023.08":  # pragma: no cover
        raise ValueError(
            f"Unknown api_version: {api_version}. Expected: '2023.08', or None"
        )
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
    sequence: Sequence[Any],
    *,
    dtype: Any,
    name: str,
) -> PandasColumn[Any]:
    ser = pd.Series(sequence, dtype=map_standard_dtype_to_pandas_dtype(dtype), name=name)
    return PandasColumn(ser)


def column_from_1d_array(
    data: Any, *, dtype: Any, name: str | None = None
) -> PandasColumn[Any]:  # pragma: no cover
    ser = pd.Series(data, dtype=map_standard_dtype_to_pandas_dtype(dtype), name=name)
    return PandasColumn(ser)


def dataframe_from_2d_array(
    data: Any, *, names: Sequence[str], dtypes: dict[str, Any]
) -> PandasDataFrame:  # pragma: no cover
    df = pd.DataFrame(data, columns=names).astype(  # type: ignore[call-overload]
        {key: map_standard_dtype_to_pandas_dtype(value) for key, value in dtypes.items()}
    )
    return PandasDataFrame(df)


def dataframe_from_dict(data: dict[str, PandasColumn[Any]]) -> PandasDataFrame:
    for _, col in data.items():
        if not isinstance(col, PandasColumn):  # pragma: no cover
            raise TypeError(f"Expected PandasColumn, got {type(col)}")
    return PandasDataFrame(
        pd.DataFrame(
            {label: column.column.rename(label) for label, column in data.items()}
        )
    )


def is_null(value: Any) -> bool:
    return value is pd.NA
