from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import pandas as pd

from dataframe_api_compat.pandas_standard.pandas_standard import LATEST_API_VERSION
from dataframe_api_compat.pandas_standard.pandas_standard import null
from dataframe_api_compat.pandas_standard.pandas_standard import PandasDataFrame
from dataframe_api_compat.pandas_standard.pandas_standard import PandasExpression
from dataframe_api_compat.pandas_standard.pandas_standard import PandasGroupBy

if TYPE_CHECKING:
    from collections.abc import Sequence


def col(name: str) -> PandasExpression:
    return PandasExpression(base_call=lambda df: df.loc[:, name])


Expression = PandasExpression
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


class String:
    ...


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
    "object": String(),
    "string": String(),
}


def map_standard_dtype_to_pandas_dtype(dtype: Any) -> Any:
    if isinstance(dtype, Int64):
        return "int64"
    if isinstance(dtype, Int32):
        return "int32"
    if isinstance(dtype, Int16):
        return "int16"
    if isinstance(dtype, Int8):
        return "int8"
    if isinstance(dtype, UInt64):
        return "uint64"
    if isinstance(dtype, UInt32):
        return "uint32"
    if isinstance(dtype, UInt16):
        return "uint16"
    if isinstance(dtype, UInt8):
        return "uint8"
    if isinstance(dtype, Float64):
        return "float64"
    if isinstance(dtype, Float32):
        return "float32"
    if isinstance(dtype, Bool):
        return "bool"
    if isinstance(dtype, String):
        return "object"
    raise AssertionError(f"Unknown dtype: {dtype}")


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame, api_version: str | None = None
) -> PandasDataFrame:
    if api_version is None:
        api_version = LATEST_API_VERSION
    return PandasDataFrame(df, api_version=api_version)


def concat(dataframes: Sequence[PandasDataFrame]) -> PandasDataFrame:
    dtypes = dataframes[0].dataframe.dtypes
    dfs = []
    api_versions = set()
    for _df in dataframes:
        try:
            pd.testing.assert_series_equal(_df.dataframe.dtypes, dtypes)
        except Exception as exc:
            raise ValueError("Expected matching columns") from exc
        else:
            dfs.append(_df.dataframe)
        api_versions.add(_df._api_version)
    if len(api_versions) > 1:  # pragma: no cover
        raise ValueError(f"Multiple api versions found: {api_versions}")
    return PandasDataFrame(
        pd.concat(
            dfs,
            axis=0,
            ignore_index=True,
        ),
        api_version=api_versions.pop(),
    )


def dataframe_from_2d_array(
    data: Any,
    *,
    names: Sequence[str],
    dtypes: dict[str, Any],
    api_version: str | None = None,
) -> PandasDataFrame:  # pragma: no cover
    df = pd.DataFrame(data, columns=names).astype(  # type: ignore[call-overload]
        {key: map_standard_dtype_to_pandas_dtype(value) for key, value in dtypes.items()}
    )
    return PandasDataFrame(df, api_version=api_version or LATEST_API_VERSION)


# def dataframe_from_dict(
#     data: dict[str, PandasExpression, api_version: str | None = None
# ) -> PandasDataFrame:
#     # todo: figure out what to do with this one
#     for _, col in data.items():
#         if not isinstance(col, PandasExpression):  # pragma: no cover
#             raise TypeError(f"Expected PandasColumn, got {type(col)}")
#     return PandasDataFrame(
#         pd.DataFrame(
#             {label: column.column.rename(label) for label, column in data.items()}
#         ),
#         api_version=api_version or LATEST_API_VERSION,
#     )


def is_null(value: Any) -> bool:
    return value is null


def is_dtype(dtype: Any, kind: str | tuple[str, ...]) -> bool:
    if isinstance(kind, str):
        kind = (kind,)
    dtypes: set[Any] = set()
    for _kind in kind:
        if _kind == "bool":
            dtypes.add(Bool)
        if _kind == "signed integer" or _kind == "integral" or _kind == "numeric":
            dtypes |= {Int64, Int32, Int16, Int8}
        if _kind == "unsigned integer" or _kind == "integral" or _kind == "numeric":
            dtypes |= {UInt64, UInt32, UInt16, UInt8}
        if _kind == "floating" or _kind == "numeric":
            dtypes |= {Float64, Float32}
        if _kind == "string":
            dtypes.add(String)
    return isinstance(dtype, tuple(dtypes))


def any_rowwise(keys: list[str], *, skip_nulls: bool = True) -> PandasExpression:
    return PandasExpression(base_call=lambda df: df.all(axis=1))


def all_rowwise(keys: list[str], *, skip_nulls: bool = True) -> PandasExpression:
    return PandasExpression(base_call=lambda df: df.all(axis=1))
