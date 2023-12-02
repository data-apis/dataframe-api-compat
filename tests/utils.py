from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import cast

import pandas as pd
import polars as pl

import dataframe_api_compat.pandas_standard
import dataframe_api_compat.polars_standard

DType = TypeVar("DType")

if TYPE_CHECKING:
    from dataframe_api import DataFrame

POLARS_VERSION = tuple(int(v) for v in pl.__version__.split("."))
PANDAS_VERSION = tuple(int(v) for v in pd.__version__.split("."))


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame | pl.DataFrame,
    api_version: str | None = None,
) -> DataFrame:
    # TODO: type return
    if isinstance(df, pd.DataFrame):
        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version,
            )
        )
    elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
                df_lazy,
                api_version=api_version,
            )
        )
    else:  # pragma: no cover
        msg = f"Got unexpected type: {type(df)}"
        raise AssertionError(msg)


def convert_dataframe_to_pandas_numpy(df: pd.DataFrame) -> pd.DataFrame:
    conversions = {
        "boolean": "bool",
        "Int64": "int64",
        "Float64": "float64",
    }
    for column in df.columns:
        dtype = str(df.dtypes[column])
        if dtype in conversions:
            try:
                df[column] = df[column].to_numpy(
                    conversions[dtype],
                    na_value=float("nan"),
                )
            except ValueError:
                # cannot convert float NaN to integer
                assert dtype == "Int64"
                df[column] = df[column].to_numpy("float64", na_value=float("nan"))
    return df


def integer_dataframe_1(library: str, api_version: str | None = None) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def integer_dataframe_2(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def integer_dataframe_3(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]},
            dtype="int64",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]},
            dtype="Int64",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def integer_dataframe_4(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame(
            {"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]},
            dtype="int64",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame(
            {"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]},
            dtype="Int64",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def integer_dataframe_5(library: str, api_version: str | None = None) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 1], "b": [4, 3]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 1], "b": [4, 3]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1, 1], "b": [4, 3]})
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def integer_dataframe_6(library: str, api_version: str | None = None) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 1, 2]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 1, 2]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 1, 2]})
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def integer_dataframe_7(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 4]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 4]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 4]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def nan_dataframe_1(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1.0, 2.0, float("nan")]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1.0, 2.0, 0.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [1.0, 1.0, 0.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1.0, 2.0, float("nan")]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def nan_dataframe_2(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [0.0, 1.0, float("nan")]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [0.0, 1.0, 0.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [1.0, 1.0, 0.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [0.0, 1.0, float("nan")]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def null_dataframe_1(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1.0, 2.0, float("nan")]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1.0, 2.0, pd.NA]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1.0, 2.0, None]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def null_dataframe_2(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame(
            {"a": [1.0, -1.0, float("nan")], "b": [1.0, -1.0, float("nan")]},
            dtype="float64",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame(
            {"a": [1.0, 0.0, pd.NA], "b": [1.0, 1.0, pd.NA]},
            dtype="Float64",
        )
        return convert_to_standard_compliant_dataframe(df / df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [1.0, float("nan"), None], "b": [1.0, 1.0, None]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def bool_dataframe_1(library: str, api_version: str = "2023.09-beta") -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame(
            {"a": [True, True, False], "b": [True, True, True]},
            dtype="bool",
        )
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "pandas-nullable":
        df = pd.DataFrame(
            {"a": [True, True, False], "b": [True, True, True]},
            dtype="boolean",
        )
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [True, True, False], "b": [True, True, True]})
        return convert_to_standard_compliant_dataframe(df, api_version=api_version)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def bool_dataframe_2(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2],
                "b": [False, True, True, True],
                "c": [True, False, False, False],
            },
        ).astype({"key": "int64", "b": "bool", "c": "bool"})
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2],
                "b": [False, True, True, True],
                "c": [True, False, False, False],
            },
        ).astype({"key": "Int64", "b": "boolean", "c": "boolean"})
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame(
            {
                "key": [1, 1, 2, 2],
                "b": [False, True, True, True],
                "c": [True, False, False, False],
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def bool_dataframe_3(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame(
            {"a": [False, False], "b": [False, True], "c": [True, True]},
            dtype="bool",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame(
            {"a": [False, False], "b": [False, True], "c": [True, True]},
            dtype="boolean",
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [False, False], "b": [False, True], "c": [True, True]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def float_dataframe_1(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [2.0, 3.0]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [2.0, 3.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame({"a": [2.0, 3.0]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def float_dataframe_2(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [2.0, 1.0]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [2.0, 1.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":  # pragma: no cover
        df = pl.DataFrame({"a": [2.0, 1.0]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def float_dataframe_3(library: str) -> DataFrame:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [float("nan"), 2.0]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [0.0, 2.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [0.0, 1.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other)
    if library == "polars-lazy":  # pragma: no cover
        df = pl.DataFrame({"a": [float("nan"), 2.0]})
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def temporal_dataframe_1(library: str) -> DataFrame:
    if library in ["pandas-numpy", "pandas-nullable"]:
        df = pd.DataFrame(
            {
                "a": [
                    datetime(2020, 1, 1, 1, 2, 1, 123000),
                    datetime(2020, 1, 2, 3, 1, 2, 321000),
                    datetime(2020, 1, 3, 5, 4, 9, 987000),
                ],
                "b": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "c": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "d": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "e": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "f": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "index": [0, 1, 2],
            },
        ).astype(
            {
                "a": "datetime64[ms]",
                "b": "timedelta64[ms]",
                "c": "datetime64[us]",
                "d": "timedelta64[us]",
                "e": "datetime64[ns]",
                "f": "timedelta64[ns]",
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame(
            {
                "a": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "b": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "c": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "d": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "e": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "f": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "index": [0, 1, 2],
            },
            schema={
                "a": pl.Datetime("ms"),
                "b": pl.Duration("ms"),
                "c": pl.Datetime("us"),
                "d": pl.Duration("us"),
                "e": pl.Datetime("ns"),
                "f": pl.Duration("ns"),
                "index": pl.Int64,
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)


def interchange_to_pandas(result: Any) -> pd.DataFrame:
    if isinstance(result.dataframe, pl.LazyFrame):
        df = result.dataframe.collect()
        df = df.to_pandas()
    elif isinstance(result.dataframe, pl.DataFrame):
        df = result.dataframe
        df = df.to_pandas()
    else:
        df = result.dataframe
    df = convert_dataframe_to_pandas_numpy(df)
    return cast(pd.DataFrame, df)


def mixed_dataframe_1(library: str) -> DataFrame:
    df: Any
    data = {
        "a": [1, 2, 3],
        "b": [1, 2, 3],
        "c": [1, 2, 3],
        "d": [1, 2, 3],
        "e": [1, 2, 3],
        "f": [1, 2, 3],
        "g": [1, 2, 3],
        "h": [1, 2, 3],
        "i": [1.0, 2.0, 3.0],
        "j": [1.0, 2.0, 3.0],
        "k": [True, False, True],
        "l": ["a", "b", "c"],
        "m": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "n": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "o": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "p": [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
        "q": [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
    }
    if library == "pandas-numpy":
        df = pd.DataFrame(data).astype(
            {
                "a": "int64",
                "b": "int32",
                "c": "int16",
                "d": "int8",
                "e": "uint64",
                "f": "uint32",
                "g": "uint16",
                "h": "uint8",
                "i": "float64",
                "j": "float32",
                "k": "bool",
                "l": "object",
                "m": "datetime64[s]",
                "n": "datetime64[ms]",
                "o": "datetime64[us]",
                "p": "timedelta64[ms]",
                "q": "timedelta64[us]",
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "pandas-nullable":
        df = pd.DataFrame(data).astype(
            {
                "a": "Int64",
                "b": "Int32",
                "c": "Int16",
                "d": "Int8",
                "e": "UInt64",
                "f": "UInt32",
                "g": "UInt16",
                "h": "UInt8",
                "i": "Float64",
                "j": "Float32",
                "k": "bool",
                "l": "string[pyarrow]",
                "m": "datetime64[s]",
                "n": "datetime64[ms]",
                "o": "datetime64[us]",
                "p": "timedelta64[ms]",
                "q": "timedelta64[us]",
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    if library == "polars-lazy":
        df = pl.DataFrame(
            data,
            schema={
                "a": pl.Int64,
                "b": pl.Int32,
                "c": pl.Int16,
                "d": pl.Int8,
                "e": pl.UInt64,
                "f": pl.UInt32,
                "g": pl.UInt16,
                "h": pl.UInt8,
                "i": pl.Float64,
                "j": pl.Float32,
                "k": pl.Boolean,
                "l": pl.Utf8,
                "m": pl.Datetime("ms"),
                "n": pl.Datetime("ms"),
                "o": pl.Datetime("us"),
                "p": pl.Duration("ms"),
                "q": pl.Duration("us"),
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    msg = f"Got unexpected library: {library}"  # pragma: no cover
    raise AssertionError(msg)
