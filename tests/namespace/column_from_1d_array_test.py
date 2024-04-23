from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1
from tests.utils import pandas_version
from tests.utils import polars_version


@pytest.mark.parametrize(
    ("pandas_dtype", "column_dtype"),
    [
        ("float64", "Float64"),
        ("float32", "Float32"),
        ("int64", "Int64"),
        ("int32", "Int32"),
        ("int16", "Int16"),
        ("int8", "Int8"),
        ("uint64", "UInt64"),
        ("uint32", "UInt32"),
        ("uint16", "UInt16"),
        ("uint8", "UInt8"),
    ],
)
def test_column_from_1d_array(
    library: BaseHandler,
    pandas_dtype: str,
    column_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).col("a").persist()
    ns = ser.__column_namespace__()
    arr = np.array([1, 2, 3], dtype=pandas_dtype)
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [1, 2, 3]
    compare_column_with_reference(
        result.col("result"),
        expected,
        dtype=getattr(ns, column_dtype),
    )


def test_column_from_1d_array_string(
    library: BaseHandler,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array(["a", "b", "c"])
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = ["a", "b", "c"]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.String)


def test_column_from_1d_array_bool(
    library: BaseHandler,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array([True, False, True])
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [True, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_datetime_from_1d_array(library: BaseHandler) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array([date(2020, 1, 1), date(2020, 1, 2)], dtype="datetime64[ms]")
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Datetime)


def test_duration_from_1d_array(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable") and pandas_version() < Version(
        "2.0.0",
    ):  # pragma: no cover
        pytest.skip(reason="pandas before non-nano")
    if library.name == "polars-lazy" and polars_version() < Version(
        "0.19.9",
    ):  # pragma: no cover
        pytest.skip(reason="upstream bug")

    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array([timedelta(1), timedelta(2)], dtype="timedelta64[ms]")
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    if library.name == "polars-lazy":
        # https://github.com/data-apis/dataframe-api/issues/329
        result = result.cast({"result": ns.Duration("ms")})
    expected = [timedelta(1), timedelta(2)]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Duration)
