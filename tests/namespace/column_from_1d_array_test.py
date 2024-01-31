from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest
from packaging.version import Version

from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


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
    library: str,
    pandas_dtype: str,
    column_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).persist().get_column("a")
    pdx = ser.__column_namespace__()
    arr = np.array([1, 2, 3], dtype=pandas_dtype)
    result = pdx.dataframe_from_columns(
        pdx.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [1, 2, 3]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=getattr(pdx, column_dtype),
    )


def test_column_from_1d_array_string(
    library: str,
) -> None:
    ser = integer_dataframe_1(library).persist().get_column("a")
    pdx = ser.__column_namespace__()
    arr = np.array(["a", "b", "c"])
    result = pdx.dataframe_from_columns(
        pdx.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = ["a", "b", "c"]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.String,
    )


def test_column_from_1d_array_bool(
    library: str,
) -> None:
    ser = integer_dataframe_1(library).persist().get_column("a")
    pdx = ser.__column_namespace__()
    arr = np.array([True, False, True])
    result = pdx.dataframe_from_columns(
        pdx.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [True, False, True]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.Bool,
    )


def test_datetime_from_1d_array(library: str) -> None:
    ser = integer_dataframe_1(library).persist().get_column("a")
    pdx = ser.__column_namespace__()
    arr = np.array([date(2020, 1, 1), date(2020, 1, 2)], dtype="datetime64[ms]")
    result = pdx.dataframe_from_columns(
        pdx.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.Datetime,
    )


@pytest.mark.skipif(
    Version("0.19.9") > POLARS_VERSION,
    reason="upstream bug",
)
@pytest.mark.skipif(
    Version("2.0.0") > PANDAS_VERSION,
    reason="pandas before non-nano",
)
def test_duration_from_1d_array(library: str) -> None:
    ser = integer_dataframe_1(library).persist().get_column("a")
    pdx = ser.__column_namespace__()
    arr = np.array([timedelta(1), timedelta(2)], dtype="timedelta64[ms]")
    result = pdx.dataframe_from_columns(
        pdx.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    if library == "polars-lazy":
        # https://github.com/data-apis/dataframe-api/issues/329
        result = result.cast({"result": pdx.Duration("ms")})
    expected = [timedelta(1), timedelta(2)]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.Duration,
    )
