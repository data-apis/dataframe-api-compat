from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import Any

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("values", "dtype", "kwargs", "expected"),
    [
        ([1, 2, 3], "Int64", {}, pd.Series([1, 2, 3], dtype="int64", name="result")),
        ([1, 2, 3], "Int32", {}, pd.Series([1, 2, 3], dtype="int32", name="result")),
        ([1, 2, 3], "Int16", {}, pd.Series([1, 2, 3], dtype="int16", name="result")),
        ([1, 2, 3], "Int8", {}, pd.Series([1, 2, 3], dtype="int8", name="result")),
        ([1, 2, 3], "UInt64", {}, pd.Series([1, 2, 3], dtype="uint64", name="result")),
        ([1, 2, 3], "UInt32", {}, pd.Series([1, 2, 3], dtype="uint32", name="result")),
        ([1, 2, 3], "UInt16", {}, pd.Series([1, 2, 3], dtype="uint16", name="result")),
        ([1, 2, 3], "UInt8", {}, pd.Series([1, 2, 3], dtype="uint8", name="result")),
        (
            [1.0, 2.0, 3.0],
            "Float64",
            {},
            pd.Series([1, 2, 3], dtype="float64", name="result"),
        ),
        (
            [1.0, 2.0, 3.0],
            "Float32",
            {},
            pd.Series([1, 2, 3], dtype="float32", name="result"),
        ),
        (
            [True, False, True],
            "Bool",
            {},
            pd.Series([True, False, True], dtype=bool, name="result"),
        ),
        (
            ["express", "yourself"],
            "String",
            {},
            pd.Series(["express", "yourself"], dtype=object, name="result"),
        ),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "Datetime",
            {"time_unit": "us"},
            pd.Series(
                [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                dtype="datetime64[us]",
                name="result",
            ),
        ),
        (
            [timedelta(1), timedelta(2)],
            "Duration",
            {"time_unit": "us"},
            pd.Series(
                [timedelta(1), timedelta(2)],
                dtype="timedelta64[us]",
                name="result",
            ),
        ),
    ],
)
def test_column_from_sequence(
    library: str,
    values: list[Any],
    dtype: str,
    kwargs: dict[str, Any],
    expected: pd.Series[Any],
) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = df.col("a")
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_columns(
        namespace.column_from_sequence(
            values,
            dtype=getattr(namespace, dtype)(**kwargs),
            name="result",
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    pd.testing.assert_series_equal(result_pd, expected)
