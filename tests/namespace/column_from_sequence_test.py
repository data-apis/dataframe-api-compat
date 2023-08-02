from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_series_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1, 2, 3], "Int64", pd.Series([1, 2, 3], dtype="int64", name="result")),
        ([1, 2, 3], "Int32", pd.Series([1, 2, 3], dtype="int32", name="result")),
        (
            [1.0, 2.0, 3.0],
            "Float64",
            pd.Series([1, 2, 3], dtype="float64", name="result"),
        ),
        (
            [1.0, 2.0, 3.0],
            "Float32",
            pd.Series([1, 2, 3], dtype="float32", name="result"),
        ),
        (
            [True, False, True],
            "Bool",
            pd.Series([True, False, True], dtype=bool, name="result"),
        ),
    ],
)
def test_column_from_sequence(
    library: str,
    values: list[Any],
    dtype: str,
    expected: pd.Series[Any],
    request: pytest.FixtureRequest,
) -> None:
    ser = integer_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {
            "result": namespace.column_from_sequence(
                values, dtype=getattr(namespace, dtype)(), name="result"
            )
        }
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)
