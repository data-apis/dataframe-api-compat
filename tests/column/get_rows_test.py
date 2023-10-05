from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_column_get_rows(library: str) -> None:
    df = integer_dataframe_1(library).collect()
    ser = df.get_column_by_name("a")
    namespace = ser.__column_namespace__()
    indices = namespace.column_from_sequence(
        [0, 2, 1], dtype=namespace.Int64(), name="result"
    )
    result = namespace.dataframe_from_dict(
        {"result": (ser.get_rows(indices)).rename("result")}
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([1, 3, 2], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_expression_get_rows(library: str) -> None:
    df = integer_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    ser = col("a")
    ser.__column_namespace__()
    indices = col("a") - 1
    result = df.select(ser.get_rows(indices).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
