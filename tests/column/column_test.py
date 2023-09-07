from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1


def test_column_column(library: str) -> None:
    namespace = integer_dataframe_1(library).__dataframe_namespace__()
    ser = namespace.column_from_sequence([1, 2, 3], name="a", dtype=namespace.Int64())
    result = ser.column
    if library == "polars-lazy":
        pd.testing.assert_series_equal(result.to_pandas(), pd.Series([1, 2, 3], name="a"))
    elif library == "pandas-numpy":  # noqa: SIM114
        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], name="a"))
    elif library == "pandas-nullable":
        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], name="a"))
    else:
        raise AssertionError()
