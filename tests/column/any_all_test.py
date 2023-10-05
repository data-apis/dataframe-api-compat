from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import bool_series_1
from tests.utils import interchange_to_pandas


def test_column_any(library: str) -> None:
    ser = bool_series_1(library)
    result = ser.any()
    assert result


def test_column_all(library: str) -> None:
    ser = bool_series_1(library)
    result = ser.all()
    assert not result


def test_expr_any(library: str) -> None:
    df = bool_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(col("a").any())
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_expr_all(library: str) -> None:
    df = bool_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(col("a").all())
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [False]})
    pd.testing.assert_frame_equal(result_pd, expected)
