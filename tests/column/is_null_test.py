from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_1

if TYPE_CHECKING:
    import pytest


def test_column_is_null_1(library: str, request: pytest.FixtureRequest) -> None:
    df = nan_dataframe_1(library)
    ser = df.get_column_by_name("a")
    result = df.insert(0, "result", ser.is_null())
    result_pd = interchange_to_pandas(result, library)["result"]
    if library == "pandas-numpy":
        expected = pd.Series([False, False, True], name="result")
    else:
        expected = pd.Series([False, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_is_null_2(library: str, request: pytest.FixtureRequest) -> None:
    df = null_dataframe_1(library)
    ser = df.get_column_by_name("a")
    result = df.insert(0, "result", ser.is_null())
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
