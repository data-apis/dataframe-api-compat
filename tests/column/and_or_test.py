from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_column_and(library: str, request: pytest.FixtureRequest) -> None:
    df = bool_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b")
    result = df.insert(0, "result", ser & other)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or(library: str) -> None:
    df = bool_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b")
    result = df.insert(0, "result", ser | other)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_and_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    df = bool_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = True
    result = df.insert(0, "result", ser & other)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    df = bool_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = True
    result = df.insert(0, "result", ser | other)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
