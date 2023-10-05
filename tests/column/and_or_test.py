from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_column_and(library: str) -> None:
    df = bool_dataframe_1(library, api_version="2023.09-beta").collect()
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b")
    result = df.assign((ser & other).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_expression_and(library: str) -> None:
    df = bool_dataframe_1(library, api_version="2023.09-beta")
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    other = namespace.col("b")
    result = df.assign((ser & other).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or(library: str) -> None:
    df = bool_dataframe_1(library).collect()
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b")
    result = df.assign((ser | other).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_and_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    df = bool_dataframe_1(library).collect()
    ser = df.get_column_by_name("a")
    other = True
    result = df.assign((ser & other).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    df = bool_dataframe_1(library).collect()
    ser = df.get_column_by_name("a")
    other = True
    result = df.assign((ser | other).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
