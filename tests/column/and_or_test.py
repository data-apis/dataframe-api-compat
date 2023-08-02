from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import bool_series_1
from tests.utils import bool_series_2
from tests.utils import convert_series_to_pandas_numpy
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_column_and(library: str, request: pytest.FixtureRequest) -> None:
    ser = bool_series_1(library, request)
    other = bool_series_2(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser & other).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or(library: str, request: pytest.FixtureRequest) -> None:
    ser = bool_series_1(library, request)
    other = bool_series_2(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser | other).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_and_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    ser = bool_series_1(library, request)
    other = True
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser & other).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    ser = bool_series_1(library, request)
    other = True
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser | other).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
