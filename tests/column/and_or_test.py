from __future__ import annotations

import pandas as pd

from tests.utils import bool_series_1
from tests.utils import bool_series_2
from tests.utils import convert_series_to_pandas_numpy


def test_column_and(library: str) -> None:
    ser = bool_series_1(library)
    other = bool_series_2(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser & other).rename("result")})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or(library: str) -> None:
    ser = bool_series_1(library)
    other = bool_series_2(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser | other).rename("result")})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_and_with_scalar(library: str) -> None:
    ser = bool_series_1(library)
    other = True
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser & other).rename("result")})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or_with_scalar(library: str) -> None:
    ser = bool_series_1(library)
    other = True
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser | other).rename("result")})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
