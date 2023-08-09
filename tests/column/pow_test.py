from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_float_powers_column(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b") * 1.0
    result = df.insert(0, "result", ser.__pow__(other))
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame(
        {"result": [1.0, 32.0, 729.0], "a": [1, 2, 3], "b": [4, 5, 6]}
    )
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_float_powers_scalar_column(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = 1.0
    result = df.insert(0, "result", ser.__pow__(other))
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"result": [1.0, 2.0, 3.0], "a": [1, 2, 3], "b": [4, 5, 6]})
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_int_powers_column(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b") * 1
    result = df.insert(0, "result", ser.__pow__(other))
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"result": [1, 32, 729], "a": [1, 2, 3], "b": [4, 5, 6]})
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_int_powers_scalar_column(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = 1
    result = df.insert(0, "result", ser.__pow__(other))
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"result": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6]})
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)
