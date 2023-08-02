from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_series_1


def test_float_powers_column(library: str, request: pytest.FixtureRequest) -> None:
    ser = integer_series_1(library, request)
    other = integer_series_1(library, request) * 1.0
    result = ser.__pow__(other)
    namespace = ser.__column_namespace__()
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": (result).rename("result")}).dataframe
    )["result"]
    expected = pd.Series([1.0, 4.0, 27.0], name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_float_powers_scalar_column(library: str, request: pytest.FixtureRequest) -> None:
    ser = integer_series_1(library, request)
    other = 1.0
    result = ser.__pow__(other)
    namespace = ser.__column_namespace__()
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": (result).rename("result")}).dataframe
    )["result"]
    expected = pd.Series([1.0, 2.0, 3.0], name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_negative_powers_column(library: str, request: pytest.FixtureRequest) -> None:
    ser = integer_series_1(library, request)
    other = integer_series_1(library, request) * -1
    with pytest.raises(ValueError):
        ser.__pow__(-1)
    with pytest.raises(ValueError):
        ser.__pow__(other)
