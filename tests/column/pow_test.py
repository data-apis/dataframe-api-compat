import pytest
import pandas as pd
from tests.utils import integer_series_1
from tests.utils import convert_series_to_pandas_numpy


def test_float_powers_column(library: str) -> None:
    ser = integer_series_1(library)
    other = integer_series_1(library) * 1.0
    result = ser.__pow__(other)
    namespace = ser.__column_namespace__()
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result}).dataframe
    )["result"]
    expected = pd.Series([1.0, 4.0, 27.0], name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_float_powers_scalar_column(library: str) -> None:
    ser = integer_series_1(library)
    other = 1.0
    result = ser.__pow__(other)
    namespace = ser.__column_namespace__()
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result}).dataframe
    )["result"]
    expected = pd.Series([1.0, 2.0, 3.0], name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_negative_powers_column(library: str) -> None:
    ser = integer_series_1(library)
    other = integer_series_1(library) * -1
    with pytest.raises(ValueError):
        ser.__pow__(-1)
    with pytest.raises(ValueError):
        ser.__pow__(other)