import pandas as pd
import pytest

from tests.utils import nan_series_1, null_series_1


def test_column_is_null_1(library: str) -> None:
    ser = nan_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.is_null()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([False, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_is_null_2(library: str, request: pytest.FixtureRequest) -> None:
    ser = null_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.is_null()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
