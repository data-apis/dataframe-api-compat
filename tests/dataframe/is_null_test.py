from tests.utils import nan_dataframe_2, null_dataframe_1
import pandas as pd
import pytest


def test_is_null_1(library: str) -> None:
    df = nan_dataframe_2(library)
    result = df.is_null().dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame({"a": [False, False, False]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_is_null_2(library: str, request: pytest.FixtureRequest) -> None:
    df = null_dataframe_1(library, request)
    result = df.is_null().dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result_pd, expected)