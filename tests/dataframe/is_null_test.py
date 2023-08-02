from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import nan_dataframe_2
from tests.utils import null_dataframe_1


def test_is_null_1(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = nan_dataframe_2(library)
    result = df.is_null().dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame({"a": [False, False, False]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_is_null_2(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = null_dataframe_1(library, request)
    result = df.is_null().dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
