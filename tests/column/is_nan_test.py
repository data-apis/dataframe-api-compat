from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import nan_series_1

if TYPE_CHECKING:
    import pytest


def test_column_is_nan(library: str, request: pytest.FixtureRequest) -> None:
    ser = nan_series_1(library, request)
    result = ser.is_nan()
    namespace = ser.__column_namespace__()
    result_df = namespace.dataframe_from_dict({"result": (result).rename("result")})
    result_pd = pd.api.interchange.from_dataframe(result_df.dataframe)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
