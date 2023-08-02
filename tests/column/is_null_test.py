from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import nan_series_1
from tests.utils import null_series_1

if TYPE_CHECKING:
    import pytest


def test_column_is_null_1(library: str, request: pytest.FixtureRequest) -> None:
    ser = nan_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser.is_null()).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([False, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_is_null_2(library: str, request: pytest.FixtureRequest) -> None:
    ser = null_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (ser.is_null()).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
