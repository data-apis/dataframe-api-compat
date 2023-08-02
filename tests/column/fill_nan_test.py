from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import interchange_to_pandas
from tests.utils import nan_series_1

if TYPE_CHECKING:
    import pytest


def test_column_fill_nan(library: str, request: pytest.FixtureRequest) -> None:
    # todo: test with nullable pandas
    ser = nan_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (ser.fill_nan(-1.0)).rename("result")}
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([0.0, 1.0, -1.0], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
