from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import integer_series_1
from tests.utils import integer_series_5
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_mean(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.assign((namespace.col("a") - namespace.col("a").mean()).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([-1, 0, 1.0], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_std(library: str, request: pytest.FixtureRequest) -> None:
    result = integer_series_5(library, request).std()
    assert abs(result - 1.7320508075688772) < 1e-8


def test_column_max(library: str) -> None:
    result = integer_series_1(library).max()
    assert result == 3
