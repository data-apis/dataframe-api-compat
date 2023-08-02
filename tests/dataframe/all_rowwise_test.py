from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import bool_dataframe_1
from tests.utils import interchange_to_pandas


def test_all_rowwise(library: str, request) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (df.all_rowwise()).rename("result")}
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
