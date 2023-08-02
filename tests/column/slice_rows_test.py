from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_dataframe_3


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected"),
    [
        (2, 7, 2, pd.Series([3, 5, 7], name="result")),
        (None, 7, 2, pd.Series([1, 3, 5, 7], name="result")),
        (2, None, 2, pd.Series([3, 5, 7], name="result")),
        (2, None, None, pd.Series([3, 4, 5, 6, 7], name="result")),
    ],
)
def test_column_slice_rows(
    library: str,
    start: int | None,
    stop: int | None,
    step: int | None,
    expected: pd.Series[Any],
    request: pytest.FixtureRequest,
) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    ser = integer_dataframe_3(library).get_column_by_name("a")
    namespace = ser.__column_namespace__()
    result = ser.slice_rows(start, stop, step)
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": (result).rename("result")}).dataframe
    )["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)
