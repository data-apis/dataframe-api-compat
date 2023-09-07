from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.utils import integer_dataframe_3
from tests.utils import interchange_to_pandas


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
) -> None:
    namespace = integer_dataframe_3(library).__dataframe_namespace__()
    ser = namespace.column_from_sequence(
        [1, 2, 3, 4, 5, 6, 7], name="a", dtype=namespace.Int64()
    )
    result = ser.slice_rows(start, stop, step)
    result_pd = interchange_to_pandas(
        namespace.dataframe_from_dict({"result": (result).rename("result")}), library
    )["result"]
    pd.testing.assert_series_equal(result_pd, expected)
