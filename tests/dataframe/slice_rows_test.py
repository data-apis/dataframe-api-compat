from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_3
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected"),
    [
        (2, 7, 2, pd.DataFrame({"a": [3, 5, 7], "b": [5, 3, 1]})),
        (None, 7, 2, pd.DataFrame({"a": [1, 3, 5, 7], "b": [7, 5, 3, 1]})),
        (2, None, 2, pd.DataFrame({"a": [3, 5, 7], "b": [5, 3, 1]})),
        (2, None, None, pd.DataFrame({"a": [3, 4, 5, 6, 7], "b": [5, 4, 3, 2, 1]})),
    ],
)
def test_slice_rows(
    library: str,
    start: int | None,
    stop: int | None,
    step: int | None,
    expected: pd.DataFrame,
) -> None:
    df = integer_dataframe_3(library)
    result = df.slice_rows(start, stop, step)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)
