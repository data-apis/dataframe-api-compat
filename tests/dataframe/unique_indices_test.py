from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_6
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("keys", "expected_data"),
    [
        (["a", "b"], {"a": [1, 1, 2, 2], "b": [3, 4, 1, 2]}),
        (None, {"a": [1, 1, 2, 2], "b": [3, 4, 1, 2]}),
        (["a"], {"a": [1, 2], "b": [4, 1]}),
        (["b"], {"a": [2, 2, 1, 1], "b": [1, 2, 3, 4]}),
    ],
)
def test_unique_indices(
    library: str,
    keys: list[str] | None,
    expected_data: dict[str, list[int]],
    request: pytest.FixtureRequest,
) -> None:
    df = integer_dataframe_6(library)
    if library == "polars-lazy":
        # not yet implemented, need to figure this out
        request.node.add_marker(pytest.mark.xfail())
    df = df.get_rows(df.unique_indices(keys))
    result = df.get_rows(df.sorted_indices(keys))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)
