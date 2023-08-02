from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_5


@pytest.mark.parametrize(
    ("ascending", "expected_data"),
    [
        (True, [1, 0]),
        (False, [0, 1]),
    ],
)
def test_sorted_indices(library: str, ascending: bool, expected_data: list[int]) -> None:
    df = integer_dataframe_5(library)
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict(
        {
            "result": (df.sorted_indices(keys=["a", "b"], ascending=ascending)).rename(
                "result"
            )
        }
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    # TODO should we standardise on the return type?
    result_pd = result_pd.astype("int64")
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)
