from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_5
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("ascending", "expected_data"),
    [
        (True, [1, 0]),
        (False, [0, 1]),
    ],
)
def test_sorted_indices(
    library: str,
    ascending: bool,
    expected_data: list[int],
    request: pytest.FixtureRequest,
) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_5(library)
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict(
        {
            "result": (df.sorted_indices(keys=["a", "b"], ascending=ascending)).rename(
                "result"
            )
        }
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    # TODO should we standardise on the return type?
    result_pd = result_pd.astype("int64")
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)
