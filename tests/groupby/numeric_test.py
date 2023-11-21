from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_4
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("min", [1, 3], [4, 6]),
        ("max", [2, 4], [5, 7]),
        ("sum", [3, 7], [9, 13]),
        ("prod", [2, 12], [20, 42]),
        ("median", [1.5, 3.5], [4.5, 6.5]),
        ("mean", [1.5, 3.5], [4.5, 6.5]),
        (
            "std",
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, 0.7071067811865476],
        ),
        ("var", [0.5, 0.5], [0.5, 0.5]),
    ],
)
def test_group_by_numeric(
    library: str,
    aggregation: str,
    expected_b: list[float],
    expected_c: list[float],
) -> None:
    df = integer_dataframe_4(library)
    result = getattr(df.group_by("key"), aggregation)()
    result = result.sort("key")
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    if library == "pandas-nullable" and tuple(
        int(v) for v in pd.__version__.split(".")
    ) < (
        2,
        0,
        0,
    ):  # pragma: no cover
        # upstream bug
        result_pd = result_pd.astype({"key": "int64"})
    else:
        pass
    pd.testing.assert_frame_equal(result_pd, expected)
