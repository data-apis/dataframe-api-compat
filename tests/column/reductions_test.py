from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("min", 1),
        ("max", 3),
        ("sum", 6),
        ("prod", 6),
        ("median", 2.0),
        ("mean", 2.0),
        ("std", 1.0),
        ("var", 1.0),
    ],
)
def test_expression_reductions(
    library: str,
    reduction: str,
    expected: float,
) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    ser = df.col("a")
    ser = ser - getattr(ser, reduction)()
    result = df.assign(ser.rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    ser_pd = interchange_to_pandas(df)["a"].rename("result")
    expected_pd = ser_pd - expected
    pd.testing.assert_series_equal(result_pd, expected_pd)
