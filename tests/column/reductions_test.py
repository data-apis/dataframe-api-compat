from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import integer_series_1
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
def test_column_reductions(
    library: str, reduction: str, expected: float, request: pytest.FixtureRequest
) -> None:
    df = integer_dataframe_1(library)
    ser = integer_series_1(library)
    ser = ser - getattr(ser, reduction)()
    result = df.insert(0, "result", ser)
    result_pd = interchange_to_pandas(result, library)["result"]
    ser_pd = interchange_to_pandas(df, library)["a"].rename("result")
    expected_pd = ser_pd - expected
    pd.testing.assert_series_equal(result_pd, expected_pd)


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
def test_column_reference_reductions(
    library: str, reduction: str, expected: float, request: pytest.FixtureRequest
) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    ser = ser - getattr(ser, reduction)()
    result = df.insert(0, "result", ser)
    result_pd = interchange_to_pandas(result, library)["result"]
    ser_pd = interchange_to_pandas(df, library)["a"].rename("result")
    expected_pd = ser_pd - expected
    pd.testing.assert_series_equal(result_pd, expected_pd)
