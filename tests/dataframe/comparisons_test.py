from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", {"a": [False, True, False], "b": [False, False, False]}),
        ("__ne__", {"a": [True, False, True], "b": [True, True, True]}),
        ("__ge__", {"a": [False, True, True], "b": [True, True, True]}),
        ("__gt__", {"a": [False, False, True], "b": [True, True, True]}),
        ("__le__", {"a": [True, True, False], "b": [False, False, False]}),
        ("__lt__", {"a": [True, False, False], "b": [False, False, False]}),
        ("__add__", {"a": [3, 4, 5], "b": [6, 7, 8]}),
        ("__sub__", {"a": [-1, 0, 1], "b": [2, 3, 4]}),
        ("__mul__", {"a": [2, 4, 6], "b": [8, 10, 12]}),
        ("__truediv__", {"a": [0.5, 1, 1.5], "b": [2, 2.5, 3]}),
        ("__floordiv__", {"a": [0, 1, 1], "b": [2, 2, 3]}),
        ("__pow__", {"a": [1, 4, 9], "b": [16, 25, 36]}),
        ("__mod__", {"a": [1, 0, 1], "b": [0, 1, 0]}),
    ],
)
def test_comparisons_with_scalar(
    library: str,
    comparison: str,
    expected_data: dict[str, object],
) -> None:
    df = integer_dataframe_1(library)
    other = 2
    result = getattr(df, comparison)(other)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__radd__", {"a": [3, 4, 5], "b": [6, 7, 8]}),
        ("__rsub__", {"a": [1, 0, -1], "b": [-2, -3, -4]}),
        ("__rmul__", {"a": [2, 4, 6], "b": [8, 10, 12]}),
    ],
)
def test_rcomparisons_with_scalar(
    library: str,
    comparison: str,
    expected_data: dict[str, object],
) -> None:
    df = integer_dataframe_1(library)
    other = 2
    result = getattr(df, comparison)(other)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)
