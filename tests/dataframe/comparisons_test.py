from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", {"a": [True, True, False], "b": [True, False, True]}),
        ("__ne__", {"a": [False, False, True], "b": [False, True, False]}),
        ("__ge__", {"a": [True, True, False], "b": [True, True, True]}),
        ("__gt__", {"a": [False, False, False], "b": [False, True, False]}),
        ("__le__", {"a": [True, True, True], "b": [True, False, True]}),
        ("__lt__", {"a": [False, False, True], "b": [False, False, False]}),
        ("__add__", {"a": [2, 4, 7], "b": [8, 7, 12]}),
        ("__sub__", {"a": [0, 0, -1], "b": [0, 3, 0]}),
        ("__mul__", {"a": [1, 4, 12], "b": [16, 10, 36]}),
        ("__truediv__", {"a": [1, 1, 0.75], "b": [1, 2.5, 1]}),
        ("__floordiv__", {"a": [1, 1, 0], "b": [1, 2, 1]}),
        ("__pow__", {"a": [1, 4, 81], "b": [256, 25, 46656]}),
        ("__mod__", {"a": [0, 0, 3], "b": [0, 1, 0]}),
    ],
)
def test_comparisons(
    library: str,
    comparison: str,
    expected_data: dict[str, object],
) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_2(library)
    if library == "polars-lazy":
        with pytest.raises(NotImplementedError):
            result = getattr(df, comparison)(other)
        return
    else:
        result = getattr(df, comparison)(other)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)


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
    request: pytest.FixtureRequest,
) -> None:
    df = integer_dataframe_1(library)
    other = 2
    result = getattr(df, comparison)(other)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_comparison_invalid(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library).get_columns_by_name(["a"])
    other = integer_dataframe_1(library).get_columns_by_name(["b"])
    with pytest.raises(
        (ValueError, pl.exceptions.DuplicateError, NotImplementedError),
    ):
        assert df > other
