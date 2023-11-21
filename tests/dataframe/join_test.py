from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import PANDAS_VERSION
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import interchange_to_pandas


def test_join_left(library: str) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename_columns({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="left")
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, 2.0, float("nan")]},
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_join_overlapping_names(library: str) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library)
    with pytest.raises(ValueError):
        _ = left.join(right, left_on="a", right_on="a", how="left")


def test_join_inner(library: str) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename_columns({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="inner")
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [1, 2], "b": [4, 5], "c": [4, 2]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_join_outer(library: str) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename_columns({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="outer").sort("a")
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [4, 5, 6, float("nan")],
            "c": [4.0, 2.0, float("nan"), 6.0],
        },
    )
    if library == "pandas-nullable" and PANDAS_VERSION < (2, 0, 0):  # pragma: no cover
        # upstream bug
        result_pd = result_pd.astype({"a": "int64"})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_join_two_keys(library: str) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename_columns({"b": "c"})
    result = left.join(right, left_on=["a", "b"], right_on=["a", "c"], how="left")
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, float("nan"), float("nan")]},
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_join_invalid(library: str) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename_columns({"b": "c"})
    with pytest.raises(ValueError):
        left.join(right, left_on=["a", "b"], right_on=["a", "c"], how="right")  # type: ignore  # noqa: PGH003
