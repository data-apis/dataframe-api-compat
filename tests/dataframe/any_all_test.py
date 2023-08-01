from __future__ import annotations
import pytest
import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import bool_dataframe_3


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(
    library: str,
    reduction: str,
    expected_data: dict[str, object],
) -> None:
    df = bool_dataframe_1(library)
    result = getattr(df, reduction)()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_any(library: str) -> None:
    df = bool_dataframe_3(library)
    result = df.any()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"a": [False], "b": [True], "c": [True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_all(library: str) -> None:
    df = bool_dataframe_3(library)
    result = df.all()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"a": [False], "b": [False], "c": [True]})
    pd.testing.assert_frame_equal(result_pd, expected)
