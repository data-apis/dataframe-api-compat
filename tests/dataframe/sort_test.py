from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_5
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize("keys", [["a", "b"], []])
def test_sort(library: str, keys: list[str]) -> None:
    df = integer_dataframe_5(library, api_version="2023.09-beta")
    result = df.sort(*keys)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [1, 1], "b": [3, 4]})
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize("keys", [["a", "b"], []])
def test_sort_descending(
    library: str,
    keys: list[str],
) -> None:
    df = integer_dataframe_5(library, api_version="2023.09-beta")
    result = df.sort(*keys, ascending=False)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [1, 1], "b": [4, 3]})
    pd.testing.assert_frame_equal(result_pd, expected)
