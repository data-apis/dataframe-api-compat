from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_select(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.select("b")
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_list_of_str(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.select("a", "b")
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_list_of_str_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError):
        _ = df.select(["a", "b"])  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_select_empty(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.select()
    assert result.column_names == []
