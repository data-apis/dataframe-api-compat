from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1


def test_get_columns_by_name(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.get_columns_by_name(["b"]).dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_get_columns_by_name_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError, match=r"Expected sequence of str, got <class \'str\'>"):
        df.get_columns_by_name("b")
