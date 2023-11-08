from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import bool_dataframe_1
from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    result = ~df
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [False, False, True], "b": [False, False, False]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_invert_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError):
        _ = ~df
