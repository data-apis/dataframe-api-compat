from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_and_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    df = bool_dataframe_1(library)
    other = True
    result = df & other
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [True, True, False], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
