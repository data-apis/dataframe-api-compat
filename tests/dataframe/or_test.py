from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import interchange_to_pandas


def test_or_with_scalar(library: str) -> None:
    df = bool_dataframe_1(library)
    other = True
    result = df | other
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [True, True, True], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_ror_with_scalar(library: str) -> None:
    df = bool_dataframe_1(library)
    other = True
    result = other | df
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [True, True, True], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
