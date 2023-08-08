from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_1

if TYPE_CHECKING:
    import pytest


def test_dataframe_is_nan(library: str, request: pytest.FixtureRequest) -> None:
    df = nan_dataframe_1(library)
    result = df.is_nan()
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
