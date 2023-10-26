from __future__ import annotations

from typing import Any
from typing import Callable

import pandas as pd
import pytest

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_2
from tests.utils import null_dataframe_1


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_is_null_1(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(nan_dataframe_2(library))
    result = df.is_null()
    result_pd = interchange_to_pandas(result)
    if library == "pandas-numpy":
        # nan and null are the same in pandas-numpy
        expected = pd.DataFrame({"a": [False, False, True]})
    else:
        expected = pd.DataFrame({"a": [False, False, False]})
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_is_null_2(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(null_dataframe_1(library))
    result = df.is_null()
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
