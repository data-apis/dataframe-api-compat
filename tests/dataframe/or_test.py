from __future__ import annotations

from typing import Any
from typing import Callable

import pandas as pd
import pytest

from tests.utils import bool_dataframe_1
from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_or_with_scalar(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(bool_dataframe_1(library))
    other = True
    result = df | other
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [True, True, True], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
