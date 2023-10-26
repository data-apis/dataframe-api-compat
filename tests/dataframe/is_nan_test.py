from __future__ import annotations

from typing import Any
from typing import Callable

import pandas as pd
import pytest

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_1


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_dataframe_is_nan(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(nan_dataframe_1(library))
    result = df.is_nan()
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
