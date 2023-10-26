from __future__ import annotations

from typing import Any
from typing import Callable

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_divmod_with_scalar(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(integer_dataframe_1(library))
    other = 2
    result_quotient, result_remainder = df.__divmod__(other)
    result_quotient_pd = interchange_to_pandas(result_quotient)
    result_remainder_pd = interchange_to_pandas(result_remainder)
    expected_quotient = pd.DataFrame({"a": [0, 1, 1], "b": [2, 2, 3]})
    expected_remainder = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 0]})
    result_quotient_pd = convert_dataframe_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_dataframe_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_frame_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_frame_equal(result_remainder_pd, expected_remainder)
