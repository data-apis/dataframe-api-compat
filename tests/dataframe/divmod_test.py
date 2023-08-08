from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import interchange_to_pandas


def test_divmod(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_2(library)
    if library == "polars-lazy":
        with pytest.raises(NotImplementedError):
            result_quotient, result_remainder = df.__divmod__(other)
        return
    else:
        result_quotient, result_remainder = df.__divmod__(other)
    result_quotient_pd = pd.api.interchange.from_dataframe(result_quotient.dataframe)
    result_remainder_pd = pd.api.interchange.from_dataframe(result_remainder.dataframe)
    expected_quotient = pd.DataFrame({"a": [1, 1, 0], "b": [1, 2, 1]})
    expected_remainder = pd.DataFrame({"a": [0, 0, 3], "b": [0, 1, 0]})
    result_quotient_pd = convert_dataframe_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_dataframe_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_frame_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_frame_equal(result_remainder_pd, expected_remainder)


def test_divmod_with_scalar(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    other = 2
    result_quotient, result_remainder = df.__divmod__(other)
    result_quotient_pd = interchange_to_pandas(result_quotient, library)
    result_remainder_pd = interchange_to_pandas(result_remainder, library)
    expected_quotient = pd.DataFrame({"a": [0, 1, 1], "b": [2, 2, 3]})
    expected_remainder = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 0]})
    result_quotient_pd = convert_dataframe_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_dataframe_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_frame_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_frame_equal(result_remainder_pd, expected_remainder)
