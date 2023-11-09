from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_rename_columns(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.rename_columns({"a": "c", "b": "e"})
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"c": [1, 2, 3], "e": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_rename_columns_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(
        TypeError,
        match="Expected Mapping, got: <class 'function'>",
    ):  # pragma: no cover
        # why is this not covered? bug in coverage?
        df.rename_columns(lambda x: x.upper())  # type: ignore  # noqa: PGH003
