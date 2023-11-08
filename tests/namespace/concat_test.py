from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import integer_dataframe_4
from tests.utils import interchange_to_pandas


def test_concat(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    namespace = df1.__dataframe_namespace__()
    result = namespace.concat([df1, df2])
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3, 1, 2, 4], "b": [4, 5, 6, 4, 2, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_concat_mismatch(library: str) -> None:
    df1 = integer_dataframe_1(library).persist()
    df2 = integer_dataframe_4(library).persist()
    namespace = df1.__dataframe_namespace__()
    # TODO check the error
    with pytest.raises((ValueError, pl.exceptions.ShapeError)):
        _ = namespace.concat([df1, df2]).persist()
