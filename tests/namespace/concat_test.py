import pandas as pd
import pytest
import polars as pl
from tests.utils import (
    integer_dataframe_1,
    integer_dataframe_2,
    integer_dataframe_4,
    convert_dataframe_to_pandas_numpy,
)


def test_concat(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    namespace = df1.__dataframe_namespace__()
    result = namespace.concat([df1, df2])
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3, 1, 2, 4], "b": [4, 5, 6, 4, 2, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_concat_mismatch(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_4(library)
    namespace = df1.__dataframe_namespace__()
    # todo check the error
    with pytest.raises((ValueError, pl.exceptions.ShapeError)):
        namespace.concat([df1, df2]).dataframe
