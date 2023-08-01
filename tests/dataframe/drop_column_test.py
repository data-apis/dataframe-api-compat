import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy, integer_dataframe_1


def test_drop_column(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.drop_column("a")
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_drop_column_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError, match="Expected str, got: <class 'list'>"):
        df.drop_column(["a"])
