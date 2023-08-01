import pandas as pd

from tests.utils import bool_dataframe_1, convert_dataframe_to_pandas_numpy


def test_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    result = ~df
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [False, False, True], "b": [False, False, False]})
    pd.testing.assert_frame_equal(result_pd, expected)
