from tests.utils import nan_dataframe_1
import pandas as pd


def test_dataframe_is_nan(library: str) -> None:
    df = nan_dataframe_1(library)
    result = df.is_nan()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
