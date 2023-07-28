from tests.utils import integer_dataframe_4, convert_dataframe_to_pandas_numpy
import pandas as pd


def test_groupby_size(library: str) -> None:
    df = integer_dataframe_4(library)
    result = df.groupby(["key"]).size()
    # got to sort
    idx = result.sorted_indices(["key"])
    result = result.get_rows(idx)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"key": [1, 2], "size": [2, 2]})
    # TODO polars returns uint32. what do we standardise to?
    result_pd["size"] = result_pd["size"].astype("int64")
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)
