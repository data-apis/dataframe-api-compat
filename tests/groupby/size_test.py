from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_4
from tests.utils import interchange_to_pandas


def test_group_by_size(library: str) -> None:
    df = integer_dataframe_4(library)
    result = df.group_by("key").size()
    result = result.sort("key")
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"key": [1, 2], "size": [2, 2]})
    # TODO polars returns uint32. what do we standardise to?
    result_pd["size"] = result_pd["size"].astype("int64")
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)
