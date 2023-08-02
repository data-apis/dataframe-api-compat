from __future__ import annotations

import numpy as np
import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1


def test_dataframe_from_2d_array(library) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    arr = np.array([[1, 4], [2, 5], [3, 6]])
    result = namespace.dataframe_from_2d_array(
        arr, names=["a", "b"], dtypes={"a": namespace.Int64(), "b": namespace.Int64()}
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)
