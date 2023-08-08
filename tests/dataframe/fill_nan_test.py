from __future__ import annotations

import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_1


def test_fill_nan(library: str) -> None:
    df = nan_dataframe_1(library)
    result = df.fill_nan(-1)
    result_pd = interchange_to_pandas(result, library)
    result_pd = result_pd.astype("float64")
    expected = pd.DataFrame({"a": [1.0, 2.0, -1.0]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_fill_nan_with_null(library: str) -> None:
    df = nan_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.fill_nan(namespace.null)
    n_nans = result.is_nan().sum()
    n_nans = interchange_to_pandas(n_nans, library)
    if library == "pandas-numpy":
        # null is nan for pandas-numpy
        assert n_nans["a"][0] == 1
    else:
        assert n_nans["a"][0] == 0
