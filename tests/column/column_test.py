from __future__ import annotations

from typing import cast

import pandas as pd
import polars as pl

from dataframe_api_compat import pandas_standard
from tests.utils import integer_dataframe_1


def test_column_column() -> None:
    namespace = integer_dataframe_1("polars-lazy").__dataframe_namespace__()
    ser = namespace.column_from_sequence([1, 2, 3], name="a", dtype=namespace.Int64())
    result_pl = ser.column
    result_pl = cast(pl.Series, result_pl)
    pd.testing.assert_series_equal(result_pl.to_pandas(), pd.Series([1, 2, 3], name="a"))
    result_pd = (
        pandas_standard.convert_to_standard_compliant_dataframe(
            pd.DataFrame({"a": [1, 2, 3]}), "2023.08-beta"
        )
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result_pd, pd.Series([1, 2, 3], name="a"))
