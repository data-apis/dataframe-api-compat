from __future__ import annotations

from typing import cast

import pandas as pd
import polars as pl

from dataframe_api_compat import pandas_standard
from dataframe_api_compat import polars_standard


def test_column_column() -> None:
    result_pl = (
        polars_standard.convert_to_standard_compliant_dataframe(
            pl.DataFrame({"a": [1, 2, 3]}), "2023.08-beta"
        )
        .get_column_by_name("a")
        .column
    )
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
