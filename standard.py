from __future__ import annotations

from typing import Any
import pandas as pd
import polars as pl
from pandas_standard import PandasDataFrame
from polars_standard import PolarsDataFrame

from dataframe_api import DataFrame


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame | pl.DataFrame,
) -> DataFrame[Any]:
    if isinstance(df, pd.DataFrame):
        return PandasDataFrame(df)
    elif isinstance(df, pl.DataFrame):
        return PolarsDataFrame(df)
