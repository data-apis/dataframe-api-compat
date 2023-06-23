from __future__ import annotations
import pandas as pd
import polars as pl
from pandas_standard import PandasDataFrame
from polars_standard import PolarsDataFrame


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame | pl.DataFrame,
) -> PandasDataFrame | PolarsDataFrame:
    if isinstance(df, pd.DataFrame):
        return PandasDataFrame(df)
    elif isinstance(df, pl.DataFrame):
        return PolarsDataFrame(df)
