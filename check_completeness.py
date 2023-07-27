"""
PYTHONPATH=../dataframe-api/spec/API_specification/ python check_completeness.py
"""

import dataframe_api
from dataframe_api_compat import pandas_standard, polars_standard
import pandas as pd
import polars as pl

# dataframe
spec = dataframe_api.DataFrame().__dir__()
pandas_spec = pandas_standard.PandasDataFrame(pd.DataFrame()).__dir__()
polars_spec = polars_standard.PolarsDataFrame(pl.DataFrame()).__dir__()

for i in spec:
    if i not in pandas_spec:
        print(f"DataFrame.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"DataFrame.{i} missing from pandas spec!")

# series
spec = dataframe_api.Column().__dir__()
pandas_spec = pandas_standard.PandasColumn(pd.Series()).__dir__()
polars_spec = polars_standard.PolarsColumn(pl.Series()).__dir__()

for i in spec:
    if i not in pandas_spec:
        print(f"Series.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"Series.{i} missing from pandas spec!")

# namespace
exclude = {"Mapping", "column_object", "dataframe_object", "groupby_object", "DType"}
spec = [i for i in dataframe_api.__dir__() if i not in exclude and not i.startswith("_")]
pandas_spec = pandas_standard.__dir__()
polars_spec = polars_standard.__dir__()

for i in spec:
    if i not in pandas_spec:
        print(f"Series.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"Series.{i} missing from pandas spec!")
