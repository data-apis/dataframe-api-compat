"""PYTHONPATH=../dataframe-api/spec/API_specification/ python check_completeness.py."""
from __future__ import annotations

import dataframe_api
import pandas as pd
import polars as pl

from dataframe_api_compat import pandas_standard
from dataframe_api_compat import polars_standard

# dataframe
spec = dataframe_api.DataFrame().__dir__()
pandas_spec = pandas_standard.PandasDataFrame(
    pd.DataFrame(), api_version="2023.08-beta"
).__dir__()
polars_spec = polars_standard.PolarsDataFrame(
    pl.DataFrame(), api_version="2023.08-beta"
).__dir__()

for i in spec:
    if i not in pandas_spec:
        print(f"DataFrame.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"DataFrame.{i} missing from polars spec!")

# series
spec = dataframe_api.Column().__dir__()
pandas_spec = pandas_standard.PandasColumn(
    pd.Series(), api_version="2023.08-beta"
).__dir__()
polars_spec = polars_standard.PolarsColumn(pl.Series(), dtype=pl.Float32(), id_=None, api_version="2023.09-beta").__dir__()  # type: ignore[arg-type]

for i in spec:
    if i not in pandas_spec:
        print(f"Series.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"Series.{i} missing from polars spec!")

# groupby
spec = dataframe_api.GroupBy().__dir__()
pandas_spec = (
    pandas_standard.PandasDataFrame(pd.DataFrame({"a": [1]}), api_version="2023.08-beta")
    .groupby(["a"])
    .__dir__()
)
polars_spec = (
    polars_standard.PolarsDataFrame(pl.DataFrame({"a": [1]}), api_version="2023.08-beta")
    .groupby(["a"])
    .__dir__()
)

for i in spec:
    if i not in pandas_spec:
        print(f"GroupBy.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"GroupBy.{i} missing from polars spec!")

# namespace
exclude = {
    "Mapping",
    "Sequence",
    "column_object",
    "dataframe_object",
    "groupby_object",
    "DType",
}
spec = [i for i in dataframe_api.__dir__() if i not in exclude and not i.startswith("_")]
pandas_spec = pandas_standard.__dir__()
polars_spec = polars_standard.__dir__()

for i in spec:
    if i not in pandas_spec:
        print(f"namespace.{i} missing from pandas spec!")
    if i not in polars_spec:
        print(f"namespace.{i} missing from polars spec!")
