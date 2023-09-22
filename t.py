from __future__ import annotations

import pandas as pd
import polars as pl


df_pandas = pd.read_parquet("iris.parquet")
df_polars = pl.scan_parquet("iris.parquet")


def my_dataframe_agnostic_function(df):
    df = df.__dataframe_consortium_standard__(api_version="2023.09-beta")

    mask = df.get_column_by_name("species") != "setosa"
    df = df.filter(mask)

    for column_name in df.get_column_names():
        if column_name == "species":
            continue
        new_column = df.get_column_by_name(column_name)
        new_column = (new_column - new_column.mean()) / new_column.std()
        df = df.insert_column(new_column.rename(f"{column_name}_scaled"))

    return df.dataframe


#  Then, either of the following will work as expected:
print(my_dataframe_agnostic_function(df_pandas))
print(my_dataframe_agnostic_function(df_polars).collect())
