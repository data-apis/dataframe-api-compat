from __future__ import annotations

import polars as pl

df_pandas = pl.read_parquet("iris.parquet")
df_polars = pl.scan_parquet("iris.parquet")


def my_dataframe_agnostic_function(df):
    df = df.__dataframe_consortium_standard__(api_version="2023.08-beta")

    mask = df.get_column_by_name("species") != "setosa"
    df = df.get_rows_by_mask(mask)

    for column_name in df.get_column_names():
        if column_name == "species":
            continue
        new_column = df.get_column_by_name(column_name)
        new_column = (new_column - new_column.mean()) / new_column.std()
        df = df.insert(new_column.rename(f"{column_name}_scaled"))

    return df.dataframe


#  Then, either of the following will work as expected:
my_dataframe_agnostic_function(df_pandas)
my_dataframe_agnostic_function(df_polars)
