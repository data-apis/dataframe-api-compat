from __future__ import annotations

import pandas as pd
import polars as pl

df_pandas = pd.read_parquet("iris.parquet")
df_polars = pl.scan_parquet("iris.parquet")


def my_dataframe_agnostic_function(df):
    df = df.__dataframe_consortium_standard__(api_version="2023.09-beta")
    namespace = df.__dataframe_namespace__()
    col = namespace.col

    mask = col("species") != "setosa"
    df = df.filter(mask)

    updated_columns = []
    for column_name in df.column_names:
        if column_name == "species":
            continue
        new_column = col(column_name)
        new_column = (new_column - new_column.mean()) / new_column.std()
        updated_columns.append(new_column)

    df = df.update_columns(*updated_columns)

    return df.dataframe


print(my_dataframe_agnostic_function(df_pandas))
print(my_dataframe_agnostic_function(df_polars).collect())
