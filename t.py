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
    df = df.get_rows_by_mask(mask)

    updated_columns = []
    for column_name in df.get_column_names():
        if column_name == "species":
            continue
        new_column = col(column_name)
        new_column = (new_column - new_column.mean()) / new_column.std()
        updated_columns.append(new_column)

    df = df.update_columns(updated_columns)

    return df.dataframe


dfpd = pd.DataFrame({"a": [1] * 10_000 + [9999]})
dfpl = pl.DataFrame({"a": [1] * 10_000 + [9999]})
dfplazy = pl.LazyFrame({"a": [1] * 10_000 + [9999]})


# dfpd = convert_to_standard_compliant_dataframe(dfpd)
# dfpl = convert_to_standard_compliant_dataframe(dfpl)
# dfplazy = convert_to_standard_compliant_dataframe(dfplazy)


def remove_outliers(df_standard, column):
    # Get a Standard-compliant DataFrame.
    # NOTE: this has not yet been upstreamed, so won't work out-of-the-box!
    # See 'resources' below for how to try it out.
    # Use methods from the Standard specification.
    col = df_standard.get_column_by_name(column)
    z_score = (col - col.mean()) / col.std()
    df_standard_filtered = df_standard.filter((z_score > -3) & (z_score < 3))
    # Return the result as a DataFrame from the original library.
    return df_standard_filtered.dataframe


print(remove_outliers(dfpd, "a"))
print(remove_outliers(dfpl, "a"))
print(remove_outliers(dfplazy, "a").collect())
