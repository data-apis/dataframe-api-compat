from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

import dataframe_api_compat


def convert_to_standard_compliant_dataframe(df: pd.DataFrame | pl.DataFrame) -> Any:
    # todo: type return
    if isinstance(df, pd.DataFrame):
        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_dataframe(
                df
            )
        )
    elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
                df
            )
        )
    else:
        raise AssertionError(f"Got unexpected type: {type(df)}")


dfpd = pd.DataFrame({"a": [1] * 10_000 + [9999]})
dfpl = pl.DataFrame({"a": [1] * 10_000 + [9999]})
dfplazy = pl.LazyFrame({"a": [1] * 10_000 + [9999]})


dfpd = convert_to_standard_compliant_dataframe(dfpd)
dfpl = convert_to_standard_compliant_dataframe(dfpl)
dfplazy = convert_to_standard_compliant_dataframe(dfplazy)


def remove_outliers(df_standard, column):
    # Get a Standard-compliant DataFrame.
    # NOTE: this has not yet been upstreamed, so won't work out-of-the-box!
    # See 'resources' below for how to try it out.
    # Use methods from the Standard specification.
    col = df_standard.get_column_by_name(column)
    z_score = (col - col.mean()) / col.std()
    df_standard_filtered = df_standard.get_rows_by_mask((z_score > -3) & (z_score < 3))
    # Return the result as a DataFrame from the original library.
    return df_standard_filtered.dataframe


print(remove_outliers(dfpd, "a"))
print(remove_outliers(dfpl, "a"))
print(remove_outliers(dfplazy, "a").collect())
