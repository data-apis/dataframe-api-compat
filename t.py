import pandas as pd
import polars as pl
from typing import Any
import dataframe_api_compat


def convert_to_standard_compliant_dataframe(df: pd.DataFrame | pl.DataFrame) -> Any:
    # todo: type return
    if isinstance(df, pd.DataFrame):
        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_dataframe(
                df
            )
        )
    elif isinstance(df, pl.DataFrame):
        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
                df
            )
        )
    else:
        raise AssertionError(f"Got unexpected type: {type(df)}")


dfpd = pd.DataFrame({"a": [1, 2, 3]})
dfpl = pl.DataFrame({"a": [1, 2, 3]})

dfpd = convert_to_standard_compliant_dataframe(dfpd)
dfpl = convert_to_standard_compliant_dataframe(dfpl)
