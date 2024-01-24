from __future__ import annotations

import sys
from typing import Any
import pytest

LIBRARIES = {
    (3, 8): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 9): ["pandas-numpy", "pandas-nullable", "polars-lazy", "modin"],
    (3, 10): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 11): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 12): ["polars-lazy"],
}


class PandasHandler:
    def __init__(self, name: str):
        assert name in ("pandas-numpy", "pandas-nullable")
        self.name = name

    def __eq__(self, other: str):
        return self.name == other

    def __str__(self) -> str:
        return self.name

    def namespace(self, api_version):
        from dataframe_api_compat.pandas_standard import Namespace

        return Namespace(api_version=api_version)

    def dataframe(self, data, api_version=None, **kwargs):
        import pandas as pd
        import dataframe_api_compat.pandas_standard

        if self.name == "pandas-nullable" and "dtype" in kwargs:
            if kwargs["dtype"] == "bool":
                kwargs["dtype"] = "boolean"
            elif kwargs["dtype"] == "int64":
                kwargs["dtype"] = "Int64"
            elif kwargs["dtype"] == "float64":
                kwargs["dtype"] = "Float64"
        df = pd.DataFrame(data, **kwargs)

        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version or "2023.11-beta",
            )
        )


class PolarsHandler:
    def __init__(self, name: str):
        assert name == "polars-lazy"
        self.name = name

    def __eq__(self, other: str):
        return self.name == other

    def __str__(self) -> str:
        return self.name

    def dataframe(self, data, api_version=None, **kwargs):
        # TODO: should we ignore kwargs? For example, dtype
        import polars as pl
        import dataframe_api_compat.polars_standard

        df = pl.DataFrame(data)

        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version or "2023.11-beta",
            )
        )


class ModinHandler:
    def __init__(self, name: str):
        assert name == "modin"
        self.name = name

    def __eq__(self, other: str):
        return self.name == other

    def __str__(self) -> str:
        return self.name

    def dataframe(self, data, api_version=None, **kwargs):
        import modin.pandas as pd
        import dataframe_api_compat.modin_standard

        cast_dtypes = None
        if "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
            cast_dtypes = kwargs.pop("dtype")

        df = pd.DataFrame(data, **kwargs)

        if cast_dtypes:
            df = df.astype(cast_dtypes)

        return (
            dataframe_api_compat.modin_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version or "2023.11-beta",
            )
        )


@pytest.fixture(scope="session", params=LIBRARIES[sys.version_info[:2]])
def library(request):
    library_name = request.param
    if library_name in ("pandas-numpy", "pandas-nullable"):
        yield PandasHandler(library_name)
    elif library_name == "polars-lazy":
        yield PolarsHandler(library_name)
    elif library_name == "modin":
        yield ModinHandler(library_name)
    else:
        raise NotImplementedError
