from __future__ import annotations

import sys
from typing import Any


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


LIBRARIES = {
    (3, 8): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 9): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 10): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 11): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 12): ["polars-lazy"],
}

LIBRARIES_HANDLERS = {
    "pandas-numpy": PandasHandler("pandas-numpy"),
    "pandas-nullable": PandasHandler("pandas-nullable"),
    "polars-lazy": PolarsHandler("polars-lazy"),
    "modin": ModinHandler("modin"),
}


def pytest_addoption(parser):
    parser.addoption(
        "--library",
        action="store",
        default=None,
        type=str,
        help="library to test",
    )


def pytest_configure(config):
    library = config.option.library
    if library is None:
        # `LIBRARIES` is already initialized
        return
    else:
        assert library in ("pandas-numpy", "pandas-nullable", "polars-lazy", "modin")
        global LIBRARIES
        LIBRARIES = {
            (3, 8): [library],
            (3, 9): [library],
            (3, 10): [library],
            (3, 11): [library],
            (3, 12): [library],
        }

    # TODO: potential way to add filterwarnings for different librariess
    # config.addinivalue_line(
    #    "filterwarnings", "ignore:Ray execution environment not yet initialized:UserWarning",
    # )


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        libraries = LIBRARIES[sys.version_info[:2]]
        lib_handlers = [LIBRARIES_HANDLERS[lib] for lib in libraries]

        metafunc.parametrize("library", lib_handlers, ids=libraries)
