from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import pandas_version
from tests.utils import polars_version


def test_scale_column(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable"):
        if pandas_version() < Version("2.1.0"):  # pragma: no cover
            pytest.skip(reason="pandas doesn't support 3.8")
        import pandas as pd

        s = pd.Series([1, 2, 3], name="a")
        ser = s.__column_consortium_standard__()
    elif library.name == "polars-lazy":
        if polars_version() < Version("0.19.0"):  # pragma: no cover
            pytest.skip(reason="before consortium standard in polars")
        import polars as pl

        s = pl.Series("a", [1, 2, 3])
        ser = s.__column_consortium_standard__()
    elif library.name == "modin":
        import modin.pandas as pd

        s = pd.Series([1, 2, 3], name="a")
        ser = s.__column_consortium_standard__()
    else:  # pragma: no cover
        msg = f"Not supported library: {library}"
        raise AssertionError(msg)

    ns = ser.__column_namespace__()
    ser = ser - ser.mean()
    compare_column_with_reference(ser, [-1, 0, 1.0], dtype=ns.Float64)


def test_scale_column_from_persisted_df(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable"):
        if pandas_version() < Version("2.1.0"):  # pragma: no cover
            pytest.skip(reason="pandas doesn't support 3.8")
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        ser = df.__dataframe_consortium_standard__().col("a")
    elif library.name == "polars-lazy":
        if polars_version() < Version("0.19.0"):  # pragma: no cover
            pytest.skip(reason="before consortium standard in polars")
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3]})
        ser = df.__dataframe_consortium_standard__().col("a")
    elif library.name == "modin":
        import modin.pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        ser = df.__dataframe_consortium_standard__().col("a")
    else:  # pragma: no cover
        msg = f"Not supported library: {library}"
        raise AssertionError(msg)

    ns = ser.__column_namespace__()
    ser = ser - ser.mean()
    compare_column_with_reference(ser, [-1, 0, 1.0], dtype=ns.Float64)
