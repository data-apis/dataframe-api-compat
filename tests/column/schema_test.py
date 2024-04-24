from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import mixed_dataframe_1
from tests.utils import pandas_version


def test_schema(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable") and pandas_version() < Version(
        "2.0.0",
    ):  # pragma: no cover
        pytest.skip(reason="no pyarrow support")
    df = mixed_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.col("a").dtype
    assert isinstance(result, namespace.Int64)
