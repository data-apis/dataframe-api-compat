from __future__ import annotations

import pytest

from tests.utils import PANDAS_VERSION
from tests.utils import mixed_dataframe_1


@pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0),
    reason="no pyarrow support",
)
def test_schema(library: str) -> None:
    df = mixed_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.col("a").dtype
    assert isinstance(result, namespace.Int64)
