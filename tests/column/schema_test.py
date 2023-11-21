from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import mixed_dataframe_1


@pytest.mark.skipif(
    tuple(int(v) for v in pd.__version__.split(".")) < (2, 0, 0),
    reason="no pyarrow support",
)
def test_schema(library: str) -> None:
    df = mixed_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.col("a").dtype
    assert isinstance(result, namespace.Int64)
