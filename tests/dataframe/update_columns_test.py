from __future__ import annotations

from typing import Any
from typing import Callable

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_update_columns(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(integer_dataframe_1(library))
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.assign(col("a") + 1)
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize("relax", [lambda x: x, lambda x: x.collect()])
def test_update_multiple_columns(library: str, relax: Callable[[Any], Any]) -> None:
    df = relax(integer_dataframe_1(library))
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.assign(col("a") + 1, col("b") + 2)
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [6, 7, 8]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_update_broadcast(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.assign(col("a").mean(), col("b") + 2)
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [2.0, 2.0, 2.0], "b": [6, 7, 8]})
    pd.testing.assert_frame_equal(result_pd, expected)
