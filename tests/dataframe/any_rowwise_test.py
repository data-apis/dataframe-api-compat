from __future__ import annotations

import pytest

from tests.utils import BaseHandler
from tests.utils import bool_dataframe_1
from tests.utils import compare_dataframe_with_reference


def test_any_horizontal(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    mask = ns.any_horizontal(*[df.col(col_name) for col_name in df.column_names])
    result = df.filter(mask)
    expected = {"a": [True, True, False], "b": [True, True, True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_any_horizontal_invalid(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    with pytest.raises(ValueError):
        _ = namespace.any_horizontal(df.col("a"), (df + 1).col("b"))
