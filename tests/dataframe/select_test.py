from __future__ import annotations

import pytest

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_select(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.select("b")
    expected = {"b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_select_list_of_str(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.select("a", "b")
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_select_list_of_str_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError):
        _ = df.select(["a", "b"])  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_select_empty(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    result = df.select()
    assert result.column_names == []
