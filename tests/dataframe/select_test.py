from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_select(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.select("b")
    expected = {"b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


def test_select_list_of_str(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.select("a", "b")
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)


def test_select_list_of_str_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError):
        _ = df.select(["a", "b"])  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_select_empty(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(ValueError, match="Can't select no columns"):
        df.select()
