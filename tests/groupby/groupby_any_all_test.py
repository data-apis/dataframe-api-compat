from __future__ import annotations
import pandas as pd
from tests.utils import integer_dataframe_4
from tests.utils import bool_dataframe_2
from tests.utils import convert_dataframe_to_pandas_numpy
import pytest


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_boolean(
    library: str, aggregation: str, expected_b: list[bool], expected_c: list[bool]
) -> None:
    df = bool_dataframe_2(library)
    result = getattr(df.groupby(["key"]), aggregation)()
    # need to sort
    idx = result.sorted_indices(["key"])
    result = result.get_rows(idx)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_groupby_invalid_any_all(library: str) -> None:
    df = integer_dataframe_4(library)
    with pytest.raises(Exception):
        df.groupby(["key"]).any()
    with pytest.raises(Exception):
        df.groupby(["key"]).all()
