import pytest
import pandas as pd
from tests.utils import integer_dataframe_4, convert_dataframe_to_pandas_numpy


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("min", [1, 3], [4, 6]),
        ("max", [2, 4], [5, 7]),
        ("sum", [3, 7], [9, 13]),
        ("prod", [2, 12], [20, 42]),
        ("median", [1.5, 3.5], [4.5, 6.5]),
        ("mean", [1.5, 3.5], [4.5, 6.5]),
        (
            "std",
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, 0.7071067811865476],
        ),
        ("var", [0.5, 0.5], [0.5, 0.5]),
    ],
)
def test_groupby_numeric(
    library: str, aggregation: str, expected_b: list[float], expected_c: list[float]
) -> None:
    df = integer_dataframe_4(library)
    result = getattr(df.groupby(["key"]), aggregation)()
    sorted_indices = result.sorted_indices(["key"])
    result = result.get_rows(sorted_indices)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result_pd, expected)
