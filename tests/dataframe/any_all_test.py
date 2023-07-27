import pytest
import pandas as pd

from tests.utils import bool_dataframe_1


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(
    library: str,
    reduction: str,
    expected_data: dict[str, object],
) -> None:
    df = bool_dataframe_1(library)
    result = getattr(df, reduction)()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)
