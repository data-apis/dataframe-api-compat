import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_shift(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.assign(df.col("a").shift(1, fill_value=999))  # type: ignore[attr-defined]
    expected = pd.DataFrame(
        {
            "a": [999, 1, 2],
            "b": [4, 5, 6],
        },
    )
    result_pd = interchange_to_pandas(result)
    pd.testing.assert_frame_equal(result_pd, expected)
