import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_columns_iter(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.assign(
        *[col / col.mean() for col in df.columns_iter()],  # type: ignore[attr-defined]
    )
    expected = pd.DataFrame(
        {
            "a": [0.5, 1.0, 1.5],
            "b": [0.8, 1.0, 1.2],
        },
    )
    result_pd = interchange_to_pandas(result)
    pd.testing.assert_frame_equal(result_pd, expected)
