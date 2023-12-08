import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_cast_integers(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.cast({"a": pdx.Int32()})
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).astype(
        {"a": "int32", "b": "int64"},
    )
    result_pd = interchange_to_pandas(result)
    pd.testing.assert_frame_equal(result_pd, expected)
