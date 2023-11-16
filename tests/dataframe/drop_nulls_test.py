import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import null_dataframe_1


def test_drop_nulls(library: str) -> None:
    df = null_dataframe_1(library)
    result = df.drop_nulls()
    expected = pd.DataFrame({"a": [1.0, 2.0]})
    result_pd = interchange_to_pandas(result)
    pd.testing.assert_frame_equal(result_pd, expected)
