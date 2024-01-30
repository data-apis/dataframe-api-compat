import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import temporal_dataframe_1


def test_filter_w_date(library: str) -> None:
    df = temporal_dataframe_1(library).select("a", "index")
    pdx = df.__dataframe_namespace__()
    result = df.filter(pdx.col("a") > pdx.date(2020, 1, 2)).select("index")
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"index": [1, 2]})
    pd.testing.assert_frame_equal(result_pd, expected)
