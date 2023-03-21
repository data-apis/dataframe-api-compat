import pytest
import pandas as pd
from pandas_standard import PandasDataFrame

@pytest.mark.parametrize(
    ('reduction', 'expected_data'),
    [
        ('any', {'a': [True], 'b': [True]}),
        ('all', {'a': [False], 'b': [True]}),
    ]
)
def test_reductions(reduction, expected_data):
    df = pd.DataFrame({'a': [True, True, False], 'b': [True, True, True]})
    dfstd = PandasDataFrame(df)
    result = getattr(dfstd, reduction)().dataframe
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected)
