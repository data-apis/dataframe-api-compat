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

@pytest.mark.parametrize(
    ('comparison', 'expected_data'),
    [
        ('__eq__', {'a': [True, True, False], 'b': [True, False, True]}),
        ('__ne__', {'a': [False, False, True], 'b': [False, True, False]}),
        ('__ge__', {'a': [True, True, False], 'b': [True, True, True]}),
        ('__gt__', {'a': [False, False, False], 'b': [False, True, False]}),
        ('__le__', {'a': [True, True, True], 'b': [True, False, True]}),
        ('__lt__', {'a': [False, False, True], 'b': [False, False, False]}),
    ]
)
def test_comparisons(comparison, expected_data):
    df = PandasDataFrame(pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
    other = PandasDataFrame(pd.DataFrame({'a': [1,2,4], 'b': [4,2,6]}))
    result = getattr(df, comparison)(other).dataframe
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected)
