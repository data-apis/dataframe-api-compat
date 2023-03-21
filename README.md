Work-in-progress draft of what the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
might look like for pandas.

Usage:
```python
from pandas_standard import PandasDataFrame

df: pd.DataFrame  # regular pandas DataFrame

df_standard = PandasDataFrame(df)  # standard-compliant DataFrame
```