Work-in-progress draft of what the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
might look like for pandas and polars.

Example usage:
```python
from pandas_standard import PandasDataFrame

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

df_standard = PandasDataFrame(df)  # standard-compliant DataFrame
```

Installation:
```
pip install git+https://github.com/MarcoGorelli/impl-dataframe-api
```
