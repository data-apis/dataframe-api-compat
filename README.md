[![Build Status](https://github.com/data-apis/dataframe-api-compat/workflows/tox/badge.svg)](https://github.com/data-apis/dataframe-api-compat/actions?workflow=tox)
[![Coverage](https://codecov.io/gh/MarcoGorelli/cython-lint/branch/main/graph/badge.svg)](https://codecov.io/gh/data-apis/dataframe-api-compat)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MarcoGorelli/dataframe-api-compat/main.svg)](https://results.pre-commit.ci/latest/github/MarcoGorelli/dataframe-api-compat/main)

# DataFrame API Compat

<h1 align="center">
	<img
		width="400"
		alt="standard-compliant DataFrame"
		src="https://github.com/data-apis/dataframe-api-compat/assets/33491632/2997cb92-fd10-4426-bd41-8dfd1e466ee2">
</h1>

Implementation of the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
for pandas and polars.

What's this?
------------
Please read our blog post! https://data-apis.org/blog/dataframe_standard_rfc/.

How to try this out
-------------------

Here's an example of how you can try this out:
```python
import pandas as pd
from dataframe_api_compat import pandas_standard
from dataframe_api_compat import polars_standard

def convert_to_standard_compliant_dataframe(df):
    if isinstance(df, pd.DataFrame):
        return pandas_standard.convert_to_standard_compliant_dataframe(df)
    elif isinstance(df, pl.DataFrame):
        return polars_standard.convert_to_standard_compliant_dataframe(df)
    else:
        raise TypeError(f"Got unexpected type: {type(df)}")

df = pd.DataFrame({'a': [1,2,3]})
df_std = convert_to_standard_compliant_dataframe(df)
```
The object `df_std` is a Standard-compliant DataFrame.

Installation
------------
```
pip install dataframe-api-compat
```
