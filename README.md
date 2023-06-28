[![Build Status](https://github.com/MarcoGorelli/dataframe-api-compat/workflows/tox/badge.svg)](https://github.com/MarcoGorelli/dataframe-api-compat/actions?workflow=tox)
[![Coverage](https://codecov.io/gh/MarcoGorelli/cython-lint/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoGorelli/dataframe-api-compat)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MarcoGorelli/dataframe-api-compat/main.svg)](https://results.pre-commit.ci/latest/github/MarcoGorelli/dataframe-api-compat/main)

# DataFrame API Compat

<h1 align="center">
	<img
		width="400"
		alt="standard-compliant DataFrame"
		src="https://github.com/MarcoGorelli/dataframe-api-compat/assets/33491632/fb4bc907-2b85-4ad7-8d13-c2b9912b97f5">
</h1>

Work-in-progress POC of what the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
might look like for pandas and polars.

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
pip install git+https://github.com/MarcoGorelli/dataframe-api-compat
```

Testing
-------
```
pytest test_standard.py --cov=dataframe_api_compat --cov=test_standard --cov-fail-under=100
```
100% branch coverage isn't the objective - it's the bare minimum.

Linting
-------
```
pre-commit run --all-files
```

Type Checking
-------------

First, clone the [dataframe_standard](https://github.com/data-apis/dataframe-api) to some
local path. Then, run:
```console
MYPYPATH=<path to dataframe-api/spec/API_specification> mypy dataframe_api_compat
MYPYPATH=<path to dataframe-api/spec/API_specification> mypy polars_standard.py
```

For example, if you cloned both repos in the same place, this could be:
```console
MYPYPATH=../dataframe-api/spec/API_specification/ mypy dataframe_api_compat
```
