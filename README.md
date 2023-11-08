[![Build Status](https://github.com/data-apis/dataframe-api-compat/workflows/tox/badge.svg)](https://github.com/data-apis/dataframe-api-compat/actions?workflow=tox)
[![Coverage](https://codecov.io/gh/MarcoGorelli/cython-lint/branch/main/graph/badge.svg)](https://codecov.io/gh/data-apis/dataframe-api-compat)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MarcoGorelli/dataframe-api-compat/main.svg)](https://results.pre-commit.ci/latest/github/MarcoGorelli/dataframe-api-compat/main)

# DataFrame API Compat

<h1 align="center">
	<img
		width="400"
		alt="standard-compliant DataFrame"
		src="https://github.com/data-apis/dataframe-api-compat/assets/33491632/9616bbdc-97d7-4936-8530-67280ba472d0">
</h1>

Implementation of the [DataFrame Standard](https://data-apis.org/dataframe-api/draft/index.html)
for pandas and polars.

What's this?
------------
Please read our blog post! https://data-apis.org/blog/dataframe_standard_rfc/.

Documentation
-------------
Please check https://data-apis.org/dataframe-api/draft/API_specification/index.html
for the methods supported by the Consortium Dataframe Standard.

How to try this out
-------------------

Here's an example of how you can try this out:
```python
import polars as pl

df = pl.DataFrame({'a': [1,2,3]})
df_std = df.__dataframe_consortium_standard__()
```
The object `df_std` is a Standard-compliant DataFrame. Check the
[API Specification](https://data-apis.org/dataframe-api/draft/API_specification/index.html)
for the full list of methods supported on it.

Here's an example of a dataframe-agnostic function:
```python
from typing import Any


def my_dataframe_agnostic_function(df_non_standard: Any) -> Any:
    df = df_non_standard.__dataframe_consortium_standard__()

    for column_name in df.column_names:
        new_column = df.col(column_name)
        new_column = (new_column - new_column.mean()) / new_column.std()
        df = df.assign(new_column.rename(f'{column_name}_scaled'))

    return df.dataframe
```

As long as you have this package installed, then either a pandas or Polars DataFrame
should work with the code above, e.g.:

```python
import pandas as pd
import polars as pl

df_pd = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
df_pl = pl.DataFrame({'a': [1,2,3], 'b': [4,5,6]})

my_dataframe_agnostic_function(df_pd)
my_dataframe_agnostic_function(df_pl)
```

Compliance with the Standard
----------------------------
The classes here also have an extra `.persist` method, which is not (yet) part of the Standard.

Installation
------------
```
pip install dataframe-api-compat
```
