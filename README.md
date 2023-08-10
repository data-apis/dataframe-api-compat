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
for pandas and polars-eager

Note: there is ongoing discussion about lazy engines in the Standard. Until that has been resolved,
this package should not be relied upon for polars-lazy.

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

Compliance with the Standard
----------------------------
This is mostly compliant. Notable differences:
- for pandas numpy dtypes, the null values (NaN) don't follow Kleene logic;
- for polars lazy, column reductions (e.g. `column.mean()`) are not implemented;
- for polars lazy, comparisons between different dataframes are not implemented.

Installation
------------
```
pip install dataframe-api-compat
```
