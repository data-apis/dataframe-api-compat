# Basics

The basic premise of the project is this: you want to write a function which supports both pandas and Polars

The steps you'll want to follow are:

1. Opt-in to the DataFrame API by calling `__dataframe_consortium_standard__` on your dataframe.
2. Express your logic using methods from the [DataFrame API](https://data-apis.org/dataframe-api/draft/API_specification/index.html)
   You may want to look at the [official examples](https://github.com/data-apis/dataframe-api/tree/main/spec/API_specification/examples)
   for inspiration.
3. If you need to return a dataframe to the user in its original library, call `DataFrame.dataframe`.

Let's try writing a simple example.

## Simple example

Make a Python file `t.py` with the following content:
```python
import pandas as pd
import polars as pl

def my_function(df):
    # 1. Opt-in to the API Standard
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    # 2. Use methods from the API Standard spec
    df_s = df_s.group_by('a').mean()
    # 3. Return a library from the user's original library
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
df_pl = pl.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})

print('* pandas result:')
print(my_function(df_pd))
print()
print('* Polars result:')
print(my_function(df_pl))
```
Let's try this out - if you run it, you should see the following output:
```
* pandas result:
   a    b
0  1  4.5
1  2  6.0

* Polars result:
naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)

AGGREGATE
        [col("b").mean()] BY [col("a")] FROM
  DF ["a", "b"]; PROJECT */2 COLUMNS; SELECTION: "None"
**********
```

Let's try to make sense of it:

- For pandas, the output is a `pandas.DataFrame`.
- But for Polars, the output is a `polars.LazyFrame`.

This is because the DataFrame API only has a single `DataFrame` class - so for Polars,
all operations are done lazily in order to make full use of Polars' query engine.
If you want to convert that to a `polars.DataFrame`, it is the caller's responsibility
to call `.collect` - for example:
```python
print('* Polars result:')
print(my_function(df_pl).collect())
```
will give you the output:
```python
* Polars result:
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ f64 │
╞═════╪═════╡
│ 1   ┆ 4.5 │
│ 2   ┆ 6.0 │
└─────┴─────┘
```
