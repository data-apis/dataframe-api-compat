# Column

In [dataframe.md](dataframe.md), you learned how to write a dataframe-agnostic function.

We only used DataFrame methods there - but what if we need to operate on its columns?

## Extracting a column

To extract a column from a dataframe, you can use the `DataFrame.col` method. Let's look
at some examples.

## Example 1: filter based on a column's values

Let's write a dataframe-agnostic function which keeps rows in a dataframe where the column
`'a'`'s values are greater than zero.
```python
import pandas as pd
import polars as pl

def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.filter(df_s.col('a') > 0)
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd))
print()
print('Polars output:')
print(my_func(df_pl).collect())
```
This outputs:
```
pandas output:
   a  b
0  1  5
1  3 -3

Polars output:
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 5   │
│ 3   ┆ -3  │
└─────┴─────┘
```

## Example 2: multiply a column's values by a constant

Let's write a dataframe-agnostic function which multiplies the values in column
`'a'` by 2.
```python
import pandas as pd
import polars as pl

def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.assign(df_s.col('a')*2)
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd))
print()
print('Polars output:')
print(my_func(df_pl).collect())
```
This outputs:
```
pandas output:
   a  b
0 -2  3
1  2  5
2  6 -3

Polars output:
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ -2  ┆ 3   │
│ 2   ┆ 5   │
│ 6   ┆ -3  │
└─────┴─────┘
```

Note that column `'a'` was overwritten. If we had wanted to add a new column called `'c'` containing column `'a'`'s
values multiplied by 2, we could have used `Column.rename`:
```python
import pandas as pd
import polars as pl

def my_func(df):
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    df_s = df_s.assign((df_s.col('a')*2).rename('c'))
    return df_s.dataframe

df_pd = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
df_pl = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
print('pandas output:')
print(my_func(df_pd))
print()
print('Polars output:')
print(my_func(df_pl).collect())
```
This outputs:
```
pandas output:
   a  b  c
0 -1  3 -2
1  1  5  2
2  3 -3  6

Polars output:
shape: (3, 3)
┌─────┬─────┬─────┐
│ a   ┆ b   ┆ c   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ -1  ┆ 3   ┆ -2  │
│ 1   ┆ 5   ┆ 2   │
│ 3   ┆ -3  ┆ 6   │
└─────┴─────┴─────┘
```

## Example 3: cross-dataframe column comparisons
