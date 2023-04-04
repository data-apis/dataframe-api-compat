# todo: test that errors are appropriately raised when calls violate standard

import numpy as np
import pytest
import pandas as pd
from pandas_standard import PandasDataFrame, PandasColumn


def test_from_dict() -> None:
    col_a = PandasColumn(pd.Series([1, 2, 3]))
    col_b = PandasColumn(pd.Series([4, 5, 6]))
    data = {"a": col_a, "b": col_b}
    result = PandasDataFrame.from_dict(data).dataframe
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(reduction: str, expected_data: dict[str, object]) -> None:
    df = pd.DataFrame({"a": [True, True, False], "b": [True, True, True]})
    dfstd = PandasDataFrame(df)
    result = getattr(dfstd, reduction)().dataframe
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", {"a": [True, True, False], "b": [True, False, True]}),
        ("__ne__", {"a": [False, False, True], "b": [False, True, False]}),
        ("__ge__", {"a": [True, True, False], "b": [True, True, True]}),
        ("__gt__", {"a": [False, False, False], "b": [False, True, False]}),
        ("__le__", {"a": [True, True, True], "b": [True, False, True]}),
        ("__lt__", {"a": [False, False, True], "b": [False, False, False]}),
        ("__add__", {"a": [2, 4, 7], "b": [8, 7, 12]}),
        ("__sub__", {"a": [0, 0, -1], "b": [0, 3, 0]}),
        ("__mul__", {"a": [1, 4, 12], "b": [16, 10, 36]}),
        ("__truediv__", {"a": [1, 1, 0.75], "b": [1, 2.5, 1]}),
        ("__floordiv__", {"a": [1, 1, 0], "b": [1, 2, 1]}),
        ("__pow__", {"a": [1, 4, 81], "b": [256, 25, 46656]}),
        ("__mod__", {"a": [0, 0, 3], "b": [0, 1, 0]}),
    ],
)
def test_comparisons(comparison: str, expected_data: dict[str, object]) -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    other = PandasDataFrame(pd.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]}))
    result = getattr(df, comparison)(other).dataframe
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result, expected)


def test_divmod() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    other = PandasDataFrame(pd.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]}))
    result_quotient, result_remainder = df.__divmod__(other)
    expected_quotient = pd.DataFrame({"a": [1, 1, 0], "b": [1, 2, 1]})
    expected_remainder = pd.DataFrame({"a": [0, 0, 3], "b": [0, 1, 0]})
    pd.testing.assert_frame_equal(result_quotient.dataframe, expected_quotient)
    pd.testing.assert_frame_equal(result_remainder.dataframe, expected_remainder)


def test_get_column_by_name() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.get_column_by_name("a")
    np.testing.assert_array_equal(
        np.array([1, 2, 3]), np.from_dlpack(result)  # type: ignore[arg-type]
    )


def test_get_columns_by_name() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.get_columns_by_name(["b"]).dataframe
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_get_rows() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.get_rows([0, 2]).dataframe
    expected = pd.DataFrame({"a": [1, 3], "b": [4, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_slice_rows() -> None:
    df = PandasDataFrame(
        pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]})
    )
    result = df.slice_rows(2, 7, 2).dataframe
    expected = pd.DataFrame({"a": [3, 5, 7], "b": [5, 3, 1]})
    pd.testing.assert_frame_equal(result, expected)


def test_get_rows_by_mask() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    mask = PandasColumn(pd.Series([True, False, True]))
    result = df.get_rows_by_mask(mask).dataframe
    expected = pd.DataFrame({"a": [1, 3], "b": [4, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_insert() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    new_col = PandasColumn(pd.Series([7, 8, 9]))
    result = df.insert(1, "c", new_col).dataframe
    expected = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_drop_column() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.drop_column("a").dataframe
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_set_column() -> None:
    # I'm hoping to get rid of this one, so holding off for now...
    ...


def test_rename_columns() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.rename_columns({"a": "c", "b": "e"}).dataframe
    expected = pd.DataFrame({"c": [1, 2, 3], "e": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_get_column_names() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.get_column_names()
    assert [name for name in result] == ["a", "b"]


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("min", [1, 3], [4, 6]),
        ("max", [2, 4], [5, 7]),
        ("sum", [3, 7], [9, 13]),
        ("prod", [2, 12], [20, 42]),
        ("median", [1.5, 3.5], [4.5, 6.5]),
        ("mean", [1.5, 3.5], [4.5, 6.5]),
        (
            "std",
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, 0.7071067811865476],
        ),
        ("var", [0.5, 0.5], [0.5, 0.5]),
    ],
)
def test_groupby_boolean(
    aggregation: str, expected_b: list[float], expected_c: list[float]
) -> None:
    df = PandasDataFrame(
        pd.DataFrame({"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]})
    )
    result = getattr(df.groupby(["key"]), aggregation)().dataframe
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_numeric(
    aggregation: str, expected_b: list[bool], expected_c: list[bool]
) -> None:
    df = PandasDataFrame(
        pd.DataFrame(
            {
                "key": [1, 1, 2, 2],
                "b": [False, True, True, True],
                "c": [True, False, False, False],
            }
        )
    )
    result = getattr(df.groupby(["key"]), aggregation)().dataframe
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result, expected)


def test_isna() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan]})
    df_std = PandasDataFrame(df)
    result = df_std.isnan().dataframe
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result, expected)


def test_any() -> None:
    df = pd.DataFrame({"a": [False, False], "b": [False, True], "c": [True, True]})
    df_std = PandasDataFrame(df)
    result = df_std.any().dataframe
    expected = pd.DataFrame({"a": [False], "b": [True], "c": [True]})
    pd.testing.assert_frame_equal(result, expected)


def test_all() -> None:
    df = pd.DataFrame({"a": [False, False], "b": [False, True], "c": [True, True]})
    df_std = PandasDataFrame(df)
    result = df_std.all().dataframe
    expected = pd.DataFrame({"a": [False], "b": [False], "c": [True]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("dtype", "expected_values"),
    [("float64", [False, False, False]), ("Float64", [False, False, True])],
)
def test_isnull(dtype: str, expected_values: list[bool]) -> None:
    df = pd.DataFrame({"a": [0.0, 1.0, np.nan]}, dtype=dtype)
    df = df / df
    df_std = PandasDataFrame(df)
    result = df_std.isnull().dataframe
    expected = pd.DataFrame({"a": expected_values})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("dtype", "expected_values"),
    [("float64", [True, False, True]), ("Float64", [True, False, False])],
)
def test_isnan(dtype: str, expected_values: list[bool]) -> None:
    df = pd.DataFrame({"a": [0.0, 1.0, np.nan]}, dtype=dtype)
    df = df / df
    df_std = PandasDataFrame(df)
    result = df_std.isnan().dataframe
    expected = pd.DataFrame({"a": expected_values})
    pd.testing.assert_frame_equal(result, expected)
