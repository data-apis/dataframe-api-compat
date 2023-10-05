from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_select(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.select("b")
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_list_of_str(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.select("a", "b")
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_expressions(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.select(col("b") + 1)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"b": [5, 6, 7]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_multiple_expressions(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.select(col("b") + 1, col("b").rename("c") + 2)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"b": [5, 6, 7], "c": [6, 7, 8]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_reduction(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.select(col("a").mean(), col("b").mean())
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [2.0], "b": [5.0]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_broadcast_right(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.select(col("a"), col("b").mean())
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [5.0, 5.0, 5.0]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_select_broadcast_left(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    col = namespace.col
    result = df.select(col("a").mean(), col("b"))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [2.0, 2.0, 2.0], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)
