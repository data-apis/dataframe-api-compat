# ruff: noqa
import sys
import argparse

sys.path.append("..")


def get_polars_objects():
    import polars as pl

    dataframe = pl.DataFrame({"a": [1]}).__dataframe_consortium_standard__()
    column = pl.Series("a", [1]).__column_consortium_standard__()
    scalar = dataframe.col("a").mean()
    namespace = dataframe.__dataframe_namespace__()
    return dataframe, column, scalar, namespace


def get_pandas_objects():
    import pandas as pd

    dataframe = pd.DataFrame({"a": [1]}).__dataframe_consortium_standard__()
    column = pd.Series([1], name="a").__column_consortium_standard__()
    scalar = dataframe.col("a").mean()
    namespace = dataframe.__dataframe_namespace__()
    return dataframe, column, scalar, namespace


def get_modin_objects():
    import modin.pandas as pd

    dataframe = pd.DataFrame({"a": [1]}).__dataframe_consortium_standard__()
    column = pd.Series([1], name="a").__column_consortium_standard__()
    scalar = dataframe.col("a").mean()
    namespace = dataframe.__dataframe_namespace__()
    return dataframe, column, scalar, namespace


def generate_members(library: str):
    mapper = {
        "pandas": get_pandas_objects,
        "polars": get_polars_objects,
        "modin": get_modin_objects,
    }
    dataframe, column, scalar, namespace = mapper[library]()

    for name, object in [
        (f"{library}-dataframe.md", dataframe),
        (f"{library}-column.md", column),
        (f"{library}-scalar.md", scalar),
        (f"{library}-namespace.md", namespace),
    ]:
        members = [
            i
            for i in object.__dir__()
            if not (i.startswith("_") and not i.startswith("__"))
        ]

        with open(name) as fd:
            content = fd.read()

        members_txt = "\n      - ".join(sorted(members)) + "\n      "

        start = content.index("members:")
        end = content.index("show_signature")
        content = content[:start] + f"members:\n      - {members_txt}" + content[end:]

        with open(name, "w") as fd:
            fd.write(content)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--library",
        type=str,
        default="pandas",
        help="Library for which members will be generated.",
    )
    args = parse.parse_args()
    sys.exit(generate_members(args.library))


if __name__ == "__main__":
    main()
