# ruff: noqa
import sys

sys.path.append("..")

import pandas as pd
import polars as pl

pd_series = pd.Series([1], name="a").__column_consortium_standard__()
pl_series = pl.Series("a", [1]).__column_consortium_standard__()

for name, object in [
    ("pandas-column.md", pd_series),
    ("polars-column.md", pl_series),
]:
    members = [i for i in object.__dir__() if not (i.startswith("_") or "namespace" in i)]

    with open(name) as fd:
        content = fd.read()

    members_txt = "\n      - ".join(sorted(members)) + "\n      "

    start = content.index("members:")
    end = content.index("show_signature")
    content = content[:start] + f"members:\n      - {members_txt}" + content[end:]

    with open(name, "w") as fd:
        fd.write(content)
