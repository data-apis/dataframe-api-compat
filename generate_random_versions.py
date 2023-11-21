"""
Generate files with:

    pip index versions pandas > pandas_versions.txt
    pip index versions polars > polars_versions.txt
"""
import random
import re

MIN_PANDAS_VERSION = "1.0.0"
MIN_POLARS_VERSION = "0.15.0"

with open("pandas_versions.txt") as fd:
    content = fd.readlines()[1]
versions = re.findall(r", (\d+\.\d+\.\d+)", content)
pandas_version = random.choice([i for i in versions if i >= MIN_PANDAS_VERSION])

with open("polars_versions.txt") as fd:
    content = fd.readlines()[1]
versions = re.findall(r", (\d+\.\d+\.\d+)", content)
polars_version = random.choice([i for i in versions if i >= MIN_POLARS_VERSION])

content = f"pandas=={pandas_version}\npolars=={polars_version}\n"
with open("random-requirements.txt", "w") as fd:
    fd.write(content)
