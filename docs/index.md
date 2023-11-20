# Dataframe-API-compat

Extremely lightweight compatibility layer between pandas and Polars:

- ✅ No dependencies.
- ✅ Lightweight: wheel is smaller than 30 kB.
- ✅ Simple, minimal, perfectly backwards-compatible* API.

Aiming to transform the data science landscape by allowing tools to natively support
any major dataframe library.

*after the first non-beta release, scheduled for February 2024.

## Who's this for?

Anyone wishing to write a library/application/service which consumes dataframes, and wishing to make it
completely dataframe-agnostic.

## Who's this not for?

This is not targeted at data scientists or analysts primarily doing exploratory data analysis.
However, if you're writing a tool for data professionals, then `dataframe-api-compat` can make
make your tool dataframe-agnostic.

## Let's get started!

- [Installation](installation.md)
- [Quick start](quick_start.md)
