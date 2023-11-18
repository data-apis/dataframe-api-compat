# Dataframe-API-compat

Extremely lightweight compatibility layer between pandas and Polars:

- ✅ No dependencies.
- ✅ Lightweight: wheel is smaller than 30 kB.
- ✅ Will support any other dataframe library which conforms to the [spec](https://data-apis.org/dataframe-api/draft/API_specification/index.html),
  even those which don't exist yet.

Make your library completely dataframe-agnostic!

## Who's this for?

Anyone wishing to write a library/application/service which consumes dataframes, and wishing to make it
completely dataframe-agnostic.

## Who's this not for?

This is not targeted at data scientists or analysts primarily doing exploratory data analysis.
However, if you're writing a tool for data professionals, then `dataframe-api-compat` may help you
make your tool dataframe-agnostic.

## Let's get started!

- [Installation](installation.md)
- [Quick start](quick_start.md)
- [Basics](basics.md)
- [Tutorial](tutorial.md)
