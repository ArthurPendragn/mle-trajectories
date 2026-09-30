"""Persisted, self-consistent samples of a dataset, built from its dataset.toml recipe.

The builder lives in ``core`` (numpy, pandas, pyarrow); ``_manifest`` is
stdlib-only, for tools that only need to recognise a sample.
"""
