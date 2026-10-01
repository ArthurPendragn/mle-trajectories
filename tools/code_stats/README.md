# code_stats

Static features of a pipeline script, from its source alone: nothing is
imported or run (stdlib `ast`, ~25 ms for a 1,000-line script), so it works for
every run, with or without skrub plans or data. The website's run page shows it
as the **Code** section, and a "code vs parent" column in the steps table.

```bash
python -m code_stats tab_playground_dec_21/mle_star/pipelines/train0_improve0.py
python -m code_stats parent.py child.py --diff
```

## Features (`analyze_source`)

- **size**: lines, lines of code (no blanks, comments, docstrings), comments, docstrings.
- **structure**: functions, classes, `for`/`while`, `if`, `try`, `with`,
  comprehensions, lambdas, maximum nesting depth, cyclomatic complexity
  (1 + decision points: branches, loops, handlers, boolean operators, comprehension ifs).
- **components**: calls into ML libraries, resolved through the file's own
  imports (`lgb.LGBMRegressor` after `import lightgbm as lgb` is
  `lightgbm.LGBMRegressor`) and classified by `catalog.py` as model, ensemble,
  transformer, pipeline, CV splitter, search, metric, optimizer, scheduler,
  loss, layer, data, training. A class subclassing `nn.Module` is a model,
  unless only other networks use it (then a layer) or it is a `...Loss`. Each
  comes with its **keyword arguments**: literal values as written, module-level
  constants resolved (`n_splits=N_SPLITS` → 5), anything else kept as the
  expression (`=params["lr"]`); `**kwargs` is recorded as `"**"`.
- **data**: files read and written (literal paths, f-strings, `os.path.join`,
  `Path / ...`; `./` stripped), `df["col"] = ...` writes and distinct column
  names, `.loc/.iloc/.at` writes, `inplace=True`, pandas- and polars-specific
  method counts.
- **other**: `pip install` at run time, shell calls, GPU use, seeds, prints.

A file that does not parse returns `ok: false` with the error (four beaver
MLE-STAR files are truncated by the agent).

## What it cannot know

It reads the *written* code: a branch that never runs still counts, a model
built in a loop counts once, a value computed at run time stays an
expression. A receiver's type is unknown, so methods are counted by name, only
for names specific to pandas or polars, and only in files that import that
library (`gc.collect()` is not polars).

## Step vs parent (`compare`)

Lines added/removed and similarity (`difflib` on lines), whether the code is
the same once comments and formatting are ignored, components added/removed,
hyperparameters changed (the i-th occurrence of a component against the i-th
in the parent; a parameter missing on a side that passes `**kwargs` is not
reported), imports and files read added/removed.
