# A scored hold-out is part of the pipeline

Skrubify rewrites two things that an original single hold-out script does on
purpose, then reports that the original score cannot be reproduced. Both
rewrites change the fitted model. The printed accuracy can stay equal while
the predictions do not.

Seen in `titanic/mle_star_flash_run_1/skrubify/1/train0_improve3.py` before it
was aligned with `train0_improve0.py`, `train0_improve1.py`, and
`train0_improve2.py`. On `/home/tim/mle-trajectories/titanic` the broken plan
and the baseline both printed accuracy `0.8321678321678322`. Six of the 143
predictions differed, and the first training-loop `accuracy_score` was
`0.8041958041958042` against `0.8070175438596491`.

## What the original does

`pipelines/1/train0_improve3.py` builds features on the whole `train.csv`,
including the rows it will score:

- `Age` and `Fare` are filled with the full-table median
- `TicketFreq` is `Ticket` mapped through the full-table `value_counts`

It then takes one stratified 80/20 split (`train_test_split`, `test_size=0.2`,
`random_state=42`). CatBoost early-stops on that validation slice. The network
scheduler reads that same slice's accuracy every epoch, for all 150 epochs,
and both models are scored there.

## What skrubify writes instead

The prompt tells the converter to do both of these
(`tools/skrubify/prompt.py`):

1. A statistic fitted before the split (a median, a `value_counts`, an
   encoder) is moved after `mark_as_X`, usually as `SimpleImputer` or another
   estimator, so each fold refits it on training rows only.
2. Early stopping on the scored split is replaced by `GetXY`. That transformer
   cuts a fresh 20% out of the outer training rows and passes that inner slice
   as `eval_set`. The same inner slice is what the network scheduler watches.
   The prompt says the original score cannot be reproduced and that a comment
   should say so.

`GetXY` is the right pattern when the original already early-stopped on a
slice of its training rows. It is the wrong pattern when the original's
`eval_set` is the slice the script scores. Refitting the median and the ticket
counts per fold is the same kind of change: the validation rows no longer
contribute to those statistics, and the training rows are imputed with a
different value.

The plan still builds and still scores. Nothing in the static checks compares
predictions with the baseline.

## How to write it

Do the original's full-table work before the marks. Cut the scored rows with
the same `train_test_split` the original uses, and hand that frame to
CatBoost and to the network. The outer splitter stays
`StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`, which
is that `train_test_split`.

```python
def holdout_rows(df):
    indices = np.arange(len(df))
    _, idx_val, _, _ = train_test_split(
        indices,
        df["Survived"],
        test_size=0.2,
        random_state=42,
        stratify=df["Survived"],
    )
    return df.iloc[idx_val]


# medians, ticket counts, and the other full-table features happen here,
# on `features`, before either mark
holdout = features.skb.apply_func(holdout_rows)
X_val = holdout[FEATURE_COLS]
y_val = holdout["Survived"]

y = features["Survived"].skb.mark_as_y()
X = features.drop(columns=["Survived"]).skb.mark_as_X(
    cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
    split_kwargs={},
)

cb_probability = X[FEATURE_COLS].skb.apply(
    CatBoostClassifier(...),
    y=y,
    fit_kwargs={
        "cat_features": CAT_COLS,
        "eval_set": (X_val, y_val),
        "early_stopping_rounds": 50,
    },
    predict_kwargs={"prediction_type": "Probability"},
)
nn_probability = X[FEATURE_COLS].skb.apply(
    NeuralPositiveProbability(),
    y=y,
    fit_kwargs={"X_val": X_val, "y_val": y_val},
)
```

`cat_features` stays in `fit_kwargs`. A constructor `cat_features` fails
`clone`. The network fits its `LabelEncoder` and `StandardScaler` on the
outer training rows only, and an unseen category still maps to code `0`.
`torch.manual_seed` goes immediately before the module is constructed. The
equal-weight `>= 0.5` average stays in one estimator
(`MeanProbabilityEnsemble` in the files above): during fit the two `.skb.apply`
results are the fitted models, so a DataOps `0.5 *` raises `TypeError`.

`train0_improve0.py`, `train0_improve1.py`, `train0_improve2.py`, and the
corrected `train0_improve3.py` in
`titanic/mle_star_flash_run_1/skrubify/1/` are that plan. After the change the
checker reports identical predictions and 116 identical training-loop
`accuracy_score` calls.

The same rewrite shows up again on the next solution's improvement chain,
`train1_improve0.py` through `train1_improve4.py`. The full-table statistics
there are an `Age` median by `Pclass` and `Sex`, a global `Age` fallback, a
`Fare` median, and `Ticket` `value_counts`. On the full table those formulas
already matched the original; the score moved because each one was refit on
outer training rows only, and because CatBoost's `eval_set` was a further
20% cut out of those rows.

That inner slice was also what the network watched. Across the five scripts
that is checkpoint selection, a top-k weight average, stochastic weight
averaging and its early-stop patience, temperature scaling, snapshot
selection, and the EMA model's per-epoch accuracy. The last of those does
not change the weights. Dropping the `accuracy_score` call still fails the
comparison, because the checker requires the training-loop metric calls to
agree. The corrected files pass the scored hold-out into both `fit_kwargs`
and keep that call.

## Why skrubify keeps emitting it

Two prompt rules require the rewrite. "Where the original leaks" says a
pre-split imputation or encoder should be refit per fold. "If the original
early-stops on the very split it reports as validation" says to keep the
patience, carve the eval set with `GetXY`, and accept that the score cannot
be reproduced. Both rules describe a leakage cleanup. They are the wrong
instruction when the conversion is supposed to produce the original's
predictions. The sentence that would stop it: a statistic or an `eval_set`
computed on the scored hold-out stays on those same rows, passed in as a
DataOp from before `mark_as_X`.
