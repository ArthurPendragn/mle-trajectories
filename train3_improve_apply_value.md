# Why the train3 improve plans did not match their baselines

Checked with skrub 0.11.0 on the full Titanic training table
(`titanic/input/train.csv`, 891 rows). There is no sample for this dataset.
The comparison is the pipeline-equivalence check: the DataOps rewrite and the
original script must print the same accuracy with `==`, return the same
`y_true` and `y_pred`, and make the same `sklearn.metrics` calls in the same
order while training. Printed wording is ignored.

## The two copies of each pipeline

The originals are the agent scripts. They are not edited.

- `titanic/mle_star_flash_run_1/pipelines/1/train3_improve0.py`
- `titanic/mle_star_flash_run_1/pipelines/1/train3_improve1.py`
- `titanic/mle_star_flash_run_1/pipelines/1/train3_improve2.py`
- `titanic/mle_star_flash_run_1/pipelines/1/train3_improve3.py`
- `titanic/mle_star_flash_run_1/pipelines/1/train3_improve4.py`

The DataOps rewrites live next to this note, under
`titanic/mle_star_flash_run_1/skrubify/1/`, with the same file names. A DataOps
plan is a skrub computation graph. `X.skb.apply(estimator, y=y)` records
"fit this estimator". `mark_as_X` / `mark_as_y` choose the rows the
cross-validation scorer holds out. `pred.skb.make_grid_search(...)` is the
scoring call. It is a scikit-learn `GridSearchCV` over the choices in the
graph.

## What every original script does

All five originals share one procedure and differ only after the neural net.

`prepare_features` runs on the whole training table, before any split. Age is
filled with the median inside each `Pclass`/`Sex` group, then with the table
median. Fare is filled with the table median. `TicketFreq` is the ticket's
count in the whole table. Those statistics therefore include the rows that are
later called the validation set.

The split is one call:

```python
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)
```

That hold-out is 143 rows. CatBoost (600 iterations, learning rate 0.03,
depth 5, `eval_metric="Accuracy"`, `random_seed=42`, early stopping after 50
rounds) uses it as `eval_set`. The network (hidden widths 64 and 32, batch 32,
AdamW at `1e-2` with weight decay `1e-4`, dropout 0.2, 150 epochs,
`ReduceLROnPlateau`) computes accuracy on it every epoch. The ensemble then
reads the two models' probabilities on those same 143 rows, picks its blending
rule from that accuracy, and the script reports accuracy on those same rows.

The ensembles are:

- `train3_improve0.py`. Linear blend. 101 weights from `linspace(0, 1, 101)`
  and 81 thresholds from `linspace(0.1, 0.9, 81)`. Keep a candidate only when
  its accuracy is strictly greater than the best so far. Score
  `0.8531468531468531`.
- `train3_improve1.py`. Clip probabilities at `1e-6`, take logits, stack them,
  and fit `LogisticRegression(C=1, solver="lbfgs", max_iter=1000,
  random_state=42)` in a `StratifiedKFold(5, shuffle=True, random_state=42)`
  on the hold-out. Threshold the out-of-fold probability at 0.5. Score
  `0.8321678321678322`.
- `train3_improve2.py`. Isotonic regression, out of fold, with
  `KFold(5, shuffle=True, random_state=42)`. Clip at `1e-6`. Search a
  power mean with Nelder-Mead from `[0.5, 1.0, 0.5]`, `maxiter=500`,
  `xatol=fatol=1e-4`, and clip the weight and the threshold. Score
  `0.8391608391608392`.
- `train3_improve3.py`. Logit blend with probabilities clipped at `1e-7`.
  101 weights times 21 temperatures from `logspace`, thresholds at the unique
  quantiles of `linspace(0.01, 0.99, 100)`. Keep a candidate only when its
  accuracy is strictly greater. Score `0.8531468531468531`.
- `train3_improve4.py`. Meta features from the two probabilities: the raw
  values, `rankdata / n`, difference, absolute difference, product, power means
  at 0.5 and 2, and three entropies. `LogisticRegressionCV` (`Cs` of 20 values
  from `logspace(-4, 4)`, `l2`, `lbfgs`, inner `cv=3`, accuracy, `max_iter=1000`,
  `random_state=42`) inside `CalibratedClassifierCV(method="sigmoid", cv=3)`,
  again out of fold on a stratified 5-fold of the hold-out. The threshold
  comes from the ROC curve; a non-finite value or a value above 1 becomes 0.5.
  Score `0.8531468531468531`.

`StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` draws this
same partition, in the same order, as that `train_test_split`.

## First failure: an intermediate model arrived as a Series

The first rewrites treated each base model as the value of its
`.skb.apply()` node, and several of them branched on `skrub.eval_mode()`:

```python
probabilities = value.predict(meta_X) if mode == "fit" else value
```

`train3_improve2.py` and `train3_improve3.py` passed that node through
`fit_kwargs` (`cb_model`, `nn_model`) and called `predict_proba` on it.
`train3_improve0.py` did the same inside its blend estimator.

That pattern is right for the last predictor in the plan. While the learner
is fitting, the final `.skb.apply()` evaluates to the fitted estimator, and
`eval_mode()` is `"fit"`. While the held-out fold is scored, `eval_mode()` is
`"predict"` and the node evaluates to predictions. Gating a final
post-processing step on that string matches the node's value.

These CatBoost and network nodes are inputs of a later `.skb.apply()`, the
ensemble. Skrub then changes the method of the earlier node so the later step
receives data: `fit` becomes `fit_transform`, and `predict` becomes
`transform`. A classifier has no `transform`, so skrub fits it and calls
`predict`. The target was marked as a column, so the 1-d prediction array is
wrapped as a pandas `Series`.

`eval_mode()` is still `"fit"`. That string is the method of the learner that
is currently running, which is the ensemble's `fit`. It does not say that this
earlier node holds an estimator. The `mode == "fit"` branch therefore calls
`.predict` on the `Series`:

```text
AttributeError: 'Series' object has no attribute 'predict'
```

`predict_proba` fails the same way. The concrete sites were
`meta_probabilities` in improve1, `CalibratedPowerMeanEnsemble.fit` in
improve2, `CalibratedLogitEnsemble.fit` in improve3, and
`probabilities_for_stacking` in improve4.

The `Series` is also predictions on the rows passed as `X` into that apply,
the training rows of the outer fold. The originals need probabilities on the
hold-out frame, which is a different set of rows.

The fitted object is stored on the apply node. `.skb.applied_estimator`
evaluates that node and returns `estimator_`. That is the object
`predict_proba` has to be called on. `train4.py` in this folder already passes
its base models that way.

## Second failure: the blend grid never finished

`train3_improve0.py` put the 101 by 81 blend in `skrub.choose_from` before it
could hit the `Series` error. `make_grid_search` refits the whole plan for
every cell: CatBoost and the 150-epoch network, 8,181 times. The equivalence
check stops a run at 1800 seconds. That run was still inside the first fits
and had recorded no metric.

Moving the sweep into one estimator `fit`, so both models are trained once,
is what lets the search finish. `choose_from` is the wrong place for a search
whose score does not require refitting CatBoost and the network.

## Third failure: the scores were still different

Stopping the crash is not the same as reproducing the original number. The
rewrites that got past the `Series` error were built as an ordinary
cross-validation:

- Age, Fare, and ticket frequency were computed inside the outer training
  fold, so the 143 scored rows did not enter those medians and counts.
- CatBoost's early stopping and the network's learning-rate schedule used an
  inner slice of that training fold.
- The ensemble was chosen on that inner slice.
- The printed accuracy was the outer hold-out, which none of those steps had
  used.

The originals do the opposite. The feature statistics see every row. Early
stopping, the ensemble search, and the printed accuracy are one and the same
143-row hold-out. The first disagreement in the metric-call log was already
epoch 1 of the network: `accuracy_score` was being computed on different rows,
so the weights diverged before the ensemble ran.

The completed-but-different scores were:

| Plan | Original | Rewrite that used an inner hold-out |
|---|---|---|
| `train3_improve0.py` | 0.8531468531468531 | 0.8391608391608392 |
| `train3_improve1.py` | 0.8321678321678322 | 0.8394088669950739 |
| `train3_improve2.py` | 0.8391608391608392 | 0.8321678321678322 |
| `train3_improve3.py` | 0.8531468531468531 | 0.8321678321678322 |
| `train3_improve4.py` | 0.8531468531468531 | 0.8321678321678322 |

One more crash showed up on improve0 once the 8,181-cell grid was removed and
the plan actually fitted. Its `OrdinalEncoder` used `unknown_value=0`. Skrub
0.11 rejects that when 0 is already a category code:

```text
ValueError: The used value for unknown_value 0 is one of the values already used for encoding the seen categories
```

That encoder was part of the inner-hold-out rewrite. The matching plans do not
use it.

## What has to be true for the score to match

The original accuracy includes the hold-out rows in the feature statistics and
in the model selection. A plan that keeps those rows out will print a
different number, even with the same estimators and the same seeds.

Matching the original means copying its procedure:

1. Build title, age, family size, ticket frequency, fare, and embarked on the
   full table, and only then call `mark_as_X` / `mark_as_y`.
2. Take the validation frame from that full table with the same
   `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
   The return order is `X_train, X_val, y_train, y_val`, so element 1 is
   `X_val` and element 3 is `y_val`. This node stays upstream of the marks.
   During fitting, a node downstream of `mark_as_X` sees only the training
   slice, so a split made there is an inner split and is a different 143
   rows from the fold being scored.
3. Set the scorer's split to
   `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`
   with `split_kwargs={}`. That is the same partition, in the same order, as
   the `train_test_split` above, so the early-stopping rows and the scored
   rows are the same rows.
4. Pass `.skb.applied_estimator` of the CatBoost apply and of the network
   apply into the ensemble. Run the original ensemble math on the validation
   frame inside `fit`. `predict` must return the labels for the scored fold
   in that fold's index order. For the out-of-fold ensembles that label vector
   is not a function of a single `predict` on `X`; it has to be stored indexed
   by the validation index and reindexed to `X.index`.
5. Leave the 8,181 blend cells out of `choose_from`. The sweep runs once,
   inside the ensemble `fit`, on probabilities that are already computed.

`train3.py` in this folder is that feature graph, that split, and those two
model applies, and it already matched its own original. The five improve plans
are that plan with the original ensemble in place of the mean-probability
blend. The equivalence check then reports a match for each file, with
identical predictions of shape `(143,)` and dtype `int64`, at the original
scores listed above.
