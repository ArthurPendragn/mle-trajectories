# `OrdinalEncoder(unknown_value=0)` fails at fit

Skrubify transcribes a `LabelEncoder` unseen-category fallback of `0` as
`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=0)`. The plan
imports. Scoring then raises:

```text
ValueError: The used value for unknown_value 0 is one of the values already
used for encoding the seen categories.
```

Seen in `titanic/mle_star_flash_run_1/skrubify/1/train0_improve3.py`, and
again in `train1_improve3.py` and `train1_improve4.py`. `.skb.apply` broadcasts
the encoder per column, so the traceback is an `Apply OrdinalEncoder` node
inside `ApplyToCols`, not one encoder on the whole frame. The fit fails
before any unknown row is transformed: on this Titanic split the scored
hold-out had no category absent from the training rows, and `unknown_value=0`
still raised.

`train1_improve0.py`, `train1_improve1.py`, and `train1_improve2.py` already
used the spelling below and still disagreed with their baselines. The
encoder was not the cause. Their `eval_set` was an inner split, so the `-1`
then `replace(-1, 0)` path never saw the scored hold-out.

That spelling is for rows that are both outputs of the encoder node. A
scored hold-out taken before `mark_as_X` never enters an `OrdinalEncoder`
downstream of the mark. The corrected `train0_improve*` and `train1_improve*`
plans fit a `LabelEncoder` inside the network estimator, on the outer
training rows, and map the hold-out with `mapping.get(v, 0)`. That is the
same alias, applied to both matrices the original encoded.

## Why `0` is illegal

`OrdinalEncoder` assigns the first seen category the code `0`, the next `1`,
and so on. With `handle_unknown="use_encoded_value"`, `fit` rejects an
`unknown_value` that sits in `0 .. cardinality-1` for any column
(`sklearn/preprocessing/_encoders.py`). A column with at least one category
always uses `0`, so `unknown_value=0` always fails. `-1` is outside that
range and is accepted.

The original pattern this is copied from is not an `OrdinalEncoder` at all:

```python
mapping = {val: i for i, val in enumerate(le.classes_)}
encoded = np.array([mapping.get(v, 0) for v in values])
```

`mapping.get(v, 0)` sends an unseen label to code `0`, which is also the first
seen class. That alias is intentional: the embedding table has one row per
seen class (`len(le.classes_)`), and an unseen label shares the first class's
row. `unknown_value=0` looks like that fallback and is a different, illegal
parameter.

## How to write it

Encode unknowns as `-1`, then put those cells back on `0`. Seen codes stay
`0 .. n-1`, so `max(codes) + 1` is still `len(classes_)`.

```python
encoded = categories.skb.apply(
    OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).set_output(transform="pandas")
)
encoded = encoded.replace(-1, 0).astype(np.int64)
```

Leave the `-1` in place only when the downstream model accepts a negative
code. An embedding index does not: `nn.Embedding` rejects `-1`.

## Why skrubify keeps emitting it

The converter copies the integer in `mapping.get(v, 0)` into `unknown_value`.
The static checks and the import under `eager_data_ops=False` never call
`fit`, and `OrdinalEncoder.fit` is the only place this is validated. A plan
that builds is not evidence the encoder can be fit. The same failure shows up
on the next conversion unless the prompt says: a `LabelEncoder` fallback of
`0` is `unknown_value=-1` followed by `replace(-1, 0)`, never
`unknown_value=0`.
