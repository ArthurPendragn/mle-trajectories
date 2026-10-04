"""final_pipeline -- FINAL refit/predict artifact, NOT a scored candidate.

Do not run this through ml-score. It is pipeline_24 (best: 0.96184 accuracy)
with the winning choose_from values hard-coded (hidden=512, n_layers=3), refit
on ALL of train.csv, then used to predict test.csv -> submission.csv.

Only deviation from pipeline_24: device is auto-resolved (cuda/mps/cpu) instead
of the hard-coded "mps" the run used on macOS.
"""
from pathlib import Path

import pandas as pd
import skrub

from features import best_features
from nn import TorchMLP

WS_ROOT = Path(__file__).resolve().parent.parent
TARGET = "Cover_Type"

train_df = pd.read_csv(WS_ROOT / "input" / "train.csv")
data = skrub.var("data", value=train_df)

y = data[TARGET].skb.mark_as_y()
X = data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
X = best_features(X.skb.drop(cols="Id"))
model = TorchMLP(
    hidden=512, n_layers=3,
    dropout=0.25, lr=1e-3, max_epochs=35, batch_size=8192, device=None,
)
pred = X.skb.apply(model, y=y)

learner = pred.skb.make_learner(fitted=True)

test_path = WS_ROOT / "input" / "test.csv"
if test_path.is_file():
    test_df = pd.read_csv(test_path)
    preds = learner.predict({"data": test_df})

    sample = pd.read_csv(WS_ROOT / "input" / "sample_submission.csv")
    id_col, target_col = sample.columns[0], sample.columns[1]
    submission = pd.DataFrame({id_col: test_df[id_col], target_col: preds})
    submission.to_csv(WS_ROOT / "submission.csv", index=False)
    print(f"wrote {WS_ROOT / 'submission.csv'} ({len(submission)} rows)")
else:
    print(f"no test.csv in {WS_ROOT / 'input'} -- final_pipeline.py written, "
          "no submission to generate")
