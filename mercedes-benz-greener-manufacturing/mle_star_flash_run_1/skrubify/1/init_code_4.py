import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from skrub import selectors as s
from sklearn.ensemble import StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read without reading anything on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and reproduce the original hold-out.
    y = data["y"].skb.mark_as_y()
    X = data.drop(columns=["ID", "y"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, then the original stacking ensemble.
    # Newer pandas versions infer CSV text as string dtype rather than object.
    # Include both representations of the original CSV categorical columns so
    # text never reaches the final numeric cast without ordinal encoding.
    categorical_selector = s.has_dtype("object") | s.string()
    categorical = X.skb.select(categorical_selector).astype(str)

    # Fit the encoder on each outer training partition, fixing the original's
    # full-table encoder leakage while retaining its string-cast semantics.
    categorical_encoded = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        cols=s.all(),
    )
    other = X.skb.drop(categorical_selector)

    # Preserve the original feature order and float32 conversion.
    features = other.skb.concat([categorical_encoded], axis=1)
    features = features[X.columns].astype(np.float32)

    # The stacking and ElasticNetCV internal folds remain part of each fit.
    model = StackingRegressor(
        estimators=[
            (
                "lgb",
                lgb.LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.02,
                    max_depth=3,
                    colsample_bytree=0.6,
                    reg_alpha=1.0,
                    random_state=42,
                    verbose=-1,
                ),
            ),
            (
                "xgb",
                xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.02,
                    max_depth=3,
                    colsample_bytree=0.6,
                    reg_alpha=1.0,
                    random_state=42,
                    verbosity=0,
                ),
            ),
            (
                "krr",
                make_pipeline(
                    RobustScaler(),
                    KernelRidge(
                        alpha=0.6,
                        kernel="polynomial",
                        degree=2,
                        coef0=2.5,
                    ),
                ),
            ),
            (
                "enet",
                make_pipeline(
                    RobustScaler(),
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.7, 0.9],
                        cv=5,
                        random_state=42,
                    ),
                ),
            ),
        ],
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    pred = features.skb.apply(model, y=y)

    # 4. Score -- the splitter on mark_as_X drives the single hold-out.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="r2"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
