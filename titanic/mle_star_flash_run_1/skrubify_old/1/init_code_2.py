import lightgbm as lgb
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedShuffleSplit


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without reading data at import time.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and features before preprocessing.
    # StratifiedShuffleSplit reproduces the original stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and native LightGBM categoricals.
    X = X.assign(
        Title=X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    )
    X = X.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    X = X.assign(IsAlone=(X["FamilySize"] == 1).astype(int))
    X = X.astype(
        {
            "Sex": "category",
            "Embarked": "category",
            "Title": "category",
            "Pclass": "category",
        }
    )
    features = X[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title",
            "FamilySize",
            "IsAlone",
        ]
    ]

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    pred = features.skb.apply(model, y=y)

    # 4. Score — the splitter on mark_as_X drives the single hold-out.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
