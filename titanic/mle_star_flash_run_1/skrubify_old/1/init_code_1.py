"""init_code_1: CatBoost on one stratified hold-out.

The original fills Fare with the full-table median, builds the title, family,
and cabin features, then splits once. It early-stops on the same validation
rows it scores. That preparation and that split stay upstream of
``mark_as_X``. ``StratifiedShuffleSplit`` with the same ``test_size`` and
``random_state`` selects those rows again as the scored fold.

CatBoost is a plain ``CatBoostClassifier``. ``cat_features`` is a fit argument
because the constructor form fails ``clone``.
"""

import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "CabinDeck",
]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]


with skrub.config_context(eager_data_ops=False):
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Full-table median, matching the original preparation before its split.
    prepared = data.assign(
        Title=data["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing"),
        FamilySize=data["SibSp"] + data["Parch"] + 1,
        CabinDeck=data["Cabin"].str[0].fillna("Missing"),
        Embarked=data["Embarked"].fillna("Missing"),
        Fare=data["Fare"].fillna(data["Fare"].median()),
    )
    model_features = prepared[FEATURE_COLS]

    # The original early-stops on the same validation rows it scores. Keep
    # that split here. The marks below are separate nodes, so this split
    # stays the full table.
    split_target = data["Survived"]
    fixed_split = model_features.skb.apply_func(
        train_test_split,
        split_target,
        test_size=0.2,
        random_state=42,
        stratify=split_target,
    )
    X_validation = fixed_split[1]
    y_validation = fixed_split[3]

    y = data["Survived"].skb.mark_as_y()
    X = model_features.skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    pred = X.skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y,
        fit_kwargs={
            "cat_features": CAT_COLS,
            "eval_set": (X_validation, y_validation),
            "early_stopping_rounds": 50,
        },
    )


if __name__ == "__main__":
    search = pred.skb.make_grid_search(
        n_jobs=1, fitted=True, refit=False, scoring="accuracy"
    )
    print(search.results_)
    for variant_score in search.results_["mean_test_score"]:
        print(f"Variant score: {variant_score}")
    print(
        "Final Validation Performance: "
        f"{search.results_['mean_test_score'].iloc[0]}"
    )
