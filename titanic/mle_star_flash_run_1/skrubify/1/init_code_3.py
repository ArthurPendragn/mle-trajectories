import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder


class TitleAgeImputer(TransformerMixin, BaseEstimator):
    """Learn title-specific age medians from each fold's training rows."""

    def fit(self, X, y=None):
        self.title_medians_ = X.groupby("Title")["Age"].median()
        return self

    def transform(self, X):
        age = X["Age"].fillna(X["Title"].map(self.title_medians_))
        return age.to_frame(name="Age")


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without accessing the file on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and preserve the exact stratified
    #    80/20 hold-out used by train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and fold-fitted preprocessing.
    features = X.assign(
        Title=X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        Embarked=X["Embarked"].fillna("S"),
    )

    # The original learned all medians before splitting. Learn them on training
    # rows only, including the overall Age median AFTER title-based filling.
    age_by_title = features[["Title", "Age"]].skb.apply(TitleAgeImputer())
    age = age_by_title.skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )
    fare = features[["Fare"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )

    # Sorted ordinal codes match LabelEncoder on each string-valued column.
    # Fit vocabularies per fold rather than on the whole table; unseen held-out
    # categories receive -1 because their training-fold vocabulary has no code.
    categories = features[["Sex", "Embarked", "Title"]].astype(str)
    encoded = categories.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    )

    features = features.assign(
        Sex=encoded["Sex"],
        Age=age["Age"],
        Fare=fare["Fare"],
        Embarked=encoded["Embarked"],
        Title=encoded["Title"],
    )
    design = features[
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
        ]
    ]

    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )
    pred = design.skb.apply(model, y=y)

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
