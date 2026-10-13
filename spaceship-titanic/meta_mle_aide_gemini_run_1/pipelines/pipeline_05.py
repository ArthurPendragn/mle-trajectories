"""
LightGBM gradient boosting pipeline for Spaceship Titanic passenger transport prediction.
Features are engineered by extracting cabin deck and side from Cabin, computing total luxury amenity expenditure across all five facilities, and adding a zero-spending binary indicator.
High-cardinality identifiers (PassengerId, Name, Cabin) are removed, and categoricals are one-hot encoded via TableVectorizer.
Three LightGBMClassifier configurations varying tree complexity (num_leaves) and learning rate are evaluated to explore leaf-wise gradient boosting as a distinct model family.
"""

from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
import skrub


def build():
    eval_dict = build_evaluation()
    X = eval_dict["X"]
    y = eval_dict["y"]
    scoring = eval_dict["scoring"]
    row_keys = eval_dict.get("row_keys")

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    total_spend = X[spend_cols].fillna(0.0).sum(axis=1)
    zero_spend = (total_spend == 0.0).astype(float)

    cabin_str = X["Cabin"].astype("string")
    deck = cabin_str.str.split("/").str[0]
    side = cabin_str.str.split("/").str[2]

    features = (
        X.assign(
            Deck=deck,
            Side=side,
            TotalSpend=total_spend,
            ZeroSpend=zero_spend,
        )
        .drop(columns=["PassengerId", "Name", "Cabin"], errors="ignore")
        .skb.set_name("features")
    )

    vec = skrub.TableVectorizer(
        cardinality_threshold=40,
        low_cardinality=OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        ),
    )
    features_vec = features.skb.apply(vec).skb.set_name("vectorizer")

    models = {
        "lgb_leaves31_lr005": LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        ),
        "lgb_leaves15_lr003": LGBMClassifier(
            num_leaves=15,
            learning_rate=0.03,
            n_estimators=400,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        ),
        "lgb_leaves63_lr003": LGBMClassifier(
            num_leaves=63,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        ),
    }

    model_choice = skrub.choose_from(models, name="model_variant")
    pred = features_vec.skb.apply(model_choice, y=y).skb.set_name("model")

    return {"pred": pred, "scoring": scoring, "row_keys": row_keys}