"""Regularized linear model baseline for Spaceship Titanic passenger transport prediction.
Features are engineered by extracting cabin deck and side and calculating total luxury amenity spend and zero-spend indicator.
Categorical features are one-hot encoded with TableVectorizer, missing numeric values are median-imputed, and all features are standardized.
A logistic regression classifier is evaluated across multiple regularization strengths to explore a linear model family distinct from gradient boosting."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skrub import TableVectorizer


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    total_spend = (
        X['RoomService'].fillna(0.0)
        + X['FoodCourt'].fillna(0.0)
        + X['ShoppingMall'].fillna(0.0)
        + X['Spa'].fillna(0.0)
        + X['VRDeck'].fillna(0.0)
    )
    zero_spend = (total_spend == 0.0).astype('float64')

    engineered = X.assign(
        Cabin_Deck=X['Cabin'].str.slice(0, 1),
        Cabin_Side=X['Cabin'].str.slice(-1),
        TotalSpend=total_spend,
        ZeroSpend=zero_spend,
    )
    features = engineered.drop(columns=['PassengerId', 'Name', 'Cabin']).skb.set_name(
        'tabular_features'
    )

    encoded = features.skb.apply(
        TableVectorizer(
            cardinality_threshold=40,
            low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            high_cardinality='drop',
        )
    ).skb.set_name('encoded_features')

    model_choice = skrub.choose_from(
        {
            'lr_c01': make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            ),
            'lr_c1': make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            ),
            'lr_c10': make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(C=10.0, max_iter=1000, random_state=42),
            ),
        },
        name='model_variant',
    )

    pred = encoded.skb.apply(model_choice, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup.get('row_keys'),
    }