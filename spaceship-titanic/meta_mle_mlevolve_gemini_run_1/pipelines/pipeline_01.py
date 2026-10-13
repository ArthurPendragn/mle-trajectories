import skrub
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    cabin_split = X['Cabin'].str.split('/', expand=True)
    deck = cabin_split[0].astype('string')
    cabin_num = cabin_split[1].astype('float64')
    side = cabin_split[2].astype('string')

    group_id = X['PassengerId'].str[:4].astype('string')
    group_num = X['PassengerId'].str[5:].astype('float64')

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)

    total_spend = room_service + food_court + shopping_mall + spa + vr_deck
    has_spend = (total_spend > 0.0).astype('float64')

    cryo_sleep = X['CryoSleep'].astype('string')
    vip = X['VIP'].astype('string')

    features = X.assign(
        Deck=deck,
        CabinNum=cabin_num,
        Side=side,
        GroupId=group_id,
        GroupNum=group_num,
        TotalSpend=total_spend,
        HasSpend=has_spend,
        CryoSleep=cryo_sleep,
        VIP=vip,
    ).drop(columns=['PassengerId', 'Cabin', 'Name']).skb.set_name('features')

    model = skrub.choose_from(
        {
            'lgb_lr005_300': make_pipeline(
                skrub.TableVectorizer(),
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                    n_jobs=8,
                ),
            ),
            'lgb_lr003_500': make_pipeline(
                skrub.TableVectorizer(),
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                    n_jobs=8,
                ),
            ),
        },
        name='estimator_variant',
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': scoring,
    }