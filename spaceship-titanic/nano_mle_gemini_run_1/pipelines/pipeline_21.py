import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub


class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fits party, surname, and cabin context statistics on training rows.
    In propagation modes, resolves deterministic transit ambiguities (deducing
    CryoSleep=False when amenity spend > 0 and propagating HomePlanet/Side
    across passenger groups) and generates targeted subpopulation features.
    """

    def __init__(self, mode="full_calibrated_propagation"):
        self.mode = mode

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        group_ids = df["PassengerId"].astype(str).str.slice(0, 4)
        cabins = df["Cabin"].fillna("Missing").astype(str)
        surnames = df["Name"].fillna("Missing").astype(str).str.split().str[-1]
        amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        total_spends = df[amenity_cols].fillna(0.0).sum(axis=1)

        self.group_size_map_ = group_ids.value_counts().to_dict()
        self.cabin_occupants_map_ = cabins.value_counts().to_dict()
        self.surname_freq_map_ = surnames.value_counts().to_dict()

        df_stats = pd.DataFrame({"group_id": group_ids, "spend": total_spends})
        group_agg = df_stats.groupby("group_id")["spend"].agg(["sum", "mean"])
        self.group_sum_spend_map_ = group_agg["sum"].to_dict()
        self.group_mean_spend_map_ = group_agg["mean"].to_dict()

        self.group_side_map_ = {}
        self.group_hp_map_ = {}
        self.surname_hp_map_ = {}

        if self.mode == "full_calibrated_propagation":
            cabin_str = df["Cabin"].dropna().astype(str)
            cabin_split = cabin_str.str.split("/", expand=True)
            known_side = cabin_split[2]
            df_side = pd.DataFrame(
                {"group_id": group_ids.loc[known_side.index], "side": known_side}
            )
            self.group_side_map_ = (
                df_side.dropna().groupby("group_id")["side"].first().to_dict()
            )

            df_hp = pd.DataFrame(
                {
                    "group_id": group_ids,
                    "home_planet": df["HomePlanet"],
                    "surname": surnames,
                }
            )
            self.group_hp_map_ = (
                df_hp.dropna(subset=["home_planet"])
                .groupby("group_id")["home_planet"]
                .first()
                .to_dict()
            )
            self.surname_hp_map_ = (
                df_hp.dropna(subset=["home_planet"])
                .groupby("surname")["home_planet"]
                .first()
                .to_dict()
            )

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        spend_sum = df[amenity_cols].fillna(0.0).sum(axis=1)
        zero_spend = (spend_sum == 0.0).astype(float)
        luxury_spend = (
            df["RoomService"].fillna(0.0)
            + df["Spa"].fillna(0.0)
            + df["VRDeck"].fillna(0.0)
        )
        log_total_spend = np.log1p(spend_sum)
        log_luxury_spend = np.log1p(luxury_spend)
        luxury_ratio = luxury_spend / (spend_sum + 1.0)
        room_service_ratio = df["RoomService"].fillna(0.0) / (spend_sum + 1.0)
        spa_ratio = df["Spa"].fillna(0.0) / (spend_sum + 1.0)
        vr_ratio = df["VRDeck"].fillna(0.0) / (spend_sum + 1.0)

        group_ids = df["PassengerId"].astype(str).str.slice(0, 4)
        group_num = pd.to_numeric(
            df["PassengerId"].astype(str).str.slice(5), errors="coerce"
        ).fillna(1.0)
        cabins = df["Cabin"].fillna("Missing").astype(str)
        surnames = df["Name"].fillna("Missing").astype(str).str.split().str[-1]
        group_size = group_ids.map(self.group_size_map_).fillna(1.0).astype(float)
        is_alone = (group_size == 1.0).astype(float)
        group_total_spend = (
            group_ids.map(self.group_sum_spend_map_).fillna(spend_sum).astype(float)
        )
        group_mean_spend = (
            group_ids.map(self.group_mean_spend_map_).fillna(spend_sum).astype(float)
        )
        cabin_occupants = (
            cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        )
        family_size = surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)
        is_child = (df["Age"].fillna(999.0) <= 12.0).astype(float)

        cabin_str = df["Cabin"].astype(str)
        cabin_parts = cabin_str.str.split("/", expand=True)
        deck = cabin_parts[0].where(df["Cabin"].notna(), "Missing")
        side = cabin_parts[2].where(df["Cabin"].notna(), "Missing")
        cabin_num = pd.to_numeric(cabin_parts[1], errors="coerce").fillna(-1.0)
        cabin_region = (cabin_num // 300.0).astype(float)

        home_planet_raw = df["HomePlanet"]
        cryo_sleep_raw = df["CryoSleep"]

        if self.mode == "full_calibrated_propagation":
            cryo_deduced = cryo_sleep_raw.copy()
            cryo_deduced = cryo_deduced.mask(
                cryo_deduced.isna() & (spend_sum > 0), False
            )
            cryo_sleep = cryo_deduced.fillna("Missing").astype(str)

            hp_propagated = home_planet_raw.copy()
            hp_propagated = hp_propagated.fillna(group_ids.map(self.group_hp_map_))
            hp_propagated = hp_propagated.fillna(surnames.map(self.surname_hp_map_))
            home_planet = hp_propagated.fillna("Missing").astype(str)

            side_propagated = side.copy()
            side_from_group = group_ids.map(self.group_side_map_)
            side_propagated = (
                side_propagated.where(side_propagated != "Missing", side_from_group)
                .fillna("Missing")
                .astype(str)
            )
            side = side_propagated
        else:
            home_planet = home_planet_raw.fillna("Missing").astype(str)
            cryo_sleep = cryo_sleep_raw.fillna("Missing").astype(str)

        destination = df["Destination"].fillna("Missing").astype(str)
        vip = df["VIP"].fillna("Missing").astype(str)

        base_dict = {
            "HomePlanet": home_planet,
            "CryoSleep": cryo_sleep,
            "Destination": destination,
            "VIP": vip,
            "Deck": deck,
            "Side": side,
            "Age": df["Age"],
            "CabinNum": cabin_num,
            "CabinRegion": cabin_region,
            "CabinOccupants": cabin_occupants,
            "GroupNum": group_num,
            "GroupSize": group_size,
            "IsAlone": is_alone,
            "GroupTotalSpend": group_total_spend,
            "GroupMeanSpend": group_mean_spend,
            "FamilySize": family_size,
            "IsChild": is_child,
            "RoomService": df["RoomService"],
            "FoodCourt": df["FoodCourt"],
            "ShoppingMall": df["ShoppingMall"],
            "Spa": df["Spa"],
            "VRDeck": df["VRDeck"],
            "TotalSpend": spend_sum,
            "ZeroSpend": zero_spend,
            "LogTotalSpend": log_total_spend,
            "LuxurySpend": luxury_spend,
            "LogLuxurySpend": log_luxury_spend,
            "LuxuryRatio": luxury_ratio,
            "RoomServiceRatio": room_service_ratio,
            "SpaRatio": spa_ratio,
            "VRDeckRatio": vr_ratio,
        }

        if self.mode != "parent_baseline":
            deck_side = deck.astype(str) + "_" + side.astype(str)
            base_dict["DeckSide"] = deck_side

            deck_g_starboard_cluster = (
                (deck == "G")
                & (side == "S")
                & (cabin_num >= 750.0)
                & (cabin_num <= 1200.0)
            ).astype(float)
            base_dict["DeckG_Starboard_Cluster"] = deck_g_starboard_cluster

            is_earth = home_planet == "Earth"
            base_dict["EarthCryo"] = (is_earth & (cryo_sleep == "True")).astype(float)
            base_dict["EarthMinor"] = (is_earth & (is_child == 1.0)).astype(float)
            base_dict["EarthZeroSpendNonCryo"] = (
                is_earth & (cryo_sleep == "False") & (zero_spend == 1.0)
            ).astype(float)
            base_dict["EarthPSO"] = (
                is_earth & (destination == "PSO J318.5-22")
            ).astype(float)

        return pd.DataFrame(base_dict, index=df.index)


def make_pipeline(mode):
    party_fe = PartyFeatureEngineer(mode=mode)
    if mode == "parent_baseline":
        cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
        num_cols = [
            "Age",
            "CabinNum",
            "CabinRegion",
            "CabinOccupants",
            "GroupNum",
            "GroupSize",
            "IsAlone",
            "GroupTotalSpend",
            "GroupMeanSpend",
            "FamilySize",
            "IsChild",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
            "TotalSpend",
            "ZeroSpend",
            "LogTotalSpend",
            "LuxurySpend",
            "LogLuxurySpend",
            "LuxuryRatio",
            "RoomServiceRatio",
            "SpaRatio",
            "VRDeckRatio",
        ]
    else:
        cat_cols = [
            "HomePlanet",
            "CryoSleep",
            "Destination",
            "VIP",
            "Deck",
            "Side",
            "DeckSide",
        ]
        num_cols = [
            "Age",
            "CabinNum",
            "CabinRegion",
            "CabinOccupants",
            "GroupNum",
            "GroupSize",
            "IsAlone",
            "GroupTotalSpend",
            "GroupMeanSpend",
            "FamilySize",
            "IsChild",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
            "TotalSpend",
            "ZeroSpend",
            "LogTotalSpend",
            "LuxurySpend",
            "LogLuxurySpend",
            "LuxuryRatio",
            "RoomServiceRatio",
            "SpaRatio",
            "VRDeckRatio",
            "DeckG_Starboard_Cluster",
            "EarthCryo",
            "EarthMinor",
            "EarthZeroSpendNonCryo",
            "EarthPSO",
        ]

    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )

    lgb_clf = LGBMClassifier(
        learning_rate=0.05,
        num_leaves=15,
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    cat_clf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        random_seed=42,
        thread_count=4,
        verbose=0,
        allow_writing_files=False,
    )
    chosen_model = VotingClassifier(
        estimators=[("lgbm", lgb_clf), ("cat", cat_clf)],
        voting="soft",
        weights=[1, 2],
    )

    return Pipeline(
        [
            ("party_fe", party_fe),
            ("prep", preprocessor),
            ("model", chosen_model),
        ]
    )


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    features = X.skb.set_name("raw_features")

    pipeline_variants = {
        "parent_baseline": make_pipeline("parent_baseline"),
        "spatial_and_subpopulations": make_pipeline("spatial_and_subpopulations"),
        "full_calibrated_propagation": make_pipeline("full_calibrated_propagation"),
    }
    chosen_pipeline = skrub.choose_from(
        pipeline_variants, name="feature_representation"
    )

    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name("model")
    return {"pred": pred, "scoring": setup["scoring"]}