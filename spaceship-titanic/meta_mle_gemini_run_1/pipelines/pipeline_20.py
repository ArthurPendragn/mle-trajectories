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


class ContextualFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extracts travel party context, spatial ship coordinates, spend distributions,
    and domain-informed interaction features (Earth transit dynamics, Deck G hotspots,
    and spend-based CryoSleep resolution).
    """

    def __init__(self, mode="parent_baseline"):
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

        cabin_str = df["Cabin"].astype(str)
        cabin_parts = cabin_str.str.split("/", expand=True)
        deck = cabin_parts[0].where(df["Cabin"].notna(), "Missing")
        side = cabin_parts[2].where(df["Cabin"].notna(), "Missing")
        cabin_num = pd.to_numeric(cabin_parts[1], errors="coerce").fillna(-1.0)
        cabin_region = (cabin_num // 300.0).astype(float)

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

        home_planet = df["HomePlanet"].fillna("Missing").astype(str)
        cryo_sleep = df["CryoSleep"].fillna("Missing").astype(str)

        # Spend-based CryoSleep deduction: CryoSleep strictly precludes amenity spending
        if self.mode in ["earth_transit", "combined_all"]:
            cryo_deduced = cryo_sleep.copy()
            cryo_deduced[(cryo_sleep == "Missing") & (spend_sum > 0.0)] = "False"
            cryo_sleep = cryo_deduced

        destination = df["Destination"].fillna("Missing").astype(str)
        vip = df["VIP"].fillna("Missing").astype(str)

        out_dict = {
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

        # Spatial deck and side hotspot features
        if self.mode in ["spatial_deckside", "combined_all"]:
            deck_side = deck + "/" + side
            deck_g_starboard_hotspot = (
                (deck == "G")
                & (side == "S")
                & (cabin_num >= 750.0)
                & (cabin_num <= 1200.0)
            ).astype(float)
            out_dict["DeckSide"] = deck_side
            out_dict["DeckG_Starboard_Hotspot"] = deck_g_starboard_hotspot

        # Earth-specific transit dynamics features
        if self.mode in ["earth_transit", "combined_all"]:
            earth_cryo = ((home_planet == "Earth") & (cryo_sleep == "True")).astype(
                float
            )
            earth_minor = ((home_planet == "Earth") & (is_child == 1.0)).astype(
                float
            )
            earth_zero_spend = (
                (home_planet == "Earth")
                & (zero_spend == 1.0)
                & (cryo_sleep != "True")
            ).astype(float)
            earth_deck_g = ((home_planet == "Earth") & (deck == "G")).astype(float)
            earth_pso = (
                (home_planet == "Earth") & (destination == "PSO J318.5-22")
            ).astype(float)

            out_dict["Earth_Cryo"] = earth_cryo
            out_dict["Earth_Minor"] = earth_minor
            out_dict["Earth_ZeroSpend"] = earth_zero_spend
            out_dict["Earth_Deck_G"] = earth_deck_g
            out_dict["Earth_PSO"] = earth_pso

        return pd.DataFrame(out_dict, index=df.index)


def make_preprocessor(cat_cols, num_cols):
    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]
    features = X.skb.set_name("raw_features")

    base_cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    base_num_cols = [
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

    # Preprocessors for each representation
    prep_baseline = make_preprocessor(base_cat_cols, base_num_cols)

    prep_spatial = make_preprocessor(
        base_cat_cols + ["DeckSide"],
        base_num_cols + ["DeckG_Starboard_Hotspot"],
    )

    prep_earth = make_preprocessor(
        base_cat_cols,
        base_num_cols
        + ["Earth_Cryo", "Earth_Minor", "Earth_ZeroSpend", "Earth_Deck_G", "Earth_PSO"],
    )

    prep_combined = make_preprocessor(
        base_cat_cols + ["DeckSide"],
        base_num_cols
        + [
            "DeckG_Starboard_Hotspot",
            "Earth_Cryo",
            "Earth_Minor",
            "Earth_ZeroSpend",
            "Earth_Deck_G",
            "Earth_PSO",
        ],
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
    ensemble = VotingClassifier(
        estimators=[("lgbm", lgb_clf), ("cat", cat_clf)],
        voting="soft",
        weights=[1, 2],
    )

    pipeline_variants = {
        "parent_baseline": Pipeline(
            [
                ("fe", ContextualFeatureEngineer(mode="parent_baseline")),
                ("prep", prep_baseline),
                ("model", ensemble),
            ]
        ),
        "spatial_deckside": Pipeline(
            [
                ("fe", ContextualFeatureEngineer(mode="spatial_deckside")),
                ("prep", prep_spatial),
                ("model", ensemble),
            ]
        ),
        "earth_transit": Pipeline(
            [
                ("fe", ContextualFeatureEngineer(mode="earth_transit")),
                ("prep", prep_earth),
                ("model", ensemble),
            ]
        ),
        "combined_all": Pipeline(
            [
                ("fe", ContextualFeatureEngineer(mode="combined_all")),
                ("prep", prep_combined),
                ("model", ensemble),
            ]
        ),
    }

    chosen_pipeline = skrub.choose_from(
        pipeline_variants, name="feature_representation"
    )
    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name("model")

    return {"pred": pred, "scoring": setup["scoring"]}