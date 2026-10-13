import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"


class EarthSubgroupAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analyzes feature separation on Earth zero-spending non-cryo passengers (n=302)
    and young dependents (Age <= 12, n=571) between Transported == True and False.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if "Transported" in df.columns:
            y = df["Transported"].astype(bool)
        else:
            y = pd.Series(True, index=df.index)

        amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        spend_sum = df[amenity_cols].fillna(0.0).sum(axis=1)

        cabin_str = df["Cabin"].fillna("Missing/Missing/Missing").astype(str)
        cabin_parts = cabin_str.str.split("/", expand=True)
        deck = cabin_parts[0].where(df["Cabin"].notna(), "Missing")
        side = cabin_parts[2].where(df["Cabin"].notna(), "Missing")
        cabin_num = pd.to_numeric(cabin_parts[1], errors="coerce").fillna(-1.0)

        group_ids = df["PassengerId"].astype(str).str.slice(0, 4)
        group_num = pd.to_numeric(
            df["PassengerId"].astype(str).str.slice(5), errors="coerce"
        ).fillna(1.0)
        group_size_map = group_ids.value_counts().to_dict()
        group_size = group_ids.map(group_size_map).fillna(1.0)

        surnames = df["Name"].fillna("Missing").astype(str).str.split().str[-1]
        family_size_map = (
            surnames[surnames != "Missing"].value_counts().to_dict()
        )
        family_size = surnames.map(family_size_map).fillna(1.0)

        cabins = df["Cabin"].fillna("Missing").astype(str)
        cabin_occ_map = cabins[cabins != "Missing"].value_counts().to_dict()
        cabin_occupants = cabins.map(cabin_occ_map).fillna(1.0)

        home_planet = df["HomePlanet"].fillna("Missing").astype(str)
        cryo_sleep = df["CryoSleep"].fillna("Missing").astype(str)
        destination = df["Destination"].fillna("Missing").astype(str)
        age = df["Age"].fillna(-1.0)

        # Filters
        is_earth = home_planet == "Earth"
        is_earth_zero_noncryo = is_earth & (cryo_sleep == "False") & (spend_sum == 0.0)
        is_earth_minor = is_earth & (age >= 0.0) & (age <= 12.0)

        records = []

        # Analyze Earth NonCryo ZeroSpend (n=302)
        df_zero = df[is_earth_zero_noncryo].copy()
        y_zero = y[is_earth_zero_noncryo]
        n_zero = len(df_zero)
        base_rate_zero = float(y_zero.mean()) if n_zero > 0 else 0.0

        for col_name, s in [
            ("Destination", destination[is_earth_zero_noncryo]),
            ("Deck", deck[is_earth_zero_noncryo]),
            ("Side", side[is_earth_zero_noncryo]),
            (
                "Deck_Side",
                (
                    deck[is_earth_zero_noncryo]
                    + "/"
                    + side[is_earth_zero_noncryo]
                ),
            ),
            (
                "GroupSize_bin",
                pd.cut(
                    group_size[is_earth_zero_noncryo],
                    bins=[0, 1, 2, 4, 10],
                    labels=["Solo", "Pair", "SmallGroup", "LargeGroup"],
                ).astype(str),
            ),
            (
                "CabinNum_bin",
                pd.cut(
                    cabin_num[is_earth_zero_noncryo],
                    bins=[-2, 0, 500, 1000, 2000],
                    labels=["None", "0-500", "500-1000", "1000+"],
                ).astype(str),
            ),
            (
                "CabinOccupants",
                cabin_occupants[is_earth_zero_noncryo].astype(str),
            ),
            ("GroupNum_pp", group_num[is_earth_zero_noncryo].astype(str)),
            (
                "FamilySize_bin",
                pd.cut(
                    family_size[is_earth_zero_noncryo],
                    bins=[0, 1, 3, 10],
                    labels=["Solo_Name", "Small_Family", "Large_Family"],
                ).astype(str),
            ),
        ]:
            vc = s.value_counts()
            for val, cnt in vc.items():
                if cnt >= 10:
                    pos_rate = float(y_zero[s == val].mean())
                    records.append(
                        {
                            "subgroup_target": "Earth_NonCryo_ZeroSpend",
                            "n_subgroup_total": n_zero,
                            "baseline_transported_rate": round(
                                base_rate_zero, 4
                            ),
                            "feature": col_name,
                            "feature_value": str(val),
                            "category_count": int(cnt),
                            "transported_count": int(y_zero[s == val].sum()),
                            "transported_rate": round(pos_rate, 4),
                            "rate_diff_from_base": round(
                                pos_rate - base_rate_zero, 4
                            ),
                        }
                    )

        # Analyze Earth Minors (Age <= 12, n=571)
        df_minor = df[is_earth_minor].copy()
        y_minor = y[is_earth_minor]
        n_minor = len(df_minor)
        base_rate_minor = float(y_minor.mean()) if n_minor > 0 else 0.0

        for col_name, s in [
            ("CryoSleep", cryo_sleep[is_earth_minor]),
            ("Destination", destination[is_earth_minor]),
            ("Deck", deck[is_earth_minor]),
            ("Side", side[is_earth_minor]),
            ("Deck_Side", deck[is_earth_minor] + "/" + side[is_earth_minor]),
            (
                "GroupSize_bin",
                pd.cut(
                    group_size[is_earth_minor],
                    bins=[0, 1, 2, 4, 10],
                    labels=["Solo", "Pair", "SmallGroup", "LargeGroup"],
                ).astype(str),
            ),
            (
                "CabinNum_bin",
                pd.cut(
                    cabin_num[is_earth_minor],
                    bins=[-2, 0, 500, 1000, 2000],
                    labels=["None", "0-500", "500-1000", "1000+"],
                ).astype(str),
            ),
            ("CabinOccupants", cabin_occupants[is_earth_minor].astype(str)),
            ("GroupNum_pp", group_num[is_earth_minor].astype(str)),
            (
                "FamilySize_bin",
                pd.cut(
                    family_size[is_earth_minor],
                    bins=[0, 1, 3, 10],
                    labels=["Solo_Name", "Small_Family", "Large_Family"],
                ).astype(str),
            ),
        ]:
            vc = s.value_counts()
            for val, cnt in vc.items():
                if cnt >= 15:
                    pos_rate = float(y_minor[s == val].mean())
                    records.append(
                        {
                            "subgroup_target": "Earth_Minors_under_13",
                            "n_subgroup_total": n_minor,
                            "baseline_transported_rate": round(
                                base_rate_minor, 4
                            ),
                            "feature": col_name,
                            "feature_value": str(val),
                            "category_count": int(cnt),
                            "transported_count": int(y_minor[s == val].sum()),
                            "transported_rate": round(pos_rate, 4),
                            "rate_diff_from_base": round(
                                pos_rate - base_rate_minor, 4
                            ),
                        }
                    )

        out_df = pd.DataFrame(records)
        return out_df.sort_values(
            ["subgroup_target", "rate_diff_from_base"], ascending=[True, False]
        ).reset_index(drop=True)


class GroupConcordanceAnalyzer(BaseEstimator, TransformerMixin):
    """
    Measures within-group transit concordance across multi-passenger groups (size >= 2),
    and computes correlations between group-level observable aggregates and individual Transported outcomes.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if "Transported" in df.columns:
            y = df["Transported"].astype(bool)
        else:
            y = pd.Series(True, index=df.index)

        amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        spend_sum = df[amenity_cols].fillna(0.0).sum(axis=1)

        group_ids = df["PassengerId"].astype(str).str.slice(0, 4)
        cabin_str = df["Cabin"].fillna("Missing/Missing/Missing").astype(str)
        cabin_parts = cabin_str.str.split("/", expand=True)
        deck = cabin_parts[0].where(df["Cabin"].notna(), "Missing")
        side = cabin_parts[2].where(df["Cabin"].notna(), "Missing")

        cryo_bool = df["CryoSleep"].fillna(False).astype(bool)
        is_minor = (df["Age"].fillna(999.0) <= 12.0).astype(float)

        df_calc = pd.DataFrame(
            {
                "group_id": group_ids,
                "transported": y,
                "cryo": cryo_bool.astype(float),
                "minor": is_minor,
                "spend": spend_sum,
                "deck": deck,
                "side": side,
            }
        )

        group_sizes = group_ids.value_counts()
        multi_groups = set(group_sizes[group_sizes >= 2].index)
        df_multi = df_calc[df_calc["group_id"].isin(multi_groups)].copy()

        # Group-level transit concordance
        group_target_stats = df_multi.groupby("group_id")["transported"].agg(
            ["count", "sum", "mean"]
        )
        total_multi_groups = len(group_target_stats)
        all_true = (
            group_target_stats["sum"] == group_target_stats["count"]
        ).sum()
        all_false = (group_target_stats["sum"] == 0).sum()
        concordant_groups = all_true + all_false
        mixed_groups = total_multi_groups - concordant_groups

        concordance_summary = pd.DataFrame(
            [
                {
                    "metric": "Total Multi-Passenger Groups (Size >= 2)",
                    "count": total_multi_groups,
                    "percentage": 100.0,
                },
                {
                    "metric": "Fully Concordant Groups (All Same Outcome)",
                    "count": int(concordant_groups),
                    "percentage": round(
                        100.0 * concordant_groups / total_multi_groups, 2
                    ),
                },
                {
                    "metric": "All Members Transported (All True)",
                    "count": int(all_true),
                    "percentage": round(
                        100.0 * all_true / total_multi_groups, 2
                    ),
                },
                {
                    "metric": "All Members Not Transported (All False)",
                    "count": int(all_false),
                    "percentage": round(
                        100.0 * all_false / total_multi_groups, 2
                    ),
                },
                {
                    "metric": "Mixed Outcomes Groups",
                    "count": int(mixed_groups),
                    "percentage": round(
                        100.0 * mixed_groups / total_multi_groups, 2
                    ),
                },
            ]
        )

        # Correlation of group observable aggregates with individual Transported status
        group_aggs = df_multi.groupby("group_id").agg(
            group_size=("transported", "count"),
            group_cryo_ratio=("cryo", "mean"),
            group_any_cryo=("cryo", lambda s: float((s > 0).any())),
            group_all_cryo=("cryo", lambda s: float((s == 1.0).all())),
            group_minor_count=("minor", "sum"),
            group_any_minor=("minor", lambda s: float((s > 0).any())),
            group_total_spend=("spend", "sum"),
            group_mean_spend=("spend", "mean"),
            group_zero_spend_all=(
                "spend",
                lambda s: float((s == 0.0).all()),
            ),
            group_deck_nunique=("deck", "nunique"),
            group_side_nunique=("side", "nunique"),
        )

        df_multi_joined = df_multi.join(group_aggs, on="group_id")
        corr_series = df_multi_joined.corr(numeric_only=True)["transported"]

        corr_rows = []
        for feature_name, corr_val in corr_series.items():
            if feature_name != "transported":
                corr_rows.append(
                    {
                        "group_feature": feature_name,
                        "pearson_corr_with_transported": round(
                            float(corr_val), 4
                        ),
                        "abs_corr": round(abs(float(corr_val)), 4),
                    }
                )

        corr_df = (
            pd.DataFrame(corr_rows)
            .sort_values("abs_corr", ascending=False)
            .reset_index(drop=True)
        )

        # Combine summary and correlations for concise reporting
        concordance_summary["section"] = "concordance_rate"
        concordance_summary["group_feature"] = concordance_summary["metric"]
        concordance_summary["stat_value"] = concordance_summary["count"].astype(
            str
        ) + " (" + concordance_summary["percentage"].astype(str) + "%)"

        corr_df["section"] = "feature_correlation"
        corr_df["stat_value"] = corr_df[
            "pearson_corr_with_transported"
        ].astype(str)

        combined = pd.concat(
            [
                concordance_summary[["section", "group_feature", "stat_value"]],
                corr_df[
                    ["section", "group_feature", "stat_value"]
                ].head(10),
            ],
            axis=0,
            ignore_index=True,
        )
        return combined


class Fold4DeficitAnalyzer(BaseEstimator, TransformerMixin):
    """
    Fits the top parent consensus ensemble (LightGBM + CatBoost + HistGB) across 5 folds,
    computes out-of-fold predictions, and segments prediction error patterns on fold 4 vs folds 0-3
    to identify the root causes of fold 4's deficit and potential feature corrections.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        y = df["Transported"].astype(bool)

        # PartyFeatureEngineer logic
        group_ids = df["PassengerId"].astype(str).str.slice(0, 4)
        cabins = df["Cabin"].fillna("Missing").astype(str)
        surnames = df["Name"].fillna("Missing").astype(str).str.split().str[-1]
        amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        spend_sum = df[amenity_cols].fillna(0.0).sum(axis=1)

        cabin_str = df["Cabin"].astype(str)
        cabin_parts = cabin_str.str.split("/", expand=True)
        deck = cabin_parts[0].where(df["Cabin"].notna(), "Missing")
        side = cabin_parts[2].where(df["Cabin"].notna(), "Missing")
        cabin_num = pd.to_numeric(cabin_parts[1], errors="coerce").fillna(-1.0)
        cabin_region = (cabin_num // 300.0).astype(float)

        group_size_map = group_ids.value_counts().to_dict()
        group_size = group_ids.map(group_size_map).fillna(1.0).astype(float)
        is_alone = (group_size == 1.0).astype(float)
        group_num = pd.to_numeric(
            df["PassengerId"].astype(str).str.slice(5), errors="coerce"
        ).fillna(1.0)

        df_stats = pd.DataFrame({"group_id": group_ids, "spend": spend_sum})
        group_agg = df_stats.groupby("group_id")["spend"].agg(["sum", "mean"])
        group_sum_spend_map = group_agg["sum"].to_dict()
        group_mean_spend_map = group_agg["mean"].to_dict()
        group_total_spend = (
            group_ids.map(group_sum_spend_map).fillna(spend_sum).astype(float)
        )
        group_mean_spend = (
            group_ids.map(group_mean_spend_map).fillna(spend_sum).astype(float)
        )

        cabin_occupants_map = cabins.value_counts().to_dict()
        cabin_occupants = (
            cabins.map(cabin_occupants_map).fillna(1.0).astype(float)
        )
        surname_freq_map = surnames.value_counts().to_dict()
        family_size = surnames.map(surname_freq_map).fillna(1.0).astype(float)

        # Missing value propagation
        hp_valid = df[df["HomePlanet"].notna() & (df["HomePlanet"] != "Missing")]
        group_to_hp = hp_valid.groupby(
            hp_valid["PassengerId"].astype(str).str.slice(0, 4)
        )["HomePlanet"].first().to_dict()
        valid_surnames = surnames[
            (surnames != "Missing")
            & df["HomePlanet"].notna()
            & (df["HomePlanet"] != "Missing")
        ]
        valid_hp = df.loc[valid_surnames.index, "HomePlanet"]
        df_surname_hp = pd.DataFrame(
            {"surname": valid_surnames, "HomePlanet": valid_hp}
        )
        surname_to_hp = (
            df_surname_hp.groupby("surname")["HomePlanet"].first().to_dict()
        )

        hp_series = df["HomePlanet"].copy()
        hp_series = hp_series.fillna(group_ids.map(group_to_hp))
        hp_series = hp_series.fillna(surnames.map(surname_to_hp))
        home_planet = hp_series.fillna("Missing").astype(str)

        cryo_series = df["CryoSleep"].copy()
        cryo_series = cryo_series.mask(
            cryo_series.isna() & (spend_sum > 0.0), False
        )
        cryo_sleep = cryo_series.fillna("Missing").astype(str)

        age_val = df["Age"]
        is_child = (age_val.fillna(999.0) <= 12.0).astype(float)
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

        destination = df["Destination"].fillna("Missing").astype(str)
        vip = df["VIP"].fillna("Missing").astype(str)

        X_processed = pd.DataFrame(
            {
                "HomePlanet": home_planet,
                "CryoSleep": cryo_sleep,
                "Destination": destination,
                "VIP": vip,
                "Deck": deck,
                "Side": side,
                "Age": age_val,
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
                "DeckSide": deck + "/" + side,
                "DeckGStarboardHotspot": (
                    (deck == "G")
                    & (side == "S")
                    & (cabin_num >= 750.0)
                    & (cabin_num <= 1200.0)
                ).astype(float),
                "HomePlanet_Cryo": home_planet + "_" + cryo_sleep,
                "IsEarthChild": (
                    (home_planet == "Earth") & (age_val.fillna(999.0) <= 12.0)
                ).astype(float),
                "Earth_NonCryo_ZeroSpend": (
                    (home_planet == "Earth")
                    & (cryo_sleep == "False")
                    & (spend_sum == 0.0)
                ).astype(float),
                "HomePlanet_Deck": home_planet + "_" + deck,
            },
            index=df.index,
        )

        base_cat_cols = [
            "HomePlanet",
            "CryoSleep",
            "Destination",
            "VIP",
            "Deck",
            "Side",
            "DeckSide",
            "HomePlanet_Cryo",
            "HomePlanet_Deck",
        ]
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
            "DeckGStarboardHotspot",
            "IsEarthChild",
            "Earth_NonCryo_ZeroSpend",
        ]

        cat_transformer = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        num_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, base_cat_cols),
                ("num", num_transformer, base_num_cols),
            ]
        )

        # 5-fold out-of-fold prediction generation with top ensemble
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(df))
        oof_folds = np.zeros(len(df), dtype=int)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_processed, y)):
            X_tr = X_processed.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_va = X_processed.iloc[val_idx]

            X_tr_trans = preprocessor.fit_transform(X_tr)
            X_va_trans = preprocessor.transform(X_va)

            lgb = LGBMClassifier(
                learning_rate=0.05,
                num_leaves=15,
                n_estimators=300,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            ).fit(X_tr_trans, y_tr)
            cat = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=5,
                random_seed=42,
                thread_count=4,
                verbose=0,
                allow_writing_files=False,
            ).fit(X_tr_trans, y_tr)
            hgb = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                random_state=42,
            ).fit(X_tr_trans, y_tr)

            p_lgb = lgb.predict_proba(X_va_trans)
            p_cat = cat.predict_proba(X_va_trans)
            p_hgb = hgb.predict_proba(X_va_trans)

            # Consensus power p=2.0 weights [2, 3, 1]
            p_pow = 2.0 * np.power(p_lgb, 2.0) + 3.0 * np.power(
                p_cat, 2.0
            ) + 1.0 * np.power(p_hgb, 2.0)
            p_norm = p_pow / np.sum(p_pow, axis=1, keepdims=True)

            oof_probs[val_idx] = p_norm[:, 1]
            oof_folds[val_idx] = fold_idx

        oof_preds = oof_probs >= 0.50
        errors = (oof_preds != y).astype(int)

        is_fold4 = oof_folds == 4
        is_folds03 = oof_folds < 4

        # Subgroups definition
        is_earth = home_planet == "Earth"
        is_earth_zero_noncryo = is_earth & (cryo_sleep == "False") & (spend_sum == 0.0)
        is_earth_minor = is_earth & (age_val.fillna(999.0) <= 12.0)
        is_deck_g = deck == "G"
        is_deck_g_s = (deck == "G") & (side == "S")
        is_deck_g_p = (deck == "G") & (side == "P")
        is_cryo_true = cryo_sleep == "True"
        is_earth_cryo = is_earth & is_cryo_true

        subgroups = {
            "All_Passengers": np.ones(len(df), dtype=bool),
            "Earth_Passengers": is_earth,
            "Earth_NonCryo_ZeroSpend": is_earth_zero_noncryo,
            "Earth_Minors_<=12": is_earth_minor,
            "Earth_CryoSleep": is_earth_cryo,
            "Deck_G_All": is_deck_g,
            "Deck_G_Port": is_deck_g_p,
            "Deck_G_Starboard": is_deck_g_s,
            "Multi_Group_Passengers": group_size > 1.0,
            "Solo_Passengers": is_alone == 1.0,
        }

        records = []
        for name, mask in subgroups.items():
            cnt_f4 = np.sum(mask & is_fold4)
            err_f4 = np.sum(errors[mask & is_fold4])
            rate_err_f4 = float(err_f4 / cnt_f4) if cnt_f4 > 0 else 0.0
            acc_f4 = 1.0 - rate_err_f4

            cnt_f03 = np.sum(mask & is_folds03)
            err_f03 = np.sum(errors[mask & is_folds03])
            rate_err_f03 = float(err_f03 / cnt_f03) if cnt_f03 > 0 else 0.0
            acc_f03 = 1.0 - rate_err_f03

            acc_deficit = acc_f03 - acc_f4
            excess_errors = int(round((rate_err_f4 - rate_err_f03) * cnt_f4))

            records.append(
                {
                    "subgroup": name,
                    "count_fold4": int(cnt_f4),
                    "accuracy_fold4": round(acc_f4, 4),
                    "count_folds03": int(cnt_f03),
                    "accuracy_folds03": round(acc_f03, 4),
                    "accuracy_deficit_f03_minus_f4": round(acc_deficit, 4),
                    "excess_errors_on_fold4": excess_errors,
                }
            )

        return pd.DataFrame(records).sort_values(
            "excess_errors_on_fold4", ascending=False
        ).reset_index(drop=True)


def build():
    train_op = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)

    earth_subgroups = train_op.skb.apply(
        EarthSubgroupAnalyzer()
    ).skb.set_name("earth_subgroup_analysis")
    group_concordance = train_op.skb.apply(
        GroupConcordanceAnalyzer()
    ).skb.set_name("group_concordance_analysis")
    fold4_deficit = train_op.skb.apply(
        Fold4DeficitAnalyzer()
    ).skb.set_name("fold4_deficit_analysis")

    return {
        "earth_subgroups": earth_subgroups,
        "group_concordance": group_concordance,
        "fold4_deficit": fold4_deficit,
    }