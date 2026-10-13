import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"


class EarthAndSpatialDiagnostics(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_df = X.copy().reset_index(drop=True)
        y_series = y.copy().reset_index(drop=True).astype(bool)

        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        total_spend = X_df[spend_cols].fillna(0).sum(axis=1)
        is_zero_spend = (total_spend == 0).astype(float)
        luxury_spend = X_df[["RoomService", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
        communal_spend = X_df[["FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)

        cabin_split = X_df["Cabin"].fillna("//").str.split("/", expand=True)
        deck = cabin_split[0].replace("", np.nan)
        cabin_num = pd.to_numeric(cabin_split[1], errors="coerce")
        side = cabin_split[2].replace("", np.nan)

        group_id = X_df["PassengerId"].str.split("_").str[0]
        group_counts = group_id.value_counts()
        group_size = group_id.map(group_counts).astype(float)
        is_alone = (group_size == 1).astype(float)

        feat_df = pd.DataFrame(
            {
                "Age": pd.to_numeric(X_df["Age"], errors="coerce"),
                "RoomService": pd.to_numeric(X_df["RoomService"], errors="coerce"),
                "FoodCourt": pd.to_numeric(X_df["FoodCourt"], errors="coerce"),
                "ShoppingMall": pd.to_numeric(X_df["ShoppingMall"], errors="coerce"),
                "Spa": pd.to_numeric(X_df["Spa"], errors="coerce"),
                "VRDeck": pd.to_numeric(X_df["VRDeck"], errors="coerce"),
                "TotalSpend": total_spend,
                "IsZeroSpend": is_zero_spend,
                "LuxurySpend": luxury_spend,
                "CommunalSpend": communal_spend,
                "CabinNum": cabin_num,
                "GroupSize": group_size,
                "IsAlone": is_alone,
                "CryoSleep": X_df["CryoSleep"].astype(float),
                "VIP": X_df["VIP"].astype(float),
            }
        )

        for col in ["HomePlanet", "Destination"]:
            codes = pd.Categorical(X_df[col]).codes.astype(float)
            codes[codes == -1] = np.nan
            feat_df[col] = codes

        deck_codes = pd.Categorical(deck).codes.astype(float)
        deck_codes[deck_codes == -1] = np.nan
        feat_df["Deck"] = deck_codes

        side_codes = pd.Categorical(side).codes.astype(float)
        side_codes[side_codes == -1] = np.nan
        feat_df["Side"] = side_codes

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_prob = np.zeros(len(X_df))

        for train_idx, val_idx in cv.split(feat_df, y_series):
            X_tr, y_tr = feat_df.iloc[train_idx], y_series.iloc[train_idx]
            X_va = feat_df.iloc[val_idx]

            lgb = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                random_state=42,
                verbose=-1,
                n_jobs=4,
            )
            lgb.fit(X_tr, y_tr)
            p_lgb = lgb.predict_proba(X_va)[:, 1]

            cat = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=5,
                random_seed=42,
                verbose=0,
                thread_count=4,
            )
            cat.fit(X_tr, y_tr)
            p_cat = cat.predict_proba(X_va)[:, 1]

            oof_prob[val_idx] = (1.0 * p_lgb + 2.0 * p_cat) / 3.0

        oof_pred = oof_prob >= 0.5
        is_error = oof_pred != y_series

        df_eval = X_df.copy()
        df_eval["y_true"] = y_series
        df_eval["oof_prob"] = oof_prob
        df_eval["oof_pred"] = oof_pred
        df_eval["is_error"] = is_error
        df_eval["Deck"] = deck
        df_eval["CabinNum"] = cabin_num
        df_eval["Side"] = side
        df_eval["TotalSpend"] = total_spend
        df_eval["GroupSize"] = group_size

        hp_records = []

        def add_hp_row(cat, val, sub):
            if len(sub) == 0:
                return
            n = len(sub)
            t_cnt = int(sub["y_true"].sum())
            t_rate = float(t_cnt / n)
            p_rate = float(sub["oof_pred"].mean())
            acc = float((sub["y_true"] == sub["oof_pred"]).mean())
            err = 1.0 - acc
            fp = int((sub["oof_pred"] & (~sub["y_true"])).sum())
            fn = int(((~sub["oof_pred"]) & sub["y_true"]).sum())
            mean_prob = float(sub["oof_prob"].mean())
            hp_records.append(
                {
                    "segment_category": cat,
                    "segment_value": str(val),
                    "passenger_count": n,
                    "transported_count": t_cnt,
                    "transported_rate": round(t_rate, 4),
                    "predicted_pos_rate": round(p_rate, 4),
                    "parent_accuracy": round(acc, 4),
                    "parent_error_rate": round(err, 4),
                    "false_positives": fp,
                    "false_negatives": fn,
                    "mean_pred_prob": round(mean_prob, 4),
                }
            )

        add_hp_row("All_HomePlanet", "All_Passengers", df_eval)
        for hp in ["Earth", "Europa", "Mars"]:
            add_hp_row(
                "All_HomePlanet",
                hp,
                df_eval[df_eval["HomePlanet"].fillna("Missing") == hp],
            )
        add_hp_row(
            "All_HomePlanet", "Missing", df_eval[df_eval["HomePlanet"].isna()]
        )

        df_earth = df_eval[df_eval["HomePlanet"] == "Earth"].copy()

        for dest in ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22", "Missing"]:
            sub = (
                df_earth[df_earth["Destination"].isna()]
                if dest == "Missing"
                else df_earth[df_earth["Destination"] == dest]
            )
            add_hp_row("Earth_by_Destination", dest, sub)

        for d in ["G", "F", "E", "Other_or_Missing"]:
            if d == "Other_or_Missing":
                sub = df_earth[~df_earth["Deck"].isin(["G", "F", "E"])]
            else:
                sub = df_earth[df_earth["Deck"] == d]
            add_hp_row("Earth_by_Deck", d, sub)

        for s in ["P", "S", "Missing"]:
            sub = (
                df_earth[df_earth["Side"].isna()]
                if s == "Missing"
                else df_earth[df_earth["Side"] == s]
            )
            add_hp_row("Earth_by_Side", s, sub)

        for d in ["G", "F", "E"]:
            for s in ["P", "S"]:
                sub = df_earth[(df_earth["Deck"] == d) & (df_earth["Side"] == s)]
                add_hp_row("Earth_by_Deck_Side", f"{d}/{s}", sub)

        df_hp_breakdown = pd.DataFrame(hp_records)

        earth_sub_records = []

        def add_earth_row(dim, label, sub):
            if len(sub) == 0:
                return
            n = len(sub)
            t_cnt = int(sub["y_true"].sum())
            t_rate = float(t_cnt / n)
            p_rate = float(sub["oof_pred"].mean())
            acc = float((sub["y_true"] == sub["oof_pred"]).mean())
            err = 1.0 - acc
            fp = int((sub["oof_pred"] & (~sub["y_true"])).sum())
            fn = int(((~sub["oof_pred"]) & sub["y_true"]).sum())
            mean_prob = float(sub["oof_prob"].mean())
            earth_sub_records.append(
                {
                    "dimension": dim,
                    "subgroup": label,
                    "passenger_count": n,
                    "transported_count": t_cnt,
                    "transported_rate": round(t_rate, 4),
                    "predicted_pos_rate": round(p_rate, 4),
                    "parent_accuracy": round(acc, 4),
                    "parent_error_rate": round(err, 4),
                    "false_positives": fp,
                    "false_negatives": fn,
                    "mean_pred_prob": round(mean_prob, 4),
                }
            )

        add_earth_row("All_Earth", "Earth_Overall", df_earth)

        add_earth_row(
            "Age", "Minors_under_13", df_earth[df_earth["Age"] < 13]
        )
        add_earth_row(
            "Age", "Teens_13_to_17", df_earth[(df_earth["Age"] >= 13) & (df_earth["Age"] < 18)]
        )
        add_earth_row(
            "Age", "YoungAdults_18_to_34", df_earth[(df_earth["Age"] >= 18) & (df_earth["Age"] < 35)]
        )
        add_earth_row(
            "Age", "MiddleAge_35_to_59", df_earth[(df_earth["Age"] >= 35) & (df_earth["Age"] < 60)]
        )
        add_earth_row(
            "Age", "Seniors_60_plus", df_earth[df_earth["Age"] >= 60]
        )
        add_earth_row("Age", "Age_Missing", df_earth[df_earth["Age"].isna()])

        add_earth_row(
            "Key_Comparison", "Earth_Minors_<13", df_earth[df_earth["Age"] < 13]
        )
        add_earth_row(
            "Key_Comparison", "Earth_Adults_>=18", df_earth[df_earth["Age"] >= 18]
        )

        add_earth_row(
            "Cryo_and_Spend", "Cryo_True (Spend=0)", df_earth[df_earth["CryoSleep"] == True]
        )
        add_earth_row(
            "Cryo_and_Spend",
            "NonCryo_ZeroSpend",
            df_earth[(df_earth["CryoSleep"] == False) & (df_earth["TotalSpend"] == 0)],
        )
        add_earth_row(
            "Cryo_and_Spend",
            "NonCryo_PositiveSpend",
            df_earth[(df_earth["CryoSleep"] == False) & (df_earth["TotalSpend"] > 0)],
        )
        add_earth_row(
            "Cryo_and_Spend",
            "CryoNA_ZeroSpend",
            df_earth[df_earth["CryoSleep"].isna() & (df_earth["TotalSpend"] == 0)],
        )
        add_earth_row(
            "Cryo_and_Spend",
            "CryoNA_PositiveSpend",
            df_earth[df_earth["CryoSleep"].isna() & (df_earth["TotalSpend"] > 0)],
        )

        add_earth_row(
            "Key_Comparison",
            "Earth_NonCryo_ZeroSpend",
            df_earth[(df_earth["CryoSleep"] == False) & (df_earth["TotalSpend"] == 0)],
        )
        add_earth_row(
            "Key_Comparison",
            "Earth_NonCryo_PositiveSpend",
            df_earth[(df_earth["CryoSleep"] == False) & (df_earth["TotalSpend"] > 0)],
        )

        add_earth_row(
            "Spend_Tier",
            "Spend_1_to_500",
            df_earth[(df_earth["TotalSpend"] > 0) & (df_earth["TotalSpend"] <= 500)],
        )
        add_earth_row(
            "Spend_Tier",
            "Spend_501_to_1500",
            df_earth[(df_earth["TotalSpend"] > 500) & (df_earth["TotalSpend"] <= 1500)],
        )
        add_earth_row(
            "Spend_Tier", "Spend_gt_1500", df_earth[df_earth["TotalSpend"] > 1500]
        )

        add_earth_row(
            "Group_Size", "Solo (Size=1)", df_earth[df_earth["GroupSize"] == 1]
        )
        add_earth_row(
            "Group_Size", "Pair (Size=2)", df_earth[df_earth["GroupSize"] == 2]
        )
        add_earth_row(
            "Group_Size",
            "Small_Group (Size 3-4)",
            df_earth[df_earth["GroupSize"].isin([3, 4])],
        )
        add_earth_row(
            "Group_Size",
            "Large_Group (Size >= 5)",
            df_earth[df_earth["GroupSize"] >= 5],
        )

        df_earth_subgroups = pd.DataFrame(earth_sub_records)

        df_deck_g = df_eval[df_eval["Deck"] == "G"].copy()
        g_records = []

        g_bins = [
            (0, 150),
            (150, 300),
            (300, 450),
            (450, 600),
            (600, 750),
            (750, 900),
            (900, 1050),
            (1050, 1200),
            (1200, 1350),
            (1350, 1600),
        ]

        for b_min, b_max in g_bins:
            sub = df_deck_g[
                (df_deck_g["CabinNum"] >= b_min) & (df_deck_g["CabinNum"] < b_max)
            ]
            if len(sub) == 0:
                continue
            n = len(sub)
            t_rate = float(sub["y_true"].mean())
            sub_p = sub[sub["Side"] == "P"]
            sub_s = sub[sub["Side"] == "S"]
            p_rate = float(sub_p["y_true"].mean()) if len(sub_p) > 0 else np.nan
            s_rate = float(sub_s["y_true"].mean()) if len(sub_s) > 0 else np.nan
            acc = float((sub["y_true"] == sub["oof_pred"]).mean())
            err = 1.0 - acc
            g_records.append(
                {
                    "bin_label": f"G_{b_min:04d}_to_{b_max:04d}",
                    "cabin_min": float(sub["CabinNum"].min()),
                    "cabin_median": float(sub["CabinNum"].median()),
                    "cabin_max": float(sub["CabinNum"].max()),
                    "passenger_count": n,
                    "overall_transported_rate": round(t_rate, 4),
                    "port_count": len(sub_p),
                    "port_transported_rate": (
                        round(p_rate, 4) if not np.isnan(p_rate) else np.nan
                    ),
                    "starboard_count": len(sub_s),
                    "starboard_transported_rate": (
                        round(s_rate, 4) if not np.isnan(s_rate) else np.nan
                    ),
                    "parent_accuracy": round(acc, 4),
                    "parent_error_rate": round(err, 4),
                    "mean_pred_prob": round(float(sub["oof_prob"].mean()), 4),
                }
            )

        g_all_p = df_deck_g[df_deck_g["Side"] == "P"]
        g_all_s = df_deck_g[df_deck_g["Side"] == "S"]

        g_records.append(
            {
                "bin_label": "DECK_G_PORT_OVERALL",
                "cabin_min": float(g_all_p["CabinNum"].min()),
                "cabin_median": float(g_all_p["CabinNum"].median()),
                "cabin_max": float(g_all_p["CabinNum"].max()),
                "passenger_count": len(g_all_p),
                "overall_transported_rate": round(
                    float(g_all_p["y_true"].mean()), 4
                ),
                "port_count": len(g_all_p),
                "port_transported_rate": round(float(g_all_p["y_true"].mean()), 4),
                "starboard_count": 0,
                "starboard_transported_rate": np.nan,
                "parent_accuracy": round(
                    float((g_all_p["y_true"] == g_all_p["oof_pred"]).mean()), 4
                ),
                "parent_error_rate": round(
                    1.0
                    - float((g_all_p["y_true"] == g_all_p["oof_pred"]).mean()),
                    4,
                ),
                "mean_pred_prob": round(float(g_all_p["oof_prob"].mean()), 4),
            }
        )

        g_records.append(
            {
                "bin_label": "DECK_G_STARBOARD_OVERALL",
                "cabin_min": float(g_all_s["CabinNum"].min()),
                "cabin_median": float(g_all_s["CabinNum"].median()),
                "cabin_max": float(g_all_s["CabinNum"].max()),
                "passenger_count": len(g_all_s),
                "overall_transported_rate": round(
                    float(g_all_s["y_true"].mean()), 4
                ),
                "port_count": 0,
                "port_transported_rate": np.nan,
                "starboard_count": len(g_all_s),
                "starboard_transported_rate": round(
                    float(g_all_s["y_true"].mean()), 4
                ),
                "parent_accuracy": round(
                    float((g_all_s["y_true"] == g_all_s["oof_pred"]).mean()), 4
                ),
                "parent_error_rate": round(
                    1.0
                    - float((g_all_s["y_true"] == g_all_s["oof_pred"]).mean()),
                    4,
                ),
                "mean_pred_prob": round(float(g_all_s["oof_prob"].mean()), 4),
            }
        )

        corr_overall = float(
            df_deck_g["CabinNum"].corr(df_deck_g["y_true"].astype(float))
        )
        corr_port = float(
            g_all_p["CabinNum"].corr(g_all_p["y_true"].astype(float))
        )
        corr_starboard = float(
            g_all_s["CabinNum"].corr(g_all_s["y_true"].astype(float))
        )

        g_records.append(
            {
                "bin_label": "DECK_G_CORRELATIONS",
                "cabin_min": corr_overall,
                "cabin_median": corr_port,
                "cabin_max": corr_starboard,
                "passenger_count": len(df_deck_g),
                "overall_transported_rate": round(
                    float(df_deck_g["y_true"].mean()), 4
                ),
                "port_count": len(g_all_p),
                "port_transported_rate": round(corr_port, 4),
                "starboard_count": len(g_all_s),
                "starboard_transported_rate": round(corr_starboard, 4),
                "parent_accuracy": round(
                    float((df_deck_g["y_true"] == df_deck_g["oof_pred"]).mean()),
                    4,
                ),
                "parent_error_rate": round(
                    1.0
                    - float(
                        (df_deck_g["y_true"] == df_deck_g["oof_pred"]).mean()
                    ),
                    4,
                ),
                "mean_pred_prob": round(float(df_deck_g["oof_prob"].mean()), 4),
            }
        )

        df_deck_g_spatial = pd.DataFrame(g_records)

        self.results_ = {
            "homeplanet_breakdown": df_hp_breakdown,
            "earth_subgroups": df_earth_subgroups,
            "deck_g_spatial": df_deck_g_spatial,
        }
        return self

    def transform(self, X):
        return self.results_


def build():
    df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(bool)

    analyzer = EarthAndSpatialDiagnostics()
    diag = X.skb.apply(analyzer, y=y, no_wrap=True)

    return {
        "homeplanet_breakdown": diag["homeplanet_breakdown"],
        "earth_subgroups": diag["earth_subgroups"],
        "deck_g_spatial": diag["deck_g_spatial"],
    }