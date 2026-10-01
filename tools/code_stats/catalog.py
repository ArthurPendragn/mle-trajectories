"""What a call into an ML library is: model, transformer, CV splitter, ...

``classify("sklearn.ensemble.RandomForestClassifier")`` ->
``{"name": "RandomForestClassifier", "lib": "sklearn", "kind": "model", ...}``.
Calls that are none of these (pandas, numpy, plain Python) return None. The
rules go by library and name, so a new estimator of a known family is
recognised without being listed.
"""
from __future__ import annotations

import re

KINDS = ("model", "ensemble", "transformer", "pipeline", "splitter", "search", "metric",
         "optimizer", "scheduler", "loss", "layer", "data", "training", "other")

_MODEL = re.compile(r"(Classifier|Regressor|Regression|Classification|SVC|SVR|Ridge|RidgeCV|Lasso|"
                    r"LassoCV|ElasticNet|ElasticNetCV|NB|Lars|Perceptron|IsolationForest|"
                    r"QuantileRegressor|KernelRidge|TweedieRegressor)$")
_SKLEARN_TRANSFORM_MODULES = ("preprocessing", "impute", "decomposition", "feature_extraction",
                              "feature_selection", "manifold", "random_projection",
                              "kernel_approximation", "cluster", "cross_decomposition")
_SEARCH = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"}
_CV_FUNCS = {"train_test_split", "cross_val_score", "cross_val_predict", "cross_validate"}
_METRIC = re.compile(r"(_score|_error|_loss|_curve|make_scorer|confusion_matrix|classification_report)$")


def _out(qualified: str, lib: str, kind: str, name: str | None = None) -> dict:
    return {"name": name or qualified.rsplit(".", 1)[-1], "qualified": qualified, "lib": lib,
            "kind": kind}


def classify(qualified: str) -> dict | None:
    parts = qualified.split(".")
    lib, name = parts[0], parts[-1]
    upper = name[:1].isupper()

    if lib == "sklearn":
        mod = ".".join(parts[1:-1])
        if name in _SEARCH:
            return _out(qualified, lib, "search")
        if name in _CV_FUNCS or (upper and "model_selection" in mod and re.search(r"(Fold|Split|LeaveOneOut|LeavePOut)$", name)):
            return _out(qualified, lib, "splitter")
        if "metrics" in mod and not upper and _METRIC.search(name):
            return _out(qualified, lib, "metric")
        if name in ("Pipeline", "make_pipeline", "FeatureUnion", "make_union", "ColumnTransformer",
                    "make_column_transformer", "TransformedTargetRegressor"):
            return _out(qualified, lib, "pipeline")
        if re.match(r"(Voting|Stacking)", name):
            return _out(qualified, lib, "ensemble")
        if upper and _MODEL.search(name):
            return _out(qualified, lib, "model")
        if upper and any(m in mod for m in _SKLEARN_TRANSFORM_MODULES):
            return _out(qualified, lib, "transformer")
        if upper and name not in ("BaseEstimator", "TransformerMixin", "ClassifierMixin", "RegressorMixin"):
            return _out(qualified, lib, "other")
        return None

    if lib in ("lightgbm", "xgboost", "catboost"):
        if re.match(r"(LGBM|XGB|CatBoost)", name):
            return _out(qualified, lib, "model")
        if name in ("train", "cv") and len(parts) == 2:     # functional API: lightgbm.train
            return _out(qualified, lib, "model", name=qualified)
        if name in ("Dataset", "DMatrix", "QuantileDMatrix", "Pool"):
            return _out(qualified, lib, "data")
        if name in ("early_stopping", "log_evaluation", "EarlyStopping"):
            return _out(qualified, lib, "training")
        return None

    if lib == "torch":
        if len(parts) >= 3 and parts[1] == "optim" and parts[2] == "lr_scheduler" and upper:
            return _out(qualified, lib, "scheduler")
        if len(parts) >= 3 and parts[1] == "optim" and upper:
            return _out(qualified, lib, "optimizer")
        if "utils" in parts and "data" in parts and upper:
            return _out(qualified, lib, "data")
        if len(parts) >= 3 and parts[1] == "nn" and upper and parts[2] != "functional":
            if name.endswith("Loss"):
                return _out(qualified, lib, "loss")
            if name in ("Module", "Parameter", "ModuleList", "ModuleDict", "Sequential"):
                return None
            return _out(qualified, lib, "layer")
        return None

    if lib in ("tabpfn", "pytorch_tabnet", "autogluon", "flaml", "interpret", "ngboost"):
        return _out(qualified, lib, "model") if upper else None
    if lib == "timm" and name == "create_model":
        return _out(qualified, lib, "model", name=qualified)
    if lib == "transformers" and upper:
        return _out(qualified, lib, "transformer" if "Tokenizer" in name or "Processor" in name else "model")
    if lib == "optuna" and name in ("create_study", "optimize"):
        return _out(qualified, lib, "search")
    if lib in ("skrub", "category_encoders", "imblearn", "feature_engine") and upper:
        if lib == "imblearn" and name == "Pipeline":
            return _out(qualified, lib, "pipeline")
        return _out(qualified, lib, "transformer")
    if lib == "statsmodels" and upper:
        return _out(qualified, lib, "model")
    return None
