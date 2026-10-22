import skrub

def build():
    return {
        "diagnostic_status": skrub.as_data_op({
            "status": "blocked_by_source_contract",
            "reason": (
                "The requested retained OOF tables are generated outputs, not task "
                "sources. The plan contract prohibits reading earlier predictions "
                "or other generated files. The supplied context contains only a "
                "ten-row baseline preview, not aligned predictions for these models."
            ),
            "not_computed": [
                "OOF identity, target and fold alignment",
                "Residual correlations",
                "Paired row-level squared-error differences",
                "Largest-error rows",
                "Cross-fitted convex weights",
                "Blend RMSE, MAE and error-tail concentration"
            ],
            "models_refitted": False,
            "evaluation_changed": False,
            "required_capability": (
                "An explicitly authorized harness-side diagnostic over retained "
                "OOF tables, outside this source-only plan contract."
            )
        }),
        "available_aggregate_evidence": skrub.as_data_op({
            "provenance": "Candidate summaries supplied in context; not recomputed.",
            "xgboost_parent": {
                "fold_rmse": [3.755154762661499, 4.704375848697228, 3.6044352905672294],
                "mean_fit_seconds": 359.99519832928974
            },
            "lightgbm": {
                "fold_rmse": [3.826301673203035, 4.782735008863314, 3.6488358495636586],
                "mean_fit_seconds": 324.9539451599121
            },
            "catboost_rmse": {
                "fold_rmse": [3.993730848217744, 4.877794698815598, 3.759756172966397],
                "mean_fit_seconds": 145.14900477727255
            },
            "neural": {
                "fold_rmse": [4.286160574717966, 5.174731598132811, 4.118796825999334],
                "mean_fit_seconds": 241.69694709777832
            },
            "interpretation": (
                "Marginal fold errors cannot identify residual covariance or "
                "establish that any blend improves predictions."
            )
        }),
        "runtime_and_next_step": skrub.as_data_op({
            "three_fold_fit_seconds_additive_estimate": {
                "xgboost_lightgbm": 2054.8474304676056,
                "all_four_families": 3215.385286092758
            },
            "runtime_caveat": (
                "These sums use reported mean fit times, exclude preparation, "
                "scoring and weight estimation, and are not measured ensemble "
                "runtimes. Both exceed the 1800-second attempt limit."
            ),
            "recommendation": (
                "Retain the selected single model until authorized OOF diagnostics "
                "demonstrate complementary errors. No deployable blend is established."
            ),
            "training_only_weight_strategy": (
                "For a future ensemble candidate, reserve date-hour groups within "
                "each outer training fold, fit base learners on the remaining "
                "training groups, and estimate nonnegative sum-to-one weights by "
                "minimizing squared error on that internal holdout. Refit base "
                "learners on the entire outer training fold with weights fixed. "
                "Outer validation labels must never determine weights. Budget "
                "internal fits and full refits before proposing this candidate."
            )
        })
    }