import skrub

def build():
    return {
        "availability": skrub.as_data_op({
            "status": "Requested artifact-only analysis cannot be performed under the plan contract.",
            "reason": "Plans may read task sources only, not predictions or other artifacts written by earlier steps. Moreover, the context supplies only ambiguous basenames, not accessible candidate-specific artifact paths.",
            "unavailable_oof": [
                "expansion_378c44d400f7_variant_1",
                "expansion_c17b312b660b_variant_1",
                "expansion_bdd46429c253_variant_1",
                "expansion_dc4e7add9cb1_variant_1"
            ],
            "alignment_verified": False,
            "target_equality_verified": False,
            "joint_coverage_verified": False,
            "residual_correlations": None,
            "equal_weight_blend_scores": None,
            "paired_blend_gains": None,
            "date_hour_weighted_diagnostics": None,
            "models_refitted": False
        }),
        "recorded_fold_rmse": skrub.as_data_op({
            "provenance": "Reported candidate scores from context; not recomputed.",
            "parent": [4.107417833577887, 3.9393424456814388, 4.3010561359141795],
            "randomized_forest": [4.2570360400841, 4.243724375508718, 4.437715956527937],
            "cuda_xgboost": [4.732506193792867, 4.337155777996736, 4.892680756631347],
            "standalone_neural_network": [4.897694064501437, 4.875391054144351, 4.999933147369817]
        }),
        "runtime_assessment": skrub.as_data_op({
            "method": "Three times the sum of reported mean fit and score times, plus an assumed 180-second shared-overhead reserve.",
            "parent_plus_forest_seconds": 3 * (
                1661.8391840457916 + 6.1241508324941 +
                606.0075220266978 + 6.073107798894246
            ) + 180,
            "parent_plus_xgboost_seconds": 3 * (
                1661.8391840457916 + 6.1241508324941 +
                601.8409351507822 + 6.777702252070109
            ) + 180,
            "parent_plus_neural_network_seconds": 3 * (
                1661.8391840457916 + 6.1241508324941 +
                331.05766717592877 + 6.089345455169678
            ) + 180,
            "timeout_seconds": 7200,
            "limitation": "These projections do not establish complete feature-building costs or guarantee timeout compliance.",
            "provisional_next_experiment": "Predetermined equal-weight parent/randomized-forest ensemble: the forest is the strongest reported distinct family. Its complementarity remains unmeasured and its runtime margin is narrow.",
            "interpretation": "No ensemble improvement is established. Any subsequent blend comparison on these folds is selection evidence, not independent confirmation; do not optimize weights on validation labels."
        })
    }