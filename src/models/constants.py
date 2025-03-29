# 您notebook 5.1.1节的原始参数（完全未改动）
BASE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt", 
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "min_data_in_leaf": 20,
    "min_child_weight": 0.02,
    "max_bin": 255,
    "verbosity": -1
}