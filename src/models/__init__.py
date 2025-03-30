from .constants import BASE_PARAMS
ORIGINAL_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 7,
    'max_depth': 4,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'scale_pos_weight': 150,
    'nthread': 8,
    'verbose': -1
}
assert BASE_PARAMS == ORIGINAL_PARAMS, "参数被修改！"