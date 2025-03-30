BASE_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss', 'aucpr'],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 100,
    'max_bin': 255,
    
    'min_data_in_leaf': 20,  # 对于不平衡数据很重要
    'seed': 42,
    
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    
    'nthread': 8,
    'verbose': -1
}
