BASE_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss', 'aucpr'],
    'learning_rate': 0.05,
    'num_leaves': 15,
    'max_depth': -1,
    'min_child_samples': 20,
    'min_data_in_leaf': 20,  # 对于不平衡数据很重要
    'max_bin': 255,
    
    'feature_fraction': 0.8,  # 特征抽样
    'bagging_fraction': 0.8,  # 数据抽样
    
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    
    'seed': 42,
    'verbose': -1
}
