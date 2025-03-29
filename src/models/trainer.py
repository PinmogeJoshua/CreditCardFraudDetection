import lightgbm as lgb
from imblearn.over_sampling import SMOTE

def train_model(X_train, y_train, X_test, y_test):
    """训练模型"""
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
    test_data = lgb.Dataset(X_test, label=y_test)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        
        'min_data_in_leaf': 20,
        'min_child_weight': 0.02,
        'max_bin': 255,
        'verbosity': -1,
        'early_stopping_round': 50
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        num_boost_round=500
    )
    return model

def save_model(model, filepath):
    """保存模型"""
    import joblib
    joblib.dump(model, filepath)
    print(f"模型已保存为 {filepath}")