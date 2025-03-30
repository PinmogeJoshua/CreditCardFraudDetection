import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, auc

def auprc_metric(y_true, y_pred):
    """
    自定义 AUPRC 评估函数，用于 LightGBM。
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率

    返回:
        (str, float, bool): 评估指标名称、AUPRC 值、是否越高越好
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
    return 'auprc', auprc, True

def train_model(X_train, y_train, X_test, y_test):
    """训练模型"""
    # 使用 SMOTE 进行过采样，解决数据不平衡问题
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 创建 LightGBM 数据集
    train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
    test_data = lgb.Dataset(X_test, label=y_test)

    # 参数设置
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        
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

    # 训练 LightGBM 模型, 使用自定义 AUPRC 评估函数
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        num_boost_round=500,
        early_stopping_rounds=50,
        feval=auprc_metric,
        verbose_eval=50
    )
    return model

def save_model(model, filepath):
    """保存模型"""
    import joblib
    joblib.dump(model, filepath)
    print(f"模型已保存为 {filepath}")
